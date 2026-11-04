#include "fast_vector/search_executor.h"

#include <algorithm>
#include <utility>

namespace fast_vector {
namespace {
std::uint64_t microseconds(SearchContext::Clock::duration duration) {
  return static_cast<std::uint64_t>(
      std::chrono::duration_cast<std::chrono::microseconds>(duration).count());
}
}  // namespace

SearchExecutor::SearchExecutor(std::shared_ptr<VectorStore> store, SearchExecutorConfig config)
    : store_(std::move(store)), config_(config) {
  if (!store_ || config.workers == 0 || config.queue_capacity == 0 || config.check_interval == 0) {
    throw std::invalid_argument("search executor requires a store and positive limits");
  }
  stats_.workers = config.workers;
  stats_.queue_capacity = config.queue_capacity;
  active_.resize(config.workers);
  workers_.reserve(config.workers);
  try {
    for (std::size_t i = 0; i < config.workers; ++i) workers_.emplace_back([this, i] { run(i); });
  } catch (...) {
    stop();
    for (auto& worker : workers_) worker.join();
    throw;
  }
}

SearchExecutor::~SearchExecutor() { shutdown(); }

SearchTicket SearchExecutor::submit(std::vector<float> query, std::size_t k,
                                    SearchContext::Clock::time_point deadline) {
  auto task = std::make_unique<SearchTask>();
  task->query = std::move(query);
  task->k = k;
  task->context = {deadline, std::make_shared<SearchCancellation>(), config_.check_interval};
  auto future = task->promise.get_future();
  auto cancellation = task->context.cancellation;
  {
    std::lock_guard lock(mutex_);
    if (stopping_ || queue_.size() >= config_.queue_capacity) {
      ++stats_.rejected;
      throw SearchRejected(stopping_);
    }
    task->enqueued = SearchContext::Clock::now();
    queue_.push_back(std::move(task));
    ++stats_.admitted;
    stats_.queued = queue_.size();
    stats_.peak_queued = std::max(stats_.peak_queued, stats_.queued);
  }
  ready_.notify_one();
  return {std::move(future), std::move(cancellation)};
}

void SearchExecutor::finish_locked(SearchTask& task, SearchOutcome outcome) {
  ++stats_.completed;
  stats_.queue_us_total += outcome.queue_us;
  stats_.execution_us_total += outcome.execution_us;
  if (!outcome.error) {
    ++stats_.succeeded;
  } else {
    try {
      std::rethrow_exception(outcome.error);
    } catch (const SearchInterrupted& error) {
      if (error.reason() == SearchStopReason::DeadlineExceeded)
        ++stats_.expired;
      else
        ++stats_.cancelled;
    } catch (...) {
      ++stats_.failed;
    }
  }
  task.promise.set_value(std::move(outcome));
}

void SearchExecutor::cancel_waiting_locked(SearchTask& task) {
  SearchOutcome outcome{};
  outcome.queue_us = microseconds(SearchContext::Clock::now() - task.enqueued);
  try {
    task.context.check();
  } catch (...) {
    outcome.error = std::current_exception();
  }
  finish_locked(task, std::move(outcome));
}

void SearchExecutor::cancel(const std::shared_ptr<SearchCancellation>& cancellation,
                            SearchStopReason reason) {
  if (!cancellation || reason == SearchStopReason::None) {
    throw std::invalid_argument("task cancellation requires a state and stop reason");
  }
  cancellation->request(reason);
  std::lock_guard lock(mutex_);
  const auto found = std::find_if(queue_.begin(), queue_.end(), [&](const auto& task) {
    return task->context.cancellation == cancellation;
  });
  if (found != queue_.end()) {
    auto task = std::move(*found);
    queue_.erase(found);
    stats_.queued = queue_.size();
    cancel_waiting_locked(*task);
  }
}

void SearchExecutor::stop() {
  {
    std::lock_guard lock(mutex_);
    if (stopping_) return;
    stopping_ = true;
    for (auto& cancellation : active_) {
      if (cancellation) cancellation->request(SearchStopReason::Shutdown);
    }
    // Promise fulfillment does not run user continuations inline.
    while (!queue_.empty()) {
      auto task = std::move(queue_.front());
      queue_.pop_front();
      stats_.queued = queue_.size();
      task->context.cancellation->request(SearchStopReason::Shutdown);
      cancel_waiting_locked(*task);
    }
  }
  ready_.notify_all();
}

void SearchExecutor::shutdown() {
  std::lock_guard shutdown_lock(shutdown_mutex_);
  stop();
  for (auto& worker : workers_)
    if (worker.joinable()) worker.join();
}

SearchExecutorStats SearchExecutor::stats() const {
  std::lock_guard lock(mutex_);
  return stats_;
}

void SearchExecutor::run(std::size_t worker) {
  while (true) {
    std::unique_ptr<SearchTask> task;
    {
      std::unique_lock lock(mutex_);
      ready_.wait(lock, [this] { return stopping_ || !queue_.empty(); });
      if (queue_.empty()) return;
      task = std::move(queue_.front());
      queue_.pop_front();
      stats_.queued = queue_.size();
      active_[worker] = task->context.cancellation;
      ++stats_.running;
      stats_.peak_running = std::max(stats_.peak_running, stats_.running);
    }
    SearchOutcome outcome{};
    const auto started = SearchContext::Clock::now();
    outcome.queue_us = microseconds(started - task->enqueued);
    bool executed = false;
    try {
      task->context.check();
      {
        std::lock_guard lock(mutex_);
        ++stats_.started;
      }
      executed = true;
      // Snapshot acquisition is inside the store call, after dequeue and cancellation checks.
      outcome.result = store_->search(task->query, task->k, task->context);
    } catch (...) {
      outcome.error = std::current_exception();
    }
    outcome.execution_us = executed ? microseconds(SearchContext::Clock::now() - started) : 0;
    {
      std::lock_guard lock(mutex_);
      active_[worker].reset();
      --stats_.running;
      finish_locked(*task, std::move(outcome));
    }
  }
}

}  // namespace fast_vector
