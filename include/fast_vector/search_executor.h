#pragma once

#include <condition_variable>
#include <cstdint>
#include <deque>
#include <future>
#include <memory>
#include <mutex>
#include <thread>
#include <vector>

#include "fast_vector/vector_store.h"

namespace fast_vector {

struct SearchExecutorConfig {
  std::size_t workers = 2;
  std::size_t queue_capacity = 64;
  std::size_t check_interval = 64;
  std::size_t queue_bytes = 64ULL * 1024 * 1024;
};

class SearchRejected final : public std::exception {
 public:
  explicit SearchRejected(bool stopping) : stopping_(stopping) {}
  [[nodiscard]] bool stopping() const noexcept { return stopping_; }
  const char* what() const noexcept override { return "search admission rejected"; }

 private:
  bool stopping_;
};

struct SearchOutcome {
  std::shared_ptr<void> memory_lease{};
  SnapshotSearchResult result;
  std::uint64_t queue_us = 0;
  std::uint64_t execution_us = 0;
  std::exception_ptr error;
};

struct SearchTicket {
  std::future<SearchOutcome> future;
  std::shared_ptr<SearchCancellation> cancellation;
};

/** Consistent scheduler counters, protected by the queue mutex. */
struct SearchExecutorStats {
  std::size_t workers = 0, queue_capacity = 0, queued = 0, running = 0;
  std::size_t peak_queued = 0, peak_running = 0;
  std::uint64_t admitted = 0, rejected = 0, started = 0, completed = 0;
  std::uint64_t succeeded = 0, failed = 0, cancelled = 0, expired = 0;
  std::uint64_t queue_us_total = 0, execution_us_total = 0;
  std::uint64_t queued_bytes = 0, peak_queued_bytes = 0, queue_byte_limit = 0;
};

/** Bounded FIFO computation scheduler. Tasks own their input; no transport objects cross threads.
 */
class SearchExecutor {
 public:
  explicit SearchExecutor(std::shared_ptr<VectorStore> store, SearchExecutorConfig config = {});
  ~SearchExecutor();
  SearchExecutor(const SearchExecutor&) = delete;
  SearchExecutor& operator=(const SearchExecutor&) = delete;

  [[nodiscard]] SearchTicket submit(
      std::vector<float> query, std::size_t k,
      SearchContext::Clock::time_point deadline = SearchContext::Clock::time_point::max(),
      std::string expected_snapshot_id = {});
  /** Remove a waiting task immediately, or signal an active task's shared state. */
  void cancel(const std::shared_ptr<SearchCancellation>& cancellation, SearchStopReason reason);
  /** Stop admission and notify/cancel tasks without joining; safe before gRPC shutdown. */
  void stop();
  /** Stop, then join every worker. Idempotent; call from an owning thread, not a worker. */
  void shutdown();
  [[nodiscard]] SearchExecutorStats stats() const;

 private:
  struct SearchTask {
    std::vector<float> query;
    std::size_t k;
    SearchContext context;
    SearchContext::Clock::time_point enqueued;
    std::promise<SearchOutcome> promise;
    std::size_t bytes = 0;
    std::shared_ptr<void> memory_lease;
  };
  void run(std::size_t worker);
  void finish_locked(SearchTask& task, SearchOutcome outcome);
  void cancel_waiting_locked(SearchTask& task);

  std::shared_ptr<VectorStore> store_;
  SearchExecutorConfig config_;
  mutable std::mutex mutex_;
  std::mutex shutdown_mutex_;
  std::condition_variable ready_;
  bool stopping_ = false;
  std::deque<std::unique_ptr<SearchTask>> queue_;
  std::vector<std::shared_ptr<SearchCancellation>> active_;
  std::vector<std::thread> workers_;
  SearchExecutorStats stats_;
};

}  // namespace fast_vector
