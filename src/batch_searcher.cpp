#include "fast_vector/batch_searcher.h"

#include <condition_variable>
#include <cstddef>
#include <exception>
#include <functional>
#include <future>
#include <limits>
#include <memory>
#include <mutex>
#include <queue>
#include <stdexcept>
#include <thread>
#include <utility>
#include <vector>

namespace fast_vector {

class BatchSearcher::Impl {
public:
    Impl(const VectorIndex& index, const std::size_t thread_count) : index_(index) {
        if (thread_count == 0) {
            throw std::invalid_argument("batch search thread count must be greater than zero");
        }
        workers_.reserve(thread_count);
        try {
            for (std::size_t i = 0; i < thread_count; ++i) {
                workers_.emplace_back([this] { worker_loop(); });
            }
        } catch (...) {
            {
                std::lock_guard lock(mutex_);
                stopping_ = true;
            }
            work_available_.notify_all();
            for (std::thread& worker : workers_) {
                worker.join();
            }
            throw;
        }
    }

    ~Impl() {
        {
            std::lock_guard lock(mutex_);
            stopping_ = true;
        }
        work_available_.notify_all();
        for (std::thread& worker : workers_) {
            worker.join();
        }
    }

    [[nodiscard]] std::vector<std::vector<SearchResult>> search(
        const std::span<const float> queries, const std::size_t query_count,
        const std::size_t k) const {
        const std::size_t dimension = index_.dimension();
        if (query_count != 0 && dimension > std::numeric_limits<std::size_t>::max() / query_count) {
            throw std::invalid_argument("batch query shape overflows");
        }
        if (queries.size() != query_count * dimension) {
            throw std::invalid_argument("batch query data does not match query_count and dimension");
        }

        std::vector<std::vector<SearchResult>> results(query_count);
        if (query_count == 0 || k == 0) {
            return results;
        }

        std::vector<std::future<void>> completions;
        completions.reserve(query_count);
        for (std::size_t query_index = 0; query_index < query_count; ++query_index) {
            completions.push_back(submit([&, query_index] {
                const std::span<const float> query(
                    queries.data() + query_index * dimension, dimension);
                results[query_index] = index_.search(query, k);
            }));
        }

        std::exception_ptr first_error;
        for (std::future<void>& completion : completions) {
            try {
                completion.get();
            } catch (...) {
                if (!first_error) {
                    first_error = std::current_exception();
                }
            }
        }
        if (first_error) {
            std::rethrow_exception(first_error);
        }
        return results;
    }

    [[nodiscard]] std::size_t thread_count() const noexcept { return workers_.size(); }

private:
    std::future<void> submit(std::function<void()> work) const {
        auto task = std::make_shared<std::packaged_task<void()>>(std::move(work));
        std::future<void> completion = task->get_future();
        {
            std::lock_guard lock(mutex_);
            if (stopping_) {
                throw std::runtime_error("batch searcher is stopping");
            }
            work_queue_.push([task] { (*task)(); });
        }
        work_available_.notify_one();
        return completion;
    }

    void worker_loop() {
        while (true) {
            std::function<void()> work;
            {
                std::unique_lock lock(mutex_);
                work_available_.wait(lock, [this] { return stopping_ || !work_queue_.empty(); });
                if (stopping_ && work_queue_.empty()) {
                    return;
                }
                work = std::move(work_queue_.front());
                work_queue_.pop();
            }
            work();
        }
    }

    const VectorIndex& index_;
    mutable std::mutex mutex_;
    mutable std::condition_variable work_available_;
    mutable std::queue<std::function<void()>> work_queue_;
    std::vector<std::thread> workers_;
    mutable bool stopping_ = false;
};

BatchSearcher::BatchSearcher(const VectorIndex& index, const std::size_t thread_count)
    : impl_(std::make_unique<Impl>(index, thread_count)) {}

BatchSearcher::~BatchSearcher() = default;

std::vector<std::vector<SearchResult>> BatchSearcher::search(
    const std::span<const float> queries, const std::size_t query_count,
    const std::size_t k) const {
    return impl_->search(queries, query_count, k);
}

std::size_t BatchSearcher::thread_count() const noexcept {
    return impl_->thread_count();
}

}  // namespace fast_vector
