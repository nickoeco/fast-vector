#pragma once

#include <atomic>
#include <chrono>
#include <cstddef>
#include <exception>
#include <memory>
#include <stdexcept>
#include <string>

namespace fast_vector {

enum class SearchStopReason { None, Cancelled, DeadlineExceeded, Shutdown };

class SearchInterrupted final : public std::exception {
 public:
  explicit SearchInterrupted(SearchStopReason reason) : reason_(reason) {}
  [[nodiscard]] SearchStopReason reason() const noexcept { return reason_; }
  const char* what() const noexcept override { return "search interrupted"; }

 private:
  SearchStopReason reason_;
};

/** Shared independently of transport objects. The first cancellation reason wins. */
class SearchCancellation {
 public:
  void request(SearchStopReason reason) noexcept {
    auto expected = SearchStopReason::None;
    reason_.compare_exchange_strong(expected, reason, std::memory_order_relaxed);
  }
  [[nodiscard]] SearchStopReason reason() const noexcept {
    return reason_.load(std::memory_order_relaxed);
  }

 private:
  std::atomic<SearchStopReason> reason_{SearchStopReason::None};
};

/** Cooperative search budget; no transport pointers or callbacks are retained. */
struct SearchContext {
  using Clock = std::chrono::steady_clock;
  Clock::time_point deadline = Clock::time_point::max();
  std::shared_ptr<SearchCancellation> cancellation;
  std::size_t check_interval = 64;
  std::string expected_snapshot_id{};

  [[nodiscard]] bool enabled() const noexcept {
    return cancellation || deadline != Clock::time_point::max();
  }
  void check() const {
    if (cancellation && cancellation->reason() != SearchStopReason::None) {
      throw SearchInterrupted(cancellation->reason());
    }
    if (deadline != Clock::time_point::max() && Clock::now() >= deadline) {
      throw SearchInterrupted(SearchStopReason::DeadlineExceeded);
    }
  }
  /** Call for each scan item. Only the configured batch boundary reads the clock. */
  void checkpoint(std::size_t& remaining) const {
    if (!enabled()) return;
    if (remaining == 0) {
      check();
      remaining = check_interval;
    }
    --remaining;
  }
  void validate() const {
    if (check_interval == 0) throw std::invalid_argument("search check interval must be positive");
  }
};

}  // namespace fast_vector
