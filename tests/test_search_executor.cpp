#include <gtest/gtest.h>

#include <atomic>
#include <chrono>
#include <condition_variable>
#include <mutex>

#include "fast_vector/flat_index.h"
#include "fast_vector/hnsw_index.h"
#include "fast_vector/search_executor.h"

namespace {
using namespace std::chrono_literals;
using namespace fast_vector;

struct Gate {
  std::mutex mutex;
  std::condition_variable changed;
  std::atomic<unsigned> calls{0};
  bool released = false;
  bool wait_for_calls(unsigned count) {
    std::unique_lock lock(mutex);
    return changed.wait_for(lock, 5s, [&] { return calls.load() >= count; });
  }
  void release() {
    {
      std::lock_guard lock(mutex);
      released = true;
    }
    changed.notify_all();
  }
};

struct ReleaseOnExit {
  std::shared_ptr<Gate> gate;
  ~ReleaseOnExit() { gate->release(); }
};

class ControlledIndex final : public VectorIndex {
 public:
  explicit ControlledIndex(std::shared_ptr<Gate> gate) : gate_(std::move(gate)) {}
  void add(VectorId, std::span<const float> vector) override {
    if (vector.size() == 2 && vector[0] == 99) {
      {
        std::lock_guard lock(gate_->mutex);
        ++gate_->calls;
      }
      gate_->changed.notify_all();
      std::unique_lock lock(gate_->mutex);
      gate_->changed.wait(lock, [&] { return gate_->released; });
    }
  }
  std::size_t size() const noexcept override { return 3; }
  std::size_t dimension() const noexcept override { return 2; }
  std::vector<SearchResult> search(std::span<const float> query, std::size_t k) const override {
    return search(query, k, SearchContext{});
  }
  std::vector<SearchResult> search(std::span<const float> query, std::size_t,
                                   const SearchContext& context) const override {
    {
      std::lock_guard lock(gate_->mutex);
      ++gate_->calls;
    }
    gate_->changed.notify_all();
    if (query[0] < 0) throw std::runtime_error("task failure");
    if (query[0] == 1) {
      std::unique_lock lock(gate_->mutex);
      while (!gate_->released) {
        context.check();
        gate_->changed.wait_for(lock, 1ms);
      }
    }
    context.check();
    return {{static_cast<VectorId>(query[0]), 1.0F}};
  }

 private:
  std::shared_ptr<Gate> gate_;
};

std::shared_ptr<VectorStore> controlled_store(const std::shared_ptr<Gate>& gate,
                                              bool read_only = true) {
  return std::make_shared<VectorStore>(std::make_unique<ControlledIndex>(gate), 10, read_only);
}

SearchStopReason interruption(SearchTicket& ticket) {
  if (ticket.future.wait_for(5s) != std::future_status::ready) {
    ADD_FAILURE() << "task promise was not fulfilled";
    return SearchStopReason::None;
  }
  const auto outcome = ticket.future.get();
  if (outcome.error) {
    try {
      std::rethrow_exception(outcome.error);
    } catch (const SearchInterrupted& error) {
      return error.reason();
    } catch (...) {
      ADD_FAILURE() << "unexpected exception type";
    }
  } else
    ADD_FAILURE() << "expected interrupted task";
  return SearchStopReason::None;
}

TEST(SearchExecutorTest, RejectsInvalidConfiguration) {
  auto gate = std::make_shared<Gate>();
  auto store = controlled_store(gate);
  EXPECT_THROW((void)SearchExecutor(nullptr), std::invalid_argument);
  EXPECT_THROW((void)SearchExecutor(store, {0, 1, 1}), std::invalid_argument);
  EXPECT_THROW((void)SearchExecutor(store, {1, 0, 1}), std::invalid_argument);
  EXPECT_THROW((void)SearchExecutor(store, {1, 1, 0}), std::invalid_argument);
}

TEST(SearchExecutorTest, BoundsQueueAndOwnsQueuedQuery) {
  auto gate = std::make_shared<Gate>();
  SearchExecutor executor(controlled_store(gate), {1, 1, 1});
  ReleaseOnExit release{gate};
  auto first = executor.submit({1, 0}, 1);
  ASSERT_TRUE(gate->wait_for_calls(1));
  std::vector<float> query{2, 0};
  auto waiting = executor.submit(query, 1);
  query[0] = 99;
  try {
    (void)executor.submit({3, 0}, 1);
    FAIL() << "full queue admitted task";
  } catch (const SearchRejected& error) {
    EXPECT_FALSE(error.stopping());
  }
  const auto busy = executor.stats();
  EXPECT_EQ(busy.running, 1U);
  EXPECT_EQ(busy.queued, 1U);
  gate->release();
  EXPECT_EQ(first.future.get().result.neighbors[0].id, 1U);
  EXPECT_EQ(waiting.future.get().result.neighbors[0].id, 2U);
  executor.shutdown();
  const auto stats = executor.stats();
  EXPECT_EQ(stats.peak_running, 1U);
  EXPECT_EQ(stats.peak_queued, 1U);
  EXPECT_EQ(stats.admitted, stats.completed);
  EXPECT_EQ(stats.completed, 2U);
  EXPECT_EQ(stats.rejected, 1U);
}

TEST(SearchExecutorTest, CancelledWaitingTaskNeverInvokesIndexAndFreesCapacity) {
  auto gate = std::make_shared<Gate>();
  SearchExecutor executor(controlled_store(gate), {1, 1, 1});
  ReleaseOnExit release{gate};
  auto first = executor.submit({1, 0}, 1);
  ASSERT_TRUE(gate->wait_for_calls(1));
  auto waiting = executor.submit({2, 0}, 1);
  executor.cancel(waiting.cancellation, SearchStopReason::Cancelled);
  EXPECT_EQ(interruption(waiting), SearchStopReason::Cancelled);
  auto next = executor.submit({3, 0}, 1);
  EXPECT_EQ(gate->calls.load(), 1U);
  gate->release();
  (void)first.future.get();
  EXPECT_EQ(next.future.get().result.neighbors[0].id, 3U);
  EXPECT_EQ(gate->calls.load(), 2U);
  EXPECT_EQ(executor.stats().cancelled, 1U);
}

TEST(SearchExecutorTest, ExpiredWaitingTaskNeverInvokesIndex) {
  auto gate = std::make_shared<Gate>();
  SearchExecutor executor(controlled_store(gate), {1, 1, 1});
  ReleaseOnExit release{gate};
  auto first = executor.submit({1, 0}, 1);
  ASSERT_TRUE(gate->wait_for_calls(1));
  auto expired = executor.submit({2, 0}, 1, SearchContext::Clock::now() - 1ms);
  gate->release();
  (void)first.future.get();
  EXPECT_EQ(interruption(expired), SearchStopReason::DeadlineExceeded);
  EXPECT_EQ(gate->calls.load(), 1U);
  EXPECT_EQ(executor.stats().expired, 1U);
  EXPECT_EQ(executor.stats().started, 1U);
}

TEST(SearchExecutorTest, RunningCancellationAndExceptionsDoNotKillWorker) {
  auto gate = std::make_shared<Gate>();
  SearchExecutor executor(controlled_store(gate), {1, 2, 1});
  ReleaseOnExit release{gate};
  auto first = executor.submit({1, 0}, 1);
  ASSERT_TRUE(gate->wait_for_calls(1));
  executor.cancel(first.cancellation, SearchStopReason::Cancelled);
  EXPECT_EQ(interruption(first), SearchStopReason::Cancelled);
  auto failing = executor.submit({-1, 0}, 1);
  auto failure = failing.future.get();
  ASSERT_NE(failure.error, nullptr);
  EXPECT_THROW(std::rethrow_exception(failure.error), std::runtime_error);
  auto recovery = executor.submit({3, 0}, 1);
  EXPECT_EQ(recovery.future.get().result.neighbors[0].id, 3U);
  EXPECT_EQ(executor.stats().failed, 1U);
  EXPECT_EQ(executor.stats().succeeded, 1U);
}

TEST(SearchExecutorTest, ShutdownSettlesEveryTaskAndRejectsFurtherAdmission) {
  auto gate = std::make_shared<Gate>();
  SearchExecutor executor(controlled_store(gate), {1, 2, 1});
  ReleaseOnExit release{gate};
  auto first = executor.submit({1, 0}, 1);
  ASSERT_TRUE(gate->wait_for_calls(1));
  auto second = executor.submit({2, 0}, 1);
  auto third = executor.submit({3, 0}, 1);
  executor.shutdown();
  EXPECT_EQ(interruption(first), SearchStopReason::Shutdown);
  EXPECT_EQ(interruption(second), SearchStopReason::Shutdown);
  EXPECT_EQ(interruption(third), SearchStopReason::Shutdown);
  const auto stats = executor.stats();
  EXPECT_EQ(stats.running, 0U);
  EXPECT_EQ(stats.queued, 0U);
  EXPECT_EQ(stats.cancelled, 3U);
  EXPECT_EQ(stats.completed, stats.admitted);
  try {
    (void)executor.submit({3, 0}, 1);
    FAIL();
  } catch (const SearchRejected& error) {
    EXPECT_TRUE(error.stopping());
  }
  executor.shutdown();
}

TEST(SearchExecutorTest, DeadlineInterruptsRunningTaskAndWorkerContinues) {
  auto gate = std::make_shared<Gate>();
  SearchExecutor executor(controlled_store(gate), {1, 1, 1});
  ReleaseOnExit release{gate};
  auto first = executor.submit({1, 0}, 1, SearchContext::Clock::now() + 50ms);
  EXPECT_EQ(interruption(first), SearchStopReason::DeadlineExceeded);
  auto next = executor.submit({2, 0}, 1);
  EXPECT_EQ(next.future.get().result.neighbors[0].id, 2U);
  EXPECT_EQ(executor.stats().expired, 1U);
}

TEST(SearchExecutorTest, DeadlineCanInterruptWritableStoreLockWait) {
  auto gate = std::make_shared<Gate>();
  auto store = controlled_store(gate, false);
  SearchExecutor executor(store, {1, 1, 1});
  auto writer = std::async(std::launch::async, [&] { store->add(99, std::vector<float>{99, 0}); });
  ReleaseOnExit release{gate};
  ASSERT_TRUE(gate->wait_for_calls(1));
  EXPECT_EQ(store->dimension(), 2U);
  EXPECT_FALSE(store->try_size().has_value());
  auto waiting = executor.submit({2, 0}, 1, SearchContext::Clock::now() + 50ms);
  EXPECT_EQ(interruption(waiting), SearchStopReason::DeadlineExceeded);
  gate->release();
  writer.get();
  auto next = executor.submit({2, 0}, 1);
  EXPECT_EQ(next.future.get().result.neighbors[0].id, 2U);
}

TEST(SearchExecutorTest, QueuedTaskAcquiresSnapshotAtExecutionNotAdmission) {
  auto gate = std::make_shared<Gate>();
  auto store = controlled_store(gate);
  SearchExecutor executor(store, {1, 1, 1});
  ReleaseOnExit release{gate};
  auto first = executor.submit({1, 0}, 1);
  ASSERT_TRUE(gate->wait_for_calls(1));
  auto waiting = executor.submit({2, 0}, 1);
  auto replacement = std::make_unique<FlatIndex>(2);
  replacement->add(20, std::vector<float>{1, 0});
  store->replace_index(std::move(replacement));
  gate->release();
  const auto old_result = first.future.get();
  const auto new_result = waiting.future.get();
  EXPECT_EQ(old_result.result.snapshot_version, 1U);
  EXPECT_EQ(new_result.result.snapshot_version, 2U);
  EXPECT_EQ(new_result.result.neighbors[0].id, 20U);
  EXPECT_EQ(store->stats().snapshot_version, 2U);
}

TEST(SearchContextTest, BatchChecksAndCancellationReasonsAreDeterministic) {
  auto cancellation = std::make_shared<SearchCancellation>();
  SearchContext context{SearchContext::Clock::time_point::max(), cancellation, 4};
  std::size_t remaining = 0;
  context.checkpoint(remaining);
  cancellation->request(SearchStopReason::Cancelled);
  for (int i = 0; i < 3; ++i) EXPECT_NO_THROW(context.checkpoint(remaining));
  cancellation->request(SearchStopReason::Shutdown);
  EXPECT_THROW(context.checkpoint(remaining), SearchInterrupted);
  EXPECT_EQ(cancellation->reason(), SearchStopReason::Cancelled);
  context.check_interval = 0;
  EXPECT_THROW(context.validate(), std::invalid_argument);
}

TEST(SearchExecutorTest, ExpectedIdentityIsCheckedAfterDequeueAndReload) {
  auto gate = std::make_shared<Gate>();
  auto store = std::make_shared<VectorStore>(std::make_unique<ControlledIndex>(gate), 10, true,
                                             SnapshotIdentity{"snapshot-a", 1, "rev-a"});
  SearchExecutor executor(store, {1, 1, 1});
  ReleaseOnExit release{gate};
  auto active = executor.submit({1, 0}, 1);
  ASSERT_TRUE(gate->wait_for_calls(1));
  auto waiting = executor.submit({2, 0}, 1, SearchContext::Clock::time_point::max(), "snapshot-a");
  auto replacement = std::make_unique<FlatIndex>(2);
  replacement->add(20, std::vector<float>{1, 0});
  store->activate_snapshot(
      store->prepare_snapshot(std::move(replacement), {"snapshot-b", 2, "rev-b"}));
  gate->release();
  const auto original = active.future.get();
  EXPECT_EQ(original.result.identity.snapshot_id, "snapshot-a");
  EXPECT_EQ(original.result.neighbors[0].id, 1U);
  const auto mismatch = waiting.future.get();
  ASSERT_NE(mismatch.error, nullptr);
  EXPECT_THROW(std::rethrow_exception(mismatch.error), SnapshotMismatchError);
  auto current = executor.submit({2, 0}, 1, SearchContext::Clock::time_point::max(), "snapshot-b");
  const auto matched = current.future.get();
  EXPECT_EQ(matched.result.neighbors[0].id, 20U);
  EXPECT_EQ(matched.result.identity.generation, 2U);
  EXPECT_EQ(matched.result.identity.embedding_revision, "rev-b");
}

TEST(SearchContextTest, BuiltinIndexesRejectExpiredAndCancelledSearchesWithoutMutation) {
  FlatIndex flat(2);
  HnswIndex hnsw({.dimension = 2, .max_connections = 2, .ef_construction = 8});
  for (auto* index : std::vector<VectorIndex*>{&flat, &hnsw}) {
    index->add(1, std::vector<float>{1, 0});
    SearchContext expired{SearchContext::Clock::now() - 1ms, {}, 1};
    EXPECT_THROW((void)index->search(std::vector<float>{1, 0}, 1, expired), SearchInterrupted);
    auto cancellation = std::make_shared<SearchCancellation>();
    cancellation->request(SearchStopReason::Cancelled);
    EXPECT_THROW((void)index->search(
                     std::vector<float>{1, 0}, 1,
                     SearchContext{SearchContext::Clock::time_point::max(), cancellation, 64}),
                 SearchInterrupted);
    EXPECT_EQ(index->search(std::vector<float>{1, 0}, 1)[0].id, 1U);
  }
}

TEST(SearchContextTest, EnabledChecksPreserveBuiltinResultsAcrossBatchSizes) {
  FlatIndex flat(2);
  HnswIndex hnsw({.dimension = 2, .max_connections = 4, .ef_construction = 16, .ef_search = 64});
  for (auto* index : std::vector<VectorIndex*>{&flat, &hnsw}) {
    for (VectorId id = 1; id <= 64; ++id) {
      index->add(
          id, std::vector<float>{static_cast<float>(id % 7 + 1), static_cast<float>(id % 11 + 1)});
    }
    const std::vector<float> query{1, 2};
    const auto expected = index->search(query, 10);
    for (const std::size_t interval : {1U, 16U, 64U, 256U}) {
      const SearchContext context{SearchContext::Clock::now() + 30s,
                                  std::make_shared<SearchCancellation>(), interval};
      EXPECT_EQ(index->search(query, 10, context), expected);
    }
  }
}
}  // namespace
