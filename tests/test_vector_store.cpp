#include <gtest/gtest.h>

#include <atomic>
#include <chrono>
#include <future>
#include <limits>
#include <memory>
#include <stdexcept>
#include <thread>
#include <vector>

#include "fast_vector/flat_index.h"
#include "fast_vector/hnsw_index.h"
#include "fast_vector/vector_store.h"

namespace {

std::unique_ptr<fast_vector::VectorIndex> make_index(const std::size_t dimension = 2) {
  return std::make_unique<fast_vector::FlatIndex>(dimension);
}

TEST(VectorStoreTest, RejectsInvalidConstruction) {
  EXPECT_THROW((void)fast_vector::VectorStore(nullptr), std::invalid_argument);
  EXPECT_THROW((void)fast_vector::VectorStore(make_index(), 0), std::invalid_argument);
}

TEST(VectorStoreTest, AddsSearchesAndReportsStats) {
  fast_vector::VectorStore store(make_index());
  store.add(10, std::vector<float>{1.0F, 0.0F});
  store.add_batch(std::vector<fast_vector::VectorRecord>{
      {20, {0.0F, 1.0F}},
      {30, {-1.0F, 0.0F}},
  });

  const auto results = store.search(std::vector<float>{1.0F, 0.0F}, 2);
  ASSERT_EQ(results.size(), 2U);
  EXPECT_EQ(results[0].id, 10U);
  EXPECT_EQ(results[1].id, 20U);

  const fast_vector::StoreStats stats = store.stats();
  EXPECT_EQ(stats.index_size, 3U);
  EXPECT_EQ(stats.dimension, 2U);
  EXPECT_EQ(stats.successful_queries, 1U);
  EXPECT_EQ(stats.failed_queries, 0U);
  EXPECT_EQ(stats.inserted_vectors, 3U);
  EXPECT_FALSE(stats.read_only);
  EXPECT_EQ(store.maximum_batch_size(), 1'000U);
  EXPECT_FALSE(store.read_only());
}

TEST(VectorStoreTest, ReadOnlyStoreRejectsWritesAndStillSearches) {
  auto index = make_index();
  index->add(10, std::vector<float>{1.0F, 0.0F});
  fast_vector::VectorStore store(std::move(index), 10, true);

  EXPECT_THROW(store.add(20, std::vector<float>{0.0F, 1.0F}), fast_vector::ReadOnlyStoreError);
  EXPECT_THROW(store.add_batch(std::vector<fast_vector::VectorRecord>{{20, {0.0F, 1.0F}}}),
               fast_vector::ReadOnlyStoreError);

  const auto results = store.search(std::vector<float>{1.0F, 0.0F}, 1);
  ASSERT_EQ(results.size(), 1U);
  EXPECT_EQ(results[0].id, 10U);
  const fast_vector::StoreStats stats = store.stats();
  EXPECT_EQ(stats.index_size, 1U);
  EXPECT_EQ(stats.inserted_vectors, 0U);
  EXPECT_TRUE(stats.read_only);
  EXPECT_TRUE(store.read_only());
}

TEST(VectorStoreTest, ReadOnlyStoreAtomicallyReplacesItsIndex) {
  auto original = make_index();
  original->add(10, std::vector<float>{1.0F, 0.0F});
  fast_vector::VectorStore store(std::move(original), 10, true);

  auto replacement = make_index(3);
  replacement->add(20, std::vector<float>{0.0F, 0.0F, 1.0F});
  store.replace_index(std::move(replacement));

  const auto results = store.search(std::vector<float>{0.0F, 0.0F, 1.0F}, 1);
  ASSERT_EQ(results.size(), 1U);
  EXPECT_EQ(results[0].id, 20U);
  EXPECT_EQ(store.stats().dimension, 3U);
  EXPECT_EQ(store.stats().inserted_vectors, 0U);
}

TEST(VectorStoreTest, RejectsInvalidIndexReplacement) {
  fast_vector::VectorStore writable(make_index());
  EXPECT_THROW(writable.replace_index(make_index()), std::logic_error);

  fast_vector::VectorStore read_only(make_index(), 10, true);
  EXPECT_THROW(read_only.replace_index(nullptr), std::invalid_argument);
  EXPECT_EQ(read_only.stats().dimension, 2U);
}

struct SearchGate {
  std::promise<void> entered;
  std::promise<void> release;
  std::shared_future<void> released = release.get_future().share();
  std::atomic<bool> destroyed{false};
};

class BlockingSnapshot final : public fast_vector::VectorIndex {
 public:
  explicit BlockingSnapshot(std::shared_ptr<SearchGate> gate) : gate_(std::move(gate)) {}
  ~BlockingSnapshot() override { gate_->destroyed.store(true); }
  void add(fast_vector::VectorId, std::span<const float>) override {
    throw std::logic_error("immutable test snapshot");
  }
  std::vector<fast_vector::SearchResult> search(std::span<const float>,
                                                std::size_t) const override {
    gate_->entered.set_value();
    gate_->released.wait();
    return {{10, 1.0F}};
  }
  std::size_t size() const noexcept override { return 1; }
  std::size_t dimension() const noexcept override { return 2; }

 private:
  std::shared_ptr<SearchGate> gate_;
};

TEST(VectorStoreTest, PublicationDoesNotWaitForOldSearchAndRetainsItsSnapshot) {
  using namespace std::chrono_literals;
  auto gate = std::make_shared<SearchGate>();
  auto entered = gate->entered.get_future();
  fast_vector::VectorStore store(std::make_unique<BlockingSnapshot>(gate), 10, true);
  auto reader = std::async(std::launch::async,
                           [&] { return store.search(std::vector<float>{1.0F, 0.0F}, 1); });
  const auto started = entered.wait_for(5s);
  if (started != std::future_status::ready) {
    gate->release.set_value();
    FAIL() << "query did not enter the old snapshot";
  }

  auto replacement = make_index(3);
  replacement->add(20, std::vector<float>{0.0F, 0.0F, 1.0F});
  auto publisher =
      std::async(std::launch::async, [&store, index = std::move(replacement)]() mutable {
        store.replace_index(std::move(index));
      });
  const auto published = publisher.wait_for(5s);
  EXPECT_EQ(published, std::future_status::ready);
  EXPECT_FALSE(gate->destroyed.load());
  if (published == std::future_status::ready) {
    publisher.get();
    const auto stats = store.stats();
    EXPECT_EQ(stats.dimension, 3U);
    const auto current = store.search(std::vector<float>{0.0F, 0.0F, 1.0F}, 1);
    EXPECT_EQ(current.front().id, 20U);
  }
  gate->release.set_value();
  const auto original = reader.get();
  if (publisher.valid()) {
    publisher.get();
  }
  EXPECT_EQ(original.front().id, 10U);
  EXPECT_TRUE(gate->destroyed.load());
}

std::unique_ptr<fast_vector::VectorIndex> make_snapshot(const bool hnsw,
                                                        const fast_vector::VectorId first_id,
                                                        const std::size_t dimension = 2) {
  std::unique_ptr<fast_vector::VectorIndex> index;
  if (hnsw) {
    index = std::make_unique<fast_vector::HnswIndex>(fast_vector::HnswConfig{
        .dimension = dimension, .max_connections = 4, .ef_construction = 16, .ef_search = 16});
  } else {
    index = make_index(dimension);
  }
  for (std::size_t row = 0; row < dimension; ++row) {
    index->add(first_id + row, std::vector<float>(dimension, 1.0F));
  }
  return index;
}

TEST(VectorStoreTest, ConcurrentQueriesUseOneFlatOrHnswSnapshotPerCall) {
  fast_vector::VectorStore store(make_snapshot(false, 10), 10, true);
  std::atomic<bool> start{false};
  std::atomic<std::size_t> invalid{0};
  std::vector<std::jthread> readers;
  for (int i = 0; i < 4; ++i) {
    readers.emplace_back([&] {
      while (!start.load(std::memory_order_acquire)) {
        std::this_thread::yield();
      }
      for (int query = 0; query < 500; ++query) {
        try {
          const auto result = store.search(std::vector<float>{1.0F, 1.0F}, 2);
          if (result.size() != 2 || (result[0].id != 10 && result[0].id != 20) ||
              result[1].id != result[0].id + 1) {
            invalid.fetch_add(1);
          }
        } catch (...) {
          invalid.fetch_add(1);
        }
      }
    });
  }
  start.store(true, std::memory_order_release);
  for (int generation = 0; generation < 100; ++generation) {
    const bool hnsw = generation % 2 == 0;
    store.replace_index(make_snapshot(hnsw, hnsw ? 20 : 10));
  }
  readers.clear();
  EXPECT_EQ(invalid.load(), 0U);
  EXPECT_EQ(store.stats().successful_queries, 2'000U);
  EXPECT_EQ(store.stats().failed_queries, 0U);
  EXPECT_EQ(store.stats().inserted_vectors, 0U);
}

TEST(VectorStoreTest, ConcurrentStatsObserveSizeAndDimensionFromOneSnapshot) {
  fast_vector::VectorStore store(make_snapshot(false, 10), 10, true);
  std::atomic<std::size_t> invalid{0};
  std::jthread reader([&] {
    for (int i = 0; i < 2'000; ++i) {
      const auto stats = store.stats();
      if (stats.index_size != stats.dimension || (stats.dimension != 2 && stats.dimension != 3)) {
        invalid.fetch_add(1);
      }
    }
  });
  for (int generation = 0; generation < 100; ++generation) {
    const bool hnsw = generation % 2 == 0;
    store.replace_index(make_snapshot(hnsw, 10, hnsw ? 3 : 2));
  }
  reader.join();
  EXPECT_EQ(invalid.load(), 0U);
}

TEST(VectorStoreTest, ReadOnlyCountersSurviveReplacementAndQueryFailure) {
  fast_vector::VectorStore store(make_snapshot(true, 10), 10, true);
  EXPECT_THROW((void)store.search(std::vector<float>{1.0F}, 1), std::invalid_argument);
  EXPECT_TRUE(store.search({}, 0).empty());
  store.replace_index(make_snapshot(false, 20));
  EXPECT_THROW(store.replace_index(nullptr), std::invalid_argument);
  EXPECT_EQ(store.search(std::vector<float>{1.0F, 1.0F}, 1).front().id, 20U);
  const auto stats = store.stats();
  EXPECT_EQ(stats.successful_queries, 2U);
  EXPECT_EQ(stats.failed_queries, 1U);
  EXPECT_EQ(stats.inserted_vectors, 0U);
}

TEST(VectorStoreTest, ReadOnlyStoreAcceptsEmptyFlatAndHnswReplacements) {
  fast_vector::VectorStore store(make_snapshot(false, 10), 10, true);
  store.replace_index(std::make_unique<fast_vector::HnswIndex>(
      fast_vector::HnswConfig{.dimension = 3, .max_connections = 2, .ef_construction = 8}));
  EXPECT_EQ(store.stats().dimension, 3U);
  EXPECT_EQ(store.stats().index_size, 0U);
  EXPECT_TRUE(store.search(std::vector<float>{1, 0, 0}, 10).empty());
  store.replace_index(make_index(2));
  EXPECT_EQ(store.stats().dimension, 2U);
  EXPECT_EQ(store.stats().index_size, 0U);
  EXPECT_TRUE(store.search(std::vector<float>{1, 0}, 10).empty());
}

TEST(VectorStoreTest, CountsFailedQueriesAndZeroKAsSuccessful) {
  fast_vector::VectorStore store(make_index());
  store.add(1, std::vector<float>{1.0F, 0.0F});

  EXPECT_THROW((void)store.search(std::vector<float>{1.0F}, 1), std::invalid_argument);
  EXPECT_THROW((void)store.search(std::vector<float>{0.0F, 0.0F}, 1), std::invalid_argument);
  EXPECT_TRUE(store.search(std::vector<float>{}, 0).empty());

  const fast_vector::StoreStats stats = store.stats();
  EXPECT_EQ(stats.successful_queries, 1U);
  EXPECT_EQ(stats.failed_queries, 2U);
}

TEST(VectorStoreTest, PrevalidatesBatchBeforeModification) {
  fast_vector::VectorStore store(make_index(), 2);
  const std::vector<fast_vector::VectorRecord> too_large{
      {1, {1.0F, 0.0F}}, {2, {0.0F, 1.0F}}, {3, {-1.0F, 0.0F}}};
  EXPECT_THROW(store.add_batch(too_large), std::length_error);

  const std::vector<fast_vector::VectorRecord> duplicate{{1, {1.0F, 0.0F}}, {1, {0.0F, 1.0F}}};
  EXPECT_THROW(store.add_batch(duplicate), std::invalid_argument);

  const std::vector<fast_vector::VectorRecord> bad_dimension{{1, {1.0F, 0.0F}}, {2, {1.0F}}};
  EXPECT_THROW(store.add_batch(bad_dimension), std::invalid_argument);

  const std::vector<fast_vector::VectorRecord> non_finite{
      {1, {1.0F, 0.0F}}, {2, {std::numeric_limits<float>::infinity(), 0.0F}}};
  EXPECT_THROW(store.add_batch(non_finite), std::invalid_argument);
  EXPECT_EQ(store.stats().index_size, 0U);
  EXPECT_EQ(store.stats().inserted_vectors, 0U);
}

TEST(VectorStoreTest, DocumentsPartialBatchWhenIndexRejectsExistingId) {
  fast_vector::VectorStore store(make_index());
  store.add(2, std::vector<float>{0.0F, 1.0F});
  const std::vector<fast_vector::VectorRecord> batch{
      {1, {1.0F, 0.0F}},
      {2, {-1.0F, 0.0F}},
      {3, {1.0F, 1.0F}},
  };

  EXPECT_THROW(store.add_batch(batch), std::invalid_argument);
  const fast_vector::StoreStats stats = store.stats();
  EXPECT_EQ(stats.index_size, 2U);
  EXPECT_EQ(stats.inserted_vectors, 2U);
  EXPECT_EQ(store.search(std::vector<float>{1.0F, 0.0F}, 10).size(), 2U);
}

TEST(VectorStoreTest, SupportsConcurrentReadersAndSerializedWriter) {
  fast_vector::VectorStore store(make_index());
  for (fast_vector::VectorId id = 1; id <= 100; ++id) {
    store.add(id, std::vector<float>{static_cast<float>(id), 1.0F});
  }

  constexpr std::size_t kReaderCount = 8;
  constexpr std::size_t kQueriesPerReader = 100;
  std::atomic<bool> start{false};
  std::atomic<std::size_t> unexpected_result_count{0};
  std::vector<std::thread> readers;
  readers.reserve(kReaderCount);
  for (std::size_t reader = 0; reader < kReaderCount; ++reader) {
    readers.emplace_back([&store, &start, &unexpected_result_count] {
      while (!start.load(std::memory_order_acquire)) {
        std::this_thread::yield();
      }
      for (std::size_t query = 0; query < kQueriesPerReader; ++query) {
        const auto results = store.search(std::vector<float>{1.0F, 0.5F}, 5);
        if (results.size() != 5U) {
          unexpected_result_count.fetch_add(1, std::memory_order_relaxed);
        }
      }
    });
  }
  std::thread writer([&store, &start] {
    while (!start.load(std::memory_order_acquire)) {
      std::this_thread::yield();
    }
    store.add(1'000, std::vector<float>{0.5F, 1.0F});
  });

  start.store(true, std::memory_order_release);
  for (std::thread& reader : readers) {
    reader.join();
  }
  writer.join();

  const fast_vector::StoreStats stats = store.stats();
  EXPECT_EQ(stats.index_size, 101U);
  EXPECT_EQ(stats.inserted_vectors, 101U);
  EXPECT_EQ(stats.successful_queries, kReaderCount * kQueriesPerReader);
  EXPECT_EQ(stats.failed_queries, 0U);
  EXPECT_EQ(unexpected_result_count.load(std::memory_order_relaxed), 0U);
}

}  // namespace
