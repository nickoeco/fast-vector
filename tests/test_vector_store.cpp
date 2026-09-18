#include <gtest/gtest.h>

#include <atomic>
#include <limits>
#include <memory>
#include <stdexcept>
#include <thread>
#include <vector>

#include "fast_vector/flat_index.h"
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
  EXPECT_EQ(store.maximum_batch_size(), 1'000U);
}

TEST(VectorStoreTest, CountsFailedQueriesAndZeroKAsSuccessful) {
  fast_vector::VectorStore store(make_index());
  store.add(1, std::vector<float>{1.0F, 0.0F});

  EXPECT_THROW(store.search(std::vector<float>{1.0F}, 1), std::invalid_argument);
  EXPECT_THROW(store.search(std::vector<float>{0.0F, 0.0F}, 1), std::invalid_argument);
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
