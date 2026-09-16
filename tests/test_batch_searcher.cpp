#include <future>
#include <stdexcept>
#include <vector>

#include <gtest/gtest.h>

#include "fast_vector/batch_searcher.h"
#include "fast_vector/flat_index.h"

namespace {

fast_vector::FlatIndex make_batch_index() {
    fast_vector::FlatIndex index(2);
    index.add(10, std::vector<float>{1.0F, 0.0F});
    index.add(20, std::vector<float>{0.0F, 1.0F});
    index.add(30, std::vector<float>{-1.0F, 0.0F});
    return index;
}

TEST(BatchSearcherTest, ParallelAndSerialResultsMatchSingleQueries) {
    const auto index = make_batch_index();
    const std::vector<float> queries{1.0F, 0.0F, 0.0F, 1.0F, -1.0F, 0.0F};
    fast_vector::BatchSearcher serial(index, 1);
    fast_vector::BatchSearcher parallel(index, 3);

    const auto serial_results = serial.search(queries, 3, 2);
    const auto parallel_results = parallel.search(queries, 3, 2);

    ASSERT_EQ(serial_results.size(), 3U);
    EXPECT_EQ(serial_results, parallel_results);
    for (std::size_t i = 0; i < 3; ++i) {
        const std::span<const float> query(queries.data() + i * 2, 2);
        EXPECT_EQ(parallel_results[i], index.search(query, 2));
    }
    EXPECT_EQ(parallel.thread_count(), 3U);
}

TEST(BatchSearcherTest, HandlesEmptyBatchAndZeroK) {
    const auto index = make_batch_index();
    fast_vector::BatchSearcher searcher(index, 2);
    EXPECT_TRUE(searcher.search({}, 0, 10).empty());

    const auto results = searcher.search(std::vector<float>{1.0F, 0.0F}, 1, 0);
    ASSERT_EQ(results.size(), 1U);
    EXPECT_TRUE(results[0].empty());
}

TEST(BatchSearcherTest, RejectsInvalidThreadCountAndShape) {
    const auto index = make_batch_index();
    EXPECT_THROW(fast_vector::BatchSearcher(index, 0), std::invalid_argument);

    fast_vector::BatchSearcher searcher(index, 2);
    EXPECT_THROW(searcher.search(std::vector<float>{1.0F, 0.0F, 1.0F}, 2, 1),
                 std::invalid_argument);
}

TEST(BatchSearcherTest, PropagatesTaskErrorsAndRemainsUsable) {
    const auto index = make_batch_index();
    fast_vector::BatchSearcher searcher(index, 2);
    const std::vector<float> invalid_queries{1.0F, 0.0F, 0.0F, 0.0F};
    EXPECT_THROW(searcher.search(invalid_queries, 2, 1), std::invalid_argument);

    const auto recovered = searcher.search(std::vector<float>{0.0F, 1.0F}, 1, 1);
    ASSERT_EQ(recovered.size(), 1U);
    ASSERT_EQ(recovered[0].size(), 1U);
    EXPECT_EQ(recovered[0][0].id, 20U);
}

TEST(BatchSearcherTest, SupportsConcurrentBatchCalls) {
    const auto index = make_batch_index();
    fast_vector::BatchSearcher searcher(index, 3);
    const std::vector<float> queries{1.0F, 0.0F, 0.0F, 1.0F, -1.0F, 0.0F};

    auto first = std::async(std::launch::async, [&] { return searcher.search(queries, 3, 2); });
    auto second =
        std::async(std::launch::async, [&] { return searcher.search(queries, 3, 2); });

    EXPECT_EQ(first.get(), second.get());
}

}  // namespace
