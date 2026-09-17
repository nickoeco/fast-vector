#include <gtest/gtest.h>

#include <limits>
#include <stdexcept>
#include <vector>

#include "fast_vector/hnsw_index.h"

namespace {

fast_vector::HnswConfig test_config(const std::size_t dimension = 2) {
  return fast_vector::HnswConfig{
      .dimension = dimension,
      .max_connections = 4,
      .ef_construction = 16,
      .ef_search = 16,
      .random_seed = 42,
  };
}

TEST(HnswIndexTest, RejectsInvalidConfiguration) {
  auto config = test_config();
  config.dimension = 0;
  EXPECT_THROW((void)fast_vector::HnswIndex{config}, std::invalid_argument);

  config = test_config();
  config.max_connections = 1;
  EXPECT_THROW((void)fast_vector::HnswIndex{config}, std::invalid_argument);

  config = test_config();
  config.ef_construction = 3;
  EXPECT_THROW((void)fast_vector::HnswIndex{config}, std::invalid_argument);

  config = test_config();
  config.ef_search = 0;
  EXPECT_THROW((void)fast_vector::HnswIndex{config}, std::invalid_argument);
}

TEST(HnswIndexTest, AddsAndFindsOneVector) {
  fast_vector::HnswIndex index(test_config());
  index.add(42, std::vector<float>{3.0F, 4.0F});

  const auto results = index.search(std::vector<float>{3.0F, 4.0F}, 1);
  ASSERT_EQ(results.size(), 1U);
  EXPECT_EQ(results[0].id, 42U);
  EXPECT_NEAR(results[0].score, 1.0F, 1.0e-5F);
  EXPECT_EQ(index.size(), 1U);
  EXPECT_EQ(index.dimension(), 2U);
  EXPECT_EQ(index.stats().node_count, 1U);
}

TEST(HnswIndexTest, ReturnsOrderedNearestNeighbors) {
  fast_vector::HnswIndex index(test_config());
  index.add(30, std::vector<float>{0.0F, 1.0F});
  index.add(20, std::vector<float>{1.0F, -1.0F});
  index.add(10, std::vector<float>{1.0F, 1.0F});
  index.add(40, std::vector<float>{-1.0F, 0.0F});

  const auto results = index.search(std::vector<float>{1.0F, 0.0F}, 3);
  ASSERT_EQ(results.size(), 3U);
  EXPECT_EQ(results[0].id, 10U);
  EXPECT_EQ(results[1].id, 20U);
  EXPECT_EQ(results[2].id, 30U);
  EXPECT_GE(results[0].score, results[1].score);
  EXPECT_GE(results[1].score, results[2].score);
}

TEST(HnswIndexTest, RejectsInvalidInputWithoutChangingIndex) {
  fast_vector::HnswIndex index(test_config());
  index.add(1, std::vector<float>{1.0F, 0.0F});

  EXPECT_THROW(index.add(1, std::vector<float>{0.0F, 1.0F}), std::invalid_argument);
  EXPECT_THROW(index.add(2, std::vector<float>{1.0F}), std::invalid_argument);
  EXPECT_THROW(index.add(2, std::vector<float>{0.0F, 0.0F}), std::invalid_argument);
  EXPECT_THROW(index.add(2, std::vector<float>{std::numeric_limits<float>::infinity(), 0.0F}),
               std::invalid_argument);
  EXPECT_EQ(index.size(), 1U);
}

TEST(HnswIndexTest, HandlesSearchBoundaries) {
  fast_vector::HnswIndex empty(test_config());
  EXPECT_TRUE(empty.search(std::vector<float>{1.0F, 0.0F}, 5).empty());
  EXPECT_TRUE(empty.search(std::vector<float>{}, 0).empty());
  EXPECT_THROW(empty.search(std::vector<float>{1.0F}, 1), std::invalid_argument);
  EXPECT_THROW(empty.search(std::vector<float>{0.0F, 0.0F}, 1), std::invalid_argument);
  EXPECT_THROW(empty.search(std::vector<float>{1.0F, 0.0F}, 1, 0), std::invalid_argument);

  fast_vector::HnswIndex index(test_config());
  index.add(1, std::vector<float>{1.0F, 0.0F});
  index.add(2, std::vector<float>{0.0F, 1.0F});
  EXPECT_EQ(index.search(std::vector<float>{1.0F, 0.0F}, 10, 1).size(), 2U);
}

TEST(HnswIndexTest, KeepsDegreeBoundAndBuildsMultipleLevels) {
  fast_vector::HnswIndex index(test_config(3));
  for (fast_vector::VectorId id = 1; id <= 200; ++id) {
    index.add(id, std::vector<float>{static_cast<float>(id), static_cast<float>(id % 17 + 1),
                                     static_cast<float>(id % 11 + 2)});
  }

  const fast_vector::HnswStats stats = index.stats();
  EXPECT_EQ(stats.node_count, 200U);
  EXPECT_GT(stats.max_level, 0U);
  EXPECT_GT(stats.directed_edge_count, 0U);
  EXPECT_LE(stats.maximum_node_degree, index.config().max_connections);
}

TEST(HnswIndexTest, FixedSeedProducesDeterministicResults) {
  fast_vector::HnswIndex first(test_config(3));
  fast_vector::HnswIndex second(test_config(3));
  for (fast_vector::VectorId id = 1; id <= 100; ++id) {
    const std::vector<float> vector{static_cast<float>(id % 13 + 1), static_cast<float>(id % 7 + 2),
                                    static_cast<float>(id % 5 + 3)};
    first.add(id, vector);
    second.add(id, vector);
  }

  EXPECT_EQ(first.stats().max_level, second.stats().max_level);
  EXPECT_EQ(first.stats().directed_edge_count, second.stats().directed_edge_count);
  EXPECT_EQ(first.search(std::vector<float>{2.0F, 3.0F, 4.0F}, 10),
            second.search(std::vector<float>{2.0F, 3.0F, 4.0F}, 10));
}

TEST(HnswIndexTest, RepeatedSearchDoesNotChangeGraph) {
  fast_vector::HnswIndex index(test_config());
  index.add(7, std::vector<float>{1.0F, 0.0F});
  index.add(8, std::vector<float>{0.0F, 1.0F});
  const fast_vector::HnswStats before = index.stats();
  const auto first = index.search(std::vector<float>{0.5F, 0.5F}, 2);
  const auto second = index.search(std::vector<float>{0.5F, 0.5F}, 2);
  const fast_vector::HnswStats after = index.stats();

  EXPECT_EQ(first, second);
  EXPECT_EQ(before.node_count, after.node_count);
  EXPECT_EQ(before.max_level, after.max_level);
  EXPECT_EQ(before.directed_edge_count, after.directed_edge_count);
}

}  // namespace
