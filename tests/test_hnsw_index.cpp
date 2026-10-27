#include <gtest/gtest.h>

#include <limits>
#include <stdexcept>
#include <unordered_set>
#include <vector>

#include "fast_vector/flat_index.h"
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

  config = test_config();
  config.neighbor_selection = static_cast<fast_vector::HnswNeighborSelection>(99);
  EXPECT_THROW((void)fast_vector::HnswIndex{config}, std::invalid_argument);

  config = test_config();
  config.max_connections = std::numeric_limits<std::uint32_t>::max() / 2U + 1U;
  config.ef_construction = config.max_connections;
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
  EXPECT_THROW((void)empty.search(std::vector<float>{1.0F}, 1), std::invalid_argument);
  EXPECT_THROW((void)empty.search(std::vector<float>{0.0F, 0.0F}, 1), std::invalid_argument);
  EXPECT_THROW((void)empty.search(std::vector<float>{1.0F, 0.0F}, 1, 0), std::invalid_argument);

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
  EXPECT_LE(stats.maximum_node_degree, index.config().max_connections * 2);
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

TEST(HnswIndexTest, MarksDeletionWithoutChangingGraphAndReservesId) {
  fast_vector::HnswIndex index(test_config());
  const auto maximum_id = std::numeric_limits<fast_vector::VectorId>::max();
  EXPECT_FALSE(index.erase(0));
  index.add(0, std::vector<float>{1, 0});
  index.add(maximum_id, std::vector<float>{0, 1});
  index.add(7, std::vector<float>{-1, 0});
  const auto before = index.stats();
  ASSERT_TRUE(index.erase(0));
  EXPECT_FALSE(index.erase(0));
  EXPECT_FALSE(index.erase(123));
  EXPECT_EQ(index.size(), 2U);
  const auto after = index.stats();
  EXPECT_EQ(after.node_count, before.node_count);
  EXPECT_EQ(after.live_count, 2U);
  EXPECT_EQ(after.deleted_count, 1U);
  EXPECT_EQ(after.directed_edge_count, before.directed_edge_count);
  EXPECT_EQ(after.max_level, before.max_level);
  EXPECT_EQ(after.maximum_node_degree, before.maximum_node_degree);
  EXPECT_EQ(index.search(std::vector<float>{1, 0}, 10),
            (std::vector<fast_vector::SearchResult>{{maximum_id, 0}, {7, -1}}));
  EXPECT_THROW(index.add(0, std::vector<float>{1, 1}), std::invalid_argument);
  ASSERT_TRUE(index.erase(maximum_id));
  EXPECT_FALSE(index.erase(maximum_id));
  EXPECT_EQ(index.search(std::vector<float>{1, 0}, 1),
            (std::vector<fast_vector::SearchResult>{{7, -1}}));
}

TEST(HnswIndexTest, DeletionKeepsScoreAndIdOrdering) {
  fast_vector::HnswIndex index(test_config());
  for (const auto id : {30U, 20U, 10U}) {
    index.add(id, std::vector<float>{1, 0});
  }
  index.add(40, std::vector<float>{-1, 0});
  ASSERT_TRUE(index.erase(30));
  EXPECT_EQ(index.search(std::vector<float>{1, 0}, 10, 1),
            (std::vector<fast_vector::SearchResult>{{10, 1}, {20, 1}, {40, -1}}));
  EXPECT_TRUE(index.search(std::vector<float>{}, 0).empty());
}

TEST(HnswIndexTest, AllDeletedIndexAcceptsNewIdsAtNewPhysicalPositions) {
  fast_vector::HnswIndex index(test_config());
  index.add(10, std::vector<float>{1, 0});
  index.add(20, std::vector<float>{0, 1});
  ASSERT_TRUE(index.erase(10));
  ASSERT_TRUE(index.erase(20));
  EXPECT_EQ(index.size(), 0U);
  EXPECT_EQ(index.stats().node_count, 2U);
  EXPECT_TRUE(index.search(std::vector<float>{1, 0}, 10).empty());
  EXPECT_THROW((void)index.search(std::vector<float>{0, 0}, 1), std::invalid_argument);
  EXPECT_THROW((void)index.search(std::vector<float>{1}, 1), std::invalid_argument);
  EXPECT_THROW((void)index.search(std::vector<float>{1, 0}, 1, 0), std::invalid_argument);
  index.add(30, std::vector<float>{1, 0});
  EXPECT_EQ(index.stats().node_count, 3U);
  EXPECT_EQ(index.stats().deleted_count, 2U);
  EXPECT_EQ(index.size(), 1U);
  EXPECT_EQ(index.search(std::vector<float>{1, 0}, 10),
            (std::vector<fast_vector::SearchResult>{{30, 1}}));
  index.add(40, std::vector<float>{0, 1});
  ASSERT_TRUE(index.erase(30));
  EXPECT_EQ(index.search(std::vector<float>{0, 1}, 10),
            (std::vector<fast_vector::SearchResult>{{40, 1}}));
}

TEST(HnswIndexTest, TombstoneSearchMatchesLiveFlatGroundTruthOnConnectedGraph) {
  for (const auto selection : {fast_vector::HnswNeighborSelection::Simple,
                               fast_vector::HnswNeighborSelection::Heuristic}) {
    auto config = test_config(5);
    config.max_connections = 32;
    config.ef_construction = 64;
    config.neighbor_selection = selection;
    fast_vector::HnswIndex index(config);
    fast_vector::FlatIndex truth(config.dimension);
    // All 48 nodes fit in each base-layer adjacency list, giving a connected test graph.
    for (fast_vector::VectorId id = 0; id < 48; ++id) {
      std::vector<float> vector(config.dimension);
      for (std::size_t d = 0; d < vector.size(); ++d) {
        vector[d] = static_cast<float>((id * (d + 3)) % 31 + 1);
      }
      index.add(id, vector);
      truth.add(id, vector);
    }
    for (fast_vector::VectorId id = 0; id < 48; ++id) {
      if (id % 4 != 0) {
        ASSERT_TRUE(index.erase(id));
        ASSERT_TRUE(truth.erase(id));
      }
    }
    for (std::size_t q = 0; q < 15; ++q) {
      std::vector<float> query(config.dimension);
      for (std::size_t d = 0; d < query.size(); ++d) {
        query[d] = static_cast<float>(((q + 2) * (d + 5)) % 29 + 1);
      }
      for (const auto k : {1U, 10U, 100U}) {
        EXPECT_EQ(index.search(query, k, 1), truth.search(query, k));
      }
    }
    EXPECT_EQ(index.stats().live_count, 12U);
    EXPECT_EQ(index.stats().deleted_count, 36U);
  }
}

TEST(HnswIndexTest, RebuildKeepsLiveIdsAndDropsReservationsOfDeletedIds) {
  for (const auto selection : {fast_vector::HnswNeighborSelection::Simple,
                               fast_vector::HnswNeighborSelection::Heuristic}) {
    auto config = test_config();
    config.neighbor_selection = selection;
    fast_vector::HnswIndex source(config);
    source.add(30, std::vector<float>{1, 0});
    source.add(10, std::vector<float>{0, 1});
    source.add(20, std::vector<float>{-1, 0});
    ASSERT_TRUE(source.erase(10));
    auto rebuilt = source.rebuild(123);
    EXPECT_EQ(rebuilt.live_ids(), (std::vector<fast_vector::VectorId>{20, 30}));
    EXPECT_EQ(rebuilt.stats().node_count, 2U);
    EXPECT_EQ(rebuilt.stats().deleted_count, 0U);
    EXPECT_EQ(rebuilt.config().random_seed, 123U);
    EXPECT_EQ(rebuilt.config().neighbor_selection, selection);
    EXPECT_EQ(rebuilt.config().kernel, config.kernel);
    EXPECT_EQ(rebuilt.search(std::vector<float>{1, 0}, 10),
              (std::vector<fast_vector::SearchResult>{{30, 1}, {20, -1}}));
    EXPECT_EQ(source.stats().node_count, 3U);
    EXPECT_EQ(source.stats().deleted_count, 1U);
    EXPECT_THROW(source.add(10, std::vector<float>{1, 1}), std::invalid_argument);
    rebuilt.add(10, std::vector<float>{1, 1});
    EXPECT_EQ(rebuilt.size(), 3U);
  }
}

TEST(HnswIndexTest, EmptyAndAllDeletedRebuildsStartEmptyAndAcceptNewVectors) {
  fast_vector::HnswIndex source(test_config());
  EXPECT_EQ(source.rebuild(42).stats().node_count, 0U);
  source.add(7, std::vector<float>{1, 0});
  ASSERT_TRUE(source.erase(7));
  auto rebuilt = source.rebuild(42);
  EXPECT_EQ(rebuilt.size(), 0U);
  EXPECT_EQ(rebuilt.stats().node_count, 0U);
  EXPECT_EQ(rebuilt.stats().deleted_count, 0U);
  rebuilt.add(7, std::vector<float>{0, 1});
  EXPECT_EQ(rebuilt.search(std::vector<float>{0, 1}, 1),
            (std::vector<fast_vector::SearchResult>{{7, 1}}));
}

TEST(HnswIndexTest, HeuristicSelectionBuildsBoundedDeterministicGraph) {
  auto config = test_config(4);
  config.neighbor_selection = fast_vector::HnswNeighborSelection::Heuristic;
  fast_vector::HnswIndex first(config);
  fast_vector::HnswIndex second(config);
  for (fast_vector::VectorId id = 1; id <= 250; ++id) {
    const std::vector<float> vector{
        static_cast<float>(id % 23 + 1), static_cast<float>(id % 17 + 2),
        static_cast<float>(id % 11 + 3), static_cast<float>(id % 7 + 4)};
    first.add(id, vector);
    second.add(id, vector);
  }

  EXPECT_EQ(first.config().neighbor_selection, fast_vector::HnswNeighborSelection::Heuristic);
  EXPECT_LE(first.stats().maximum_node_degree, config.max_connections * 2);
  EXPECT_EQ(first.stats().directed_edge_count, second.stats().directed_edge_count);
  EXPECT_EQ(first.search(std::vector<float>{2.0F, 3.0F, 4.0F, 5.0F}, 10),
            second.search(std::vector<float>{2.0F, 3.0F, 4.0F, 5.0F}, 10));
}

TEST(HnswIndexTest, LargerEfSearchDoesNotReduceOverlapOnFixedDataset) {
  auto config = test_config(8);
  config.max_connections = 8;
  config.ef_construction = 64;
  config.ef_search = 10;
  config.neighbor_selection = fast_vector::HnswNeighborSelection::Heuristic;
  fast_vector::HnswIndex approximate(config);
  fast_vector::FlatIndex exact(config.dimension);
  for (fast_vector::VectorId id = 1; id <= 300; ++id) {
    std::vector<float> vector(config.dimension);
    for (std::size_t component = 0; component < vector.size(); ++component) {
      vector[component] = static_cast<float>((id * (component + 3)) % 101 + 1);
    }
    approximate.add(id, vector);
    exact.add(id, vector);
  }

  std::size_t low_overlap = 0;
  std::size_t high_overlap = 0;
  for (std::size_t query_index = 0; query_index < 20; ++query_index) {
    std::vector<float> query(config.dimension);
    for (std::size_t component = 0; component < query.size(); ++component) {
      query[component] = static_cast<float>(((query_index + 5) * (component + 7)) % 97 + 1);
    }
    const auto truth = exact.search(query, 10);
    const auto low = approximate.search(query, 10, 10);
    const auto high = approximate.search(query, 10, 80);
    std::unordered_set<fast_vector::VectorId> truth_ids;
    for (const auto& result : truth) {
      truth_ids.insert(result.id);
    }
    for (const auto& result : low) {
      low_overlap += truth_ids.contains(result.id) ? 1U : 0U;
    }
    for (const auto& result : high) {
      high_overlap += truth_ids.contains(result.id) ? 1U : 0U;
    }
  }

  EXPECT_GE(high_overlap, low_overlap);
}

}  // namespace
