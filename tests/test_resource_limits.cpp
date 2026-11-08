#include <gtest/gtest.h>

#include <chrono>
#include <filesystem>
#include <limits>

#include "fast_vector/flat_index_io.h"
#include "fast_vector/hnsw_index.h"
#include "fast_vector/vector_store.h"

namespace {
using namespace fast_vector;
std::unique_ptr<VectorIndex> index(VectorId id) {
  auto result = std::make_unique<FlatIndex>(2);
  result->add(id, std::vector<float>{1, 0});
  return result;
}
TEST(ResourceLimitsTest, LeasesSurviveOwnerAndRejectOverflow) {
  auto budget = std::make_shared<MemoryBudget>(10);
  auto lease = budget->reserve(10);
  EXPECT_EQ(budget->used(), 10);
  EXPECT_THROW((void)budget->reserve(1), ResourceExhausted);
  lease.reset();
  EXPECT_EQ(budget->used(), 0);
  EXPECT_THROW((void)resource_multiply(UINT64_MAX, 2), ResourceExhausted);
  EXPECT_THROW((void)resource_add(UINT64_MAX, 1), ResourceExhausted);
}
TEST(ResourceLimitsTest, OldIndexReferencesKeepSnapshotBudgetCharged) {
  auto budget = std::make_shared<MemoryBudget>(1040);
  VectorStore store(index(1), 10, true, {}, {}, budget);
  auto old = store.serving_snapshot()->index;
  EXPECT_EQ(budget->used(), 520);
  store.replace_index(index(2));
  EXPECT_EQ(budget->used(), 1040);
  EXPECT_THROW(store.replace_index(index(3)), ResourceExhausted);
  EXPECT_EQ(store.search(std::vector<float>{1, 0}, 1).at(0).id, 2);
  old.reset();
  EXPECT_EQ(budget->used(), 520);
  store.replace_index(index(3));
  EXPECT_EQ(budget->used(), 520);
}
TEST(ResourceLimitsTest, WritesBatchAndTopKFailBeforeMutation) {
  ResourceLimits limits;
  limits.maximum_vectors = 2;
  limits.maximum_top_k = 1;
  VectorStore store(index(1), 10, false, {}, limits);
  const std::vector<VectorRecord> batch{{2, {1, 0}}, {3, {0, 1}}};
  EXPECT_THROW(store.add_batch(batch), ResourceExhausted);
  EXPECT_EQ(store.stats().index_size, 1);
  store.add(2, std::vector<float>{0, 1});
  EXPECT_THROW(store.add(3, std::vector<float>{0, 1}), ResourceExhausted);
  EXPECT_THROW((void)store.search(std::vector<float>{1, 0}, 2), ResourceExhausted);
  EXPECT_TRUE(store.search({}, 0).empty());
  EXPECT_EQ(store.stats().index_size, 2);
}
TEST(ResourceLimitsTest, LoaderRejectsDeclaredShapeBeforeAllocatingRows) {
  const auto path =
      std::filesystem::temp_directory_path() /
      ("fast-vector-budget-flat-" +
       std::to_string(std::chrono::steady_clock::now().time_since_epoch().count()) + ".fv");
  FlatIndex flat(2);
  flat.add(1, std::vector<float>{1, 0});
  save_flat_index(flat, path);
  ResourceLimits limits;
  limits.maximum_vectors = 0;
  EXPECT_THROW((void)load_flat_index(path, DotProductKernel::Scalar, limits), ResourceExhausted);
  EXPECT_THROW((void)snapshot_loading_units(path, limits), ResourceExhausted);
  std::filesystem::remove(path);
}
TEST(ResourceLimitsTest, PhysicalHnswNodesRemainChargedAfterDeletion) {
  auto graph = std::make_unique<HnswIndex>(
      HnswConfig{.dimension = 2, .max_connections = 2, .ef_construction = 4});
  graph->add(1, std::vector<float>{1, 0});
  EXPECT_TRUE(graph->erase(1));
  ResourceLimits limits;
  limits.maximum_vectors = 1;
  VectorStore store(std::move(graph), 10, false, {}, limits);
  EXPECT_THROW(store.add(2, std::vector<float>{1, 0}), ResourceExhausted);
  EXPECT_EQ(store.stats().index_size, 0);
}
}  // namespace
