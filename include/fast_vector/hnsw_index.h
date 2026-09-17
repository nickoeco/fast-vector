#pragma once

#include <cstddef>
#include <cstdint>
#include <random>
#include <span>
#include <unordered_map>
#include <vector>

#include "fast_vector/distance.h"
#include "fast_vector/vector_index.h"

namespace fast_vector {

/** Construction and default search parameters for an in-memory HNSW index. */
struct HnswConfig {
  std::size_t dimension;
  std::size_t max_connections = 16;
  std::size_t ef_construction = 200;
  std::size_t ef_search = 50;
  std::uint64_t random_seed = 42;
  DotProductKernel kernel = DotProductKernel::Scalar;
};

/** Lightweight structural counters for validation and benchmark reporting. */
struct HnswStats {
  std::size_t node_count;
  std::size_t max_level;
  std::size_t directed_edge_count;
  std::size_t maximum_node_degree;
};

/** Approximate cosine-similarity index using a hierarchical navigable small world graph. */
class HnswIndex final : public VectorIndex {
 public:
  explicit HnswIndex(HnswConfig config);

  void add(VectorId id, std::span<const float> vector) override;

  [[nodiscard]] std::vector<SearchResult> search(std::span<const float> query,
                                                 std::size_t k) const override;

  /** Search with a per-call candidate-list size. Values below k are raised to k. */
  [[nodiscard]] std::vector<SearchResult> search(std::span<const float> query, std::size_t k,
                                                 std::size_t ef_search) const;

  [[nodiscard]] std::size_t size() const noexcept override;
  [[nodiscard]] std::size_t dimension() const noexcept override;
  [[nodiscard]] const HnswConfig& config() const noexcept;
  [[nodiscard]] HnswStats stats() const noexcept;

 private:
  using NodeIndex = std::uint32_t;
  using RawDotProduct = float (*)(const float*, const float*, std::size_t);

  struct Node {
    std::vector<std::vector<NodeIndex>> levels;
  };

  struct Candidate {
    NodeIndex node;
    float score;
  };

  [[nodiscard]] std::size_t random_level();
  [[nodiscard]] bool is_better(const Candidate& lhs, const Candidate& rhs) const;
  [[nodiscard]] float similarity(std::span<const float> query, NodeIndex node) const;
  [[nodiscard]] NodeIndex greedy_search_layer(std::span<const float> query, NodeIndex entry,
                                              std::size_t level) const;
  [[nodiscard]] std::vector<Candidate> search_layer(std::span<const float> query, NodeIndex entry,
                                                    std::size_t ef, std::size_t level) const;
  [[nodiscard]] std::vector<NodeIndex> select_neighbors(
      const std::vector<Candidate>& candidates) const;
  void prune_neighbors(NodeIndex node, std::size_t level);

  HnswConfig config_;
  RawDotProduct dot_product_;
  std::vector<float> vectors_;
  std::vector<VectorId> ids_;
  std::vector<Node> graph_;
  std::unordered_map<VectorId, NodeIndex> id_to_node_;
  std::mt19937_64 random_;
  NodeIndex entry_point_ = 0;
  std::size_t max_level_ = 0;
};

}  // namespace fast_vector
