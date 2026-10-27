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

class HnswIndexSerializer;

/** Strategy used to choose graph neighbors from a similarity-ranked candidate list. */
enum class HnswNeighborSelection {
  Simple,
  Heuristic,
};

/** Construction and default search parameters for an in-memory HNSW index. */
struct HnswConfig {
  std::size_t dimension;
  std::size_t max_connections = 16;
  std::size_t ef_construction = 200;
  std::size_t ef_search = 50;
  std::uint64_t random_seed = 42;
  DotProductKernel kernel = DotProductKernel::Scalar;
  HnswNeighborSelection neighbor_selection = HnswNeighborSelection::Simple;
};

/** Lightweight structural counters for validation and benchmark reporting. */
struct HnswStats {
  /** Physical nodes, including tombstones; live_count + deleted_count equals node_count. */
  std::size_t node_count;
  std::size_t max_level;
  std::size_t directed_edge_count;
  std::size_t maximum_node_degree;
  std::size_t live_count;
  std::size_t deleted_count;
};

/** Approximate cosine-similarity index using a hierarchical navigable small world graph. */
class HnswIndex final : public VectorIndex {
 public:
  explicit HnswIndex(HnswConfig config);

  /** Strong logical exception guarantee, including graph topology and RNG state. */
  void add(VectorId id, std::span<const float> vector) override;

  /** Return live IDs in ascending order. Requires no concurrent mutation. */
  [[nodiscard]] std::vector<VectorId> live_ids() const;

  /** Build a tombstone-free index in ID order with an explicit fresh RNG seed.
   * The source is unchanged on success or failure; normalized vector bits are preserved.
   */
  [[nodiscard]] HnswIndex rebuild(std::uint64_t random_seed) const;

  /** Mark an ID deleted without changing graph edges; absent/already deleted returns false.
   * Deleted IDs remain reserved. Requires exclusive external access, like insertion.
   */
  [[nodiscard]] bool erase(VectorId id);

  [[nodiscard]] std::vector<SearchResult> search(std::span<const float> query,
                                                 std::size_t k) const override;

  /** Search with a per-call candidate-list size. Values below k are raised to k. */
  [[nodiscard]] std::vector<SearchResult> search(std::span<const float> query, std::size_t k,
                                                 std::size_t ef_search) const;

  /** Number of live vectors, excluding tombstones. */
  [[nodiscard]] std::size_t size() const noexcept override;
  [[nodiscard]] std::size_t dimension() const noexcept override;
  [[nodiscard]] const HnswConfig& config() const noexcept;
  [[nodiscard]] HnswStats stats() const noexcept;

 private:
  friend class HnswIndexSerializer;
  class InsertionTransaction;
  using NodeIndex = std::uint32_t;
  using RawDotProduct = float (*)(const float*, const float*, std::size_t);

  struct Node {
    std::vector<std::vector<NodeIndex>> levels;
    bool deleted = false;
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
                                                    std::size_t ef, std::size_t level,
                                                    bool live_only = false) const;
  [[nodiscard]] std::vector<NodeIndex> select_neighbors(const std::vector<Candidate>& candidates,
                                                        std::size_t maximum_count) const;
  [[nodiscard]] std::size_t maximum_connections(std::size_t level) const noexcept;
  void insert_normalized(VectorId id, std::span<const float> normalized);
  void prune_neighbors(NodeIndex node, std::size_t level, InsertionTransaction& transaction);

  HnswConfig config_;
  RawDotProduct dot_product_;
  std::vector<float> vectors_;
  std::vector<VectorId> ids_;
  std::vector<Node> graph_;
  std::unordered_map<VectorId, NodeIndex> id_to_node_;
  std::mt19937_64 random_;
  NodeIndex entry_point_ = 0;
  std::size_t max_level_ = 0;
  std::size_t deleted_count_ = 0;
};

}  // namespace fast_vector
