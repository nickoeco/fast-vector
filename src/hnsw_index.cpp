#include "fast_vector/hnsw_index.h"

#include <algorithm>
#include <cmath>
#include <limits>
#include <queue>
#include <stdexcept>
#include <utility>

#include "distance_kernels.h"

namespace fast_vector {
namespace {

constexpr std::size_t kMaximumGeneratedLevel = 63;

HnswConfig validate_config(HnswConfig config) {
  if (config.dimension == 0) {
    throw std::invalid_argument("index dimension must be greater than zero");
  }
  if (config.max_connections < 2) {
    throw std::invalid_argument("HNSW max_connections must be at least two");
  }
  if (config.max_connections > std::numeric_limits<std::size_t>::max() / 2) {
    throw std::invalid_argument("HNSW max_connections is too large");
  }
  if (config.ef_construction < config.max_connections) {
    throw std::invalid_argument("HNSW ef_construction must be at least max_connections");
  }
  if (config.ef_search == 0) {
    throw std::invalid_argument("HNSW ef_search must be greater than zero");
  }
  return config;
}

}  // namespace

HnswIndex::HnswIndex(HnswConfig config)
    : config_(validate_config(config)),
      dot_product_(detail::resolve_dot_product_kernel(config_.kernel)),
      random_(config_.random_seed) {}

void HnswIndex::add(const VectorId id, const std::span<const float> vector) {
  if (vector.size() != config_.dimension) {
    throw std::invalid_argument("vector dimension does not match index dimension");
  }
  if (id_to_node_.contains(id)) {
    throw std::invalid_argument("vector ID already exists");
  }
  if (size() >= static_cast<std::size_t>(std::numeric_limits<NodeIndex>::max())) {
    throw std::length_error("HNSW node index capacity exceeded");
  }

  std::vector<float> normalized = normalize_l2(vector);
  const std::size_t level = random_level();
  const NodeIndex new_node = static_cast<NodeIndex>(size());

  if (graph_.empty()) {
    vectors_.insert(vectors_.end(), normalized.begin(), normalized.end());
    ids_.push_back(id);
    graph_.push_back(Node{std::vector<std::vector<NodeIndex>>(level + 1)});
    id_to_node_.emplace(id, new_node);
    entry_point_ = new_node;
    max_level_ = level;
    return;
  }

  NodeIndex entry = entry_point_;
  for (std::size_t current_level = max_level_; current_level > level; --current_level) {
    entry = greedy_search_layer(normalized, entry, current_level);
  }

  std::vector<std::vector<NodeIndex>> connection_plan(level + 1);
  const std::size_t first_connected_level = std::min(level, max_level_);
  for (std::size_t current_level = first_connected_level + 1; current_level-- > 0;) {
    const std::vector<Candidate> candidates =
        search_layer(normalized, entry, config_.ef_construction, current_level);
    connection_plan[current_level] =
        select_neighbors(candidates, maximum_connections(current_level));
    if (!candidates.empty()) {
      entry = candidates.front().node;
    }
  }

  vectors_.insert(vectors_.end(), normalized.begin(), normalized.end());
  ids_.push_back(id);
  graph_.push_back(Node{std::move(connection_plan)});
  id_to_node_.emplace(id, new_node);

  for (std::size_t current_level = 0; current_level <= first_connected_level; ++current_level) {
    const std::vector<NodeIndex> neighbors = graph_[new_node].levels[current_level];
    for (const NodeIndex neighbor : neighbors) {
      graph_[neighbor].levels[current_level].push_back(new_node);
      prune_neighbors(neighbor, current_level);
    }
  }

  if (level > max_level_) {
    entry_point_ = new_node;
    max_level_ = level;
  }
}

std::vector<SearchResult> HnswIndex::search(const std::span<const float> query,
                                            const std::size_t k) const {
  return search(query, k, config_.ef_search);
}

std::vector<SearchResult> HnswIndex::search(const std::span<const float> query, const std::size_t k,
                                            const std::size_t ef_search) const {
  if (k == 0) {
    return {};
  }
  if (query.size() != config_.dimension) {
    throw std::invalid_argument("query dimension does not match index dimension");
  }
  if (ef_search == 0) {
    throw std::invalid_argument("HNSW ef_search must be greater than zero");
  }

  const std::vector<float> normalized = normalize_l2(query);
  if (graph_.empty()) {
    return {};
  }

  NodeIndex entry = entry_point_;
  for (std::size_t level = max_level_; level > 0; --level) {
    entry = greedy_search_layer(normalized, entry, level);
  }

  const std::size_t result_count = std::min(k, size());
  const std::size_t effective_ef = std::max(result_count, ef_search);
  std::vector<Candidate> candidates = search_layer(normalized, entry, effective_ef, 0);
  if (candidates.size() > result_count) {
    candidates.resize(result_count);
  }

  std::vector<SearchResult> results;
  results.reserve(candidates.size());
  for (const Candidate& candidate : candidates) {
    results.push_back(SearchResult{ids_[candidate.node], candidate.score});
  }
  return results;
}

std::size_t HnswIndex::size() const noexcept { return ids_.size(); }

std::size_t HnswIndex::dimension() const noexcept { return config_.dimension; }

const HnswConfig& HnswIndex::config() const noexcept { return config_; }

HnswStats HnswIndex::stats() const noexcept {
  HnswStats result{size(), graph_.empty() ? 0 : max_level_, 0, 0};
  for (const Node& node : graph_) {
    std::size_t degree = 0;
    for (const std::vector<NodeIndex>& neighbors : node.levels) {
      result.directed_edge_count += neighbors.size();
      degree = std::max(degree, neighbors.size());
    }
    result.maximum_node_degree = std::max(result.maximum_node_degree, degree);
  }
  return result;
}

std::size_t HnswIndex::random_level() {
  // Mapping generator bits directly avoids implementation-defined distribution behavior.
  constexpr double kDenominator = 9007199254740993.0;  // 2^53 + 1
  const std::uint64_t mantissa = (random_() >> 11U) + 1U;
  const double uniform = static_cast<double>(mantissa) / kDenominator;
  const double multiplier = 1.0 / std::log(static_cast<double>(config_.max_connections));
  const auto level = static_cast<std::size_t>(-std::log(uniform) * multiplier);
  return std::min(level, kMaximumGeneratedLevel);
}

bool HnswIndex::is_better(const Candidate& lhs, const Candidate& rhs) const {
  if (lhs.score != rhs.score) {
    return lhs.score > rhs.score;
  }
  return ids_[lhs.node] < ids_[rhs.node];
}

float HnswIndex::similarity(const std::span<const float> query, const NodeIndex node) const {
  const float* stored = vectors_.data() + static_cast<std::size_t>(node) * config_.dimension;
  return dot_product_(query.data(), stored, config_.dimension);
}

HnswIndex::NodeIndex HnswIndex::greedy_search_layer(const std::span<const float> query,
                                                    NodeIndex entry,
                                                    const std::size_t level) const {
  Candidate best{entry, similarity(query, entry)};
  bool improved = true;
  while (improved) {
    improved = false;
    for (const NodeIndex neighbor : graph_[best.node].levels[level]) {
      const Candidate candidate{neighbor, similarity(query, neighbor)};
      if (is_better(candidate, best)) {
        best = candidate;
        improved = true;
      }
    }
  }
  return best.node;
}

std::vector<HnswIndex::Candidate> HnswIndex::search_layer(const std::span<const float> query,
                                                          const NodeIndex entry,
                                                          const std::size_t ef,
                                                          const std::size_t level) const {
  const auto better = [this](const Candidate& lhs, const Candidate& rhs) {
    return is_better(lhs, rhs);
  };
  const auto worse = [this](const Candidate& lhs, const Candidate& rhs) {
    return is_better(rhs, lhs);
  };
  std::priority_queue<Candidate, std::vector<Candidate>, decltype(worse)> pending(worse);
  std::priority_queue<Candidate, std::vector<Candidate>, decltype(better)> best(better);
  std::vector<bool> visited(size(), false);

  const Candidate initial{entry, similarity(query, entry)};
  pending.push(initial);
  best.push(initial);
  visited[entry] = true;

  while (!pending.empty()) {
    const Candidate current = pending.top();
    if (best.size() >= ef && is_better(best.top(), current)) {
      break;
    }
    pending.pop();

    for (const NodeIndex neighbor : graph_[current.node].levels[level]) {
      if (visited[neighbor]) {
        continue;
      }
      visited[neighbor] = true;
      const Candidate candidate{neighbor, similarity(query, neighbor)};
      if (best.size() < ef || is_better(candidate, best.top())) {
        pending.push(candidate);
        best.push(candidate);
        if (best.size() > ef) {
          best.pop();
        }
      }
    }
  }

  std::vector<Candidate> result;
  result.reserve(best.size());
  while (!best.empty()) {
    result.push_back(best.top());
    best.pop();
  }
  std::sort(result.begin(), result.end(), better);
  return result;
}

std::vector<HnswIndex::NodeIndex> HnswIndex::select_neighbors(
    const std::vector<Candidate>& candidates, const std::size_t maximum_count) const {
  const std::size_t count = std::min(maximum_count, candidates.size());
  std::vector<NodeIndex> selected;
  selected.reserve(count);

  if (config_.neighbor_selection == HnswNeighborSelection::Simple) {
    for (std::size_t i = 0; i < count; ++i) {
      selected.push_back(candidates[i].node);
    }
    return selected;
  }

  std::vector<NodeIndex> pruned;
  pruned.reserve(candidates.size());
  for (const Candidate& candidate : candidates) {
    bool diverse = true;
    const std::span<const float> candidate_vector(
        vectors_.data() + static_cast<std::size_t>(candidate.node) * config_.dimension,
        config_.dimension);
    for (const NodeIndex selected_node : selected) {
      // For cosine distance, d(c, r) > d(c, q) becomes sim(c, r) < sim(c, q).
      if (similarity(candidate_vector, selected_node) >= candidate.score) {
        diverse = false;
        break;
      }
    }
    if (diverse && selected.size() < count) {
      selected.push_back(candidate.node);
    } else {
      pruned.push_back(candidate.node);
    }
  }

  // Filling rejected candidates keeps graph degree predictable on clustered data.
  for (const NodeIndex candidate : pruned) {
    if (selected.size() == count) {
      break;
    }
    selected.push_back(candidate);
  }
  return selected;
}

std::size_t HnswIndex::maximum_connections(const std::size_t level) const noexcept {
  // The dense base layer improves connectivity; upper layers remain sparse shortcuts.
  return level == 0 ? config_.max_connections * 2 : config_.max_connections;
}

void HnswIndex::prune_neighbors(const NodeIndex node, const std::size_t level) {
  std::vector<NodeIndex>& neighbors = graph_[node].levels[level];
  const std::size_t maximum_count = maximum_connections(level);
  if (neighbors.size() <= maximum_count) {
    return;
  }

  const std::span<const float> owner(
      vectors_.data() + static_cast<std::size_t>(node) * config_.dimension, config_.dimension);
  std::vector<Candidate> candidates;
  candidates.reserve(neighbors.size());
  for (const NodeIndex neighbor : neighbors) {
    candidates.push_back(Candidate{neighbor, similarity(owner, neighbor)});
  }
  std::sort(candidates.begin(), candidates.end(),
            [this](const Candidate& lhs, const Candidate& rhs) { return is_better(lhs, rhs); });
  const std::vector<NodeIndex> selected = select_neighbors(candidates, maximum_count);

  for (const NodeIndex previous_neighbor : neighbors) {
    if (std::find(selected.begin(), selected.end(), previous_neighbor) == selected.end()) {
      const NodeIndex removed_neighbor = previous_neighbor;
      std::vector<NodeIndex>& reverse = graph_[removed_neighbor].levels[level];
      std::erase(reverse, node);
    }
  }
  neighbors = selected;
}

}  // namespace fast_vector
