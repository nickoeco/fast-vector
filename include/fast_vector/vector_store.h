#pragma once

#include <atomic>
#include <cstddef>
#include <cstdint>
#include <memory>
#include <shared_mutex>
#include <span>
#include <vector>

#include "fast_vector/vector_index.h"

namespace fast_vector {

/** An owning vector payload used by batch insertion APIs. */
struct VectorRecord {
  VectorId id;
  std::vector<float> values;
};

/** Snapshot of application-layer counters and index dimensions. */
struct StoreStats {
  std::size_t index_size;
  std::size_t dimension;
  std::uint64_t successful_queries;
  std::uint64_t failed_queries;
  std::uint64_t inserted_vectors;
};

/** Thread-safe application layer around a single VectorIndex instance. */
class VectorStore {
 public:
  explicit VectorStore(std::unique_ptr<VectorIndex> index, std::size_t maximum_batch_size = 1'000);

  void add(VectorId id, std::span<const float> vector);

  /**
   * Add a prevalidated batch while holding one exclusive lock.
   * Existing index IDs can still cause a partially applied batch.
   */
  void add_batch(std::span<const VectorRecord> vectors);

  [[nodiscard]] std::vector<SearchResult> search(std::span<const float> query, std::size_t k) const;

  [[nodiscard]] StoreStats stats() const;
  [[nodiscard]] std::size_t maximum_batch_size() const noexcept;

 private:
  void validate_batch(std::span<const VectorRecord> vectors) const;

  std::unique_ptr<VectorIndex> index_;
  std::size_t maximum_batch_size_;
  mutable std::shared_mutex mutex_;
  mutable std::atomic<std::uint64_t> successful_queries_{0};
  mutable std::atomic<std::uint64_t> failed_queries_{0};
  std::atomic<std::uint64_t> inserted_vectors_{0};
};

}  // namespace fast_vector
