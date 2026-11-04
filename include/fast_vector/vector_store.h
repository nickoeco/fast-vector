#pragma once

#include <atomic>
#include <cstddef>
#include <cstdint>
#include <memory>
#include <mutex>
#include <optional>
#include <shared_mutex>
#include <span>
#include <stdexcept>
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
  bool read_only;
  std::uint64_t snapshot_version;
};

struct SnapshotSearchResult {
  std::vector<SearchResult> neighbors;
  // Process-local publication sequence; zero identifies the mutable ownership path.
  std::uint64_t snapshot_version;
};

/** Raised when a mutating operation targets a read-only store. */
class ReadOnlyStoreError : public std::logic_error {
 public:
  ReadOnlyStoreError() : std::logic_error("vector store is read-only") {}
};

/** Thread-safe application layer around a single VectorIndex instance. */
class VectorStore {
 public:
  explicit VectorStore(std::unique_ptr<VectorIndex> index, std::size_t maximum_batch_size = 1'000,
                       bool read_only = false);

  void add(VectorId id, std::span<const float> vector);

  /**
   * Add a prevalidated batch while holding one exclusive lock.
   * Existing index IDs can still cause a partially applied batch.
   */
  void add_batch(std::span<const VectorRecord> vectors);

  /** Publish an immutable snapshot without waiting for old read-only queries.
   * Each query retains its snapshot until search completes. Requires a read-only store.
   */
  void replace_index(std::unique_ptr<VectorIndex> index);

  [[nodiscard]] std::vector<SearchResult> search(std::span<const float> query, std::size_t k) const;
  [[nodiscard]] SnapshotSearchResult search(std::span<const float> query, std::size_t k,
                                            const SearchContext& context) const;

  [[nodiscard]] StoreStats stats() const;
  /** Fixed for writable indexes; snapshot-local for read-only stores. Does not await a writer. */
  [[nodiscard]] std::size_t dimension() const;
  /** Best-effort metadata for logging; nullopt if a writable index is busy. */
  [[nodiscard]] std::optional<std::size_t> try_size() const;
  [[nodiscard]] std::size_t maximum_batch_size() const noexcept;
  [[nodiscard]] bool read_only() const noexcept;

 private:
  void validate_batch(std::span<const VectorRecord> vectors) const;
  struct IndexSnapshot {
    std::shared_ptr<const VectorIndex> index;
    std::uint64_t version;
  };
  [[nodiscard]] std::shared_ptr<const IndexSnapshot> acquire_snapshot() const;
  void publish_snapshot(std::shared_ptr<const VectorIndex> snapshot);

  // Exactly one ownership path is populated, determined by the immutable read_only_ flag.
  std::unique_ptr<VectorIndex> writable_index_;
#if defined(__cpp_lib_atomic_shared_ptr) && __cpp_lib_atomic_shared_ptr >= 201711L
  std::atomic<std::shared_ptr<const IndexSnapshot>> snapshot_{nullptr};
#else
  // libstdc++ 11 lacks the C++20 specialization; access only through shared_ptr atomic APIs.
  std::shared_ptr<const IndexSnapshot> snapshot_;
#endif
  std::size_t maximum_batch_size_;
  const bool read_only_;
  mutable std::shared_timed_mutex mutex_;
  std::mutex publication_mutex_;
  mutable std::atomic<std::uint64_t> successful_queries_{0};
  mutable std::atomic<std::uint64_t> failed_queries_{0};
  std::atomic<std::uint64_t> inserted_vectors_{0};
};

}  // namespace fast_vector
