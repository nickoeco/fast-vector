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
#include <string>
#include <vector>

#include "fast_vector/durable_flat_index.h"
#include "fast_vector/resource_limits.h"
#include "fast_vector/vector_index.h"

namespace fast_vector {

struct SnapshotIdentity {
  std::string snapshot_id{};
  std::uint64_t generation = 0;
  std::string embedding_revision{};
  bool operator==(const SnapshotIdentity&) const = default;
};

/** The complete immutable unit acquired by a read-only query. */
struct ServingSnapshot {
  std::shared_ptr<const VectorIndex> index;
  std::uint64_t version;
  SnapshotIdentity identity;
};

class SnapshotMismatchError : public std::logic_error {
 public:
  SnapshotMismatchError() : std::logic_error("snapshot identity or generation mismatch") {}
};

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
  SnapshotIdentity identity{};
  DurabilityStats durability{};
};

struct SnapshotSearchResult {
  std::vector<SearchResult> neighbors;
  // Process-local publication sequence; zero identifies the mutable ownership path.
  std::uint64_t snapshot_version;
  SnapshotIdentity identity{};
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
                       bool read_only = false, SnapshotIdentity identity = {},
                       ResourceLimits limits = {}, std::shared_ptr<MemoryBudget> budget = nullptr);
  void validate_top_k(std::size_t k) const;
  const ResourceLimits& resource_limits() const noexcept { return limits_; }
  std::shared_ptr<MemoryBudget> memory_budget() const noexcept { return budget_; }

  void add(VectorId id, std::span<const float> vector);
  /** Durable mode requires a stable client operation ID; volatile mode rejects one. */
  [[nodiscard]] CommitResult add(VectorId id, std::span<const float> vector,
                                 const std::string& operation_id, std::uint64_t epoch = 0);
  [[nodiscard]] std::uint64_t checkpoint(std::uint64_t expected_epoch = 0);
  /** Return the checkpoint LSN and its epoch window under one exclusive lock. */
  [[nodiscard]] CheckpointResult checkpoint_result(std::uint64_t expected_epoch = 0);
  [[nodiscard]] DurabilityStats durability_stats() const;

  /**
   * Add a prevalidated batch while holding one exclusive lock.
   * Existing index IDs can still cause a partially applied batch.
   */
  void add_batch(std::span<const VectorRecord> vectors);

  /** Publish an immutable snapshot without waiting for old read-only queries.
   * Each query retains its snapshot until search completes. Requires a read-only store.
   */
  void replace_index(std::unique_ptr<VectorIndex> index);
  /** Allocate publication metadata during preparation, leaving activation as a pointer swap. */
  [[nodiscard]] std::shared_ptr<const ServingSnapshot> prepare_snapshot(
      std::unique_ptr<VectorIndex> index, SnapshotIdentity identity);
  void activate_snapshot(std::shared_ptr<const ServingSnapshot> snapshot);
  [[nodiscard]] std::shared_ptr<const ServingSnapshot> serving_snapshot() const;

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
  std::shared_ptr<void> reserve_index(const VectorIndex& index) const;
  std::shared_ptr<const VectorIndex> account_index(std::shared_ptr<const VectorIndex> index) const;
  void check_growth(std::size_t additions) const;
  ResourceLimits limits_;
  std::shared_ptr<MemoryBudget> budget_;
  void validate_batch(std::span<const VectorRecord> vectors) const;
  [[nodiscard]] std::shared_ptr<const ServingSnapshot> acquire_snapshot() const;
  void publish_snapshot(std::shared_ptr<const VectorIndex> snapshot,
                        SnapshotIdentity identity = {});

  // Exactly one ownership path is populated, determined by the immutable read_only_ flag.
  std::unique_ptr<VectorIndex> writable_index_;
#if defined(__cpp_lib_atomic_shared_ptr) && __cpp_lib_atomic_shared_ptr >= 201711L
  std::atomic<std::shared_ptr<const ServingSnapshot>> snapshot_{nullptr};
#else
  // libstdc++ 11 lacks the C++20 specialization; access only through shared_ptr atomic APIs.
  std::shared_ptr<const ServingSnapshot> snapshot_;
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
