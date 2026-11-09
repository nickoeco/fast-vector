#pragma once

#include <cstddef>
#include <cstdint>
#include <filesystem>
#include <functional>
#include <memory>
#include <span>
#include <stdexcept>
#include <string>
#include <vector>

#include "fast_vector/flat_index.h"
#include "fast_vector/resource_limits.h"

namespace fast_vector {

/** Test boundaries; callbacks may throw or terminate the process. Empty in normal service use. */
enum class WalBoundary {
  BeforeAppend,
  AfterPartialAppend,
  AfterAppend,
  BeforeSync,
  AfterSync,
  BeforeApply,
  AfterApply,
  BeforeResponse,
  BeforeCheckpointPublish,
  AfterCheckpointRename,
  AfterWalCreated,
  BeforeWriterSwitch,
  AfterWriterSwitch,
  BeforeWalCleanup,
  AfterWalCleanup
};

struct WalOptions {
  std::filesystem::path directory;
  std::size_t dimension = 128;
  std::uint64_t maximum_wal_bytes = 1024ULL * 1024 * 1024;
  DotProductKernel kernel = DotProductKernel::Scalar;
  std::function<void(WalBoundary)> fault_hook;
  ResourceLimits limits{};
  std::shared_ptr<MemoryBudget> loading_budget{};
  bool rotate_on_checkpoint = false;
  std::uint64_t maximum_checkpoint_bytes = 2ULL * 1024 * 1024 * 1024;
};

struct CommitResult {
  std::uint64_t lsn = 0;
  std::size_t index_size = 0;
  bool replayed = false;
};

struct DurabilityStats {
  bool enabled = false;
  bool recovery_required = false;
  std::uint64_t applied_lsn = 0;
  std::uint64_t checkpoint_lsn = 0;
  std::uint64_t recovered_records = 0;
  std::uint64_t discarded_tail_bytes = 0;
  std::uint64_t recovery_us = 0;
  std::uint64_t last_sync_us = 0;
  std::uint64_t last_write_us = 0;
  std::uint64_t last_checkpoint_us = 0;
  std::uint64_t dedup_entries = 0;
  std::uint64_t dedup_metadata_bytes = 0;
  std::uint64_t dedup_file_bytes = 0;
  std::uint64_t wal_scan_bytes = 0;
  std::uint64_t wal_generation = 0;
  std::uint64_t active_wal_bytes = 0;
  std::uint64_t cleaned_files = 0;
  bool cleanup_pending = false;
};

class RecoveryRequiredError : public std::runtime_error {
 public:
  RecoveryRequiredError() : std::runtime_error("durable writer requires restart and recovery") {}
};

class DurabilityModeError : public std::logic_error {
 public:
  explicit DurabilityModeError(const char* message) : std::logic_error(message) {}
};

/** Single-writer Linux coordinator. Caller serializes every access (VectorStore does so).
 * Owns the directory lock, WAL and FlatIndex. Construction completes recovery before returning.
 */
class DurableFlatIndex final : public VectorIndex {
 public:
  explicit DurableFlatIndex(WalOptions options);
  ~DurableFlatIndex() override;
  DurableFlatIndex(const DurableFlatIndex&) = delete;
  DurableFlatIndex& operator=(const DurableFlatIndex&) = delete;

  /** Reject an insertion without an operation ID; no volatile bypass exists. */
  void add(VectorId id, std::span<const float> vector) override;
  [[nodiscard]] CommitResult commit(const std::string& operation_id, VectorId id,
                                    std::span<const float> vector,
                                    const ResourceLimits* admission = nullptr);
  /** Pause writes through the caller's lock and publish a verified recovery point. */
  [[nodiscard]] std::uint64_t checkpoint();
  [[nodiscard]] DurabilityStats durability_stats() const noexcept;
  [[nodiscard]] std::size_t size() const noexcept override;
  [[nodiscard]] std::size_t dimension() const noexcept override;
  [[nodiscard]] std::vector<SearchResult> search(std::span<const float> query,
                                                 std::size_t k) const override;
  [[nodiscard]] std::vector<SearchResult> search(std::span<const float> query, std::size_t k,
                                                 const SearchContext& context) const override;

 private:
  class Impl;
  std::unique_ptr<Impl> impl_;
};

}  // namespace fast_vector
