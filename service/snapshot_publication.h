#pragma once

#include <chrono>
#include <filesystem>
#include <memory>
#include <mutex>
#include <string>

#include "fast_vector/distance.h"
#include "fast_vector/v1/vector_search.pb.h"
#include "fast_vector/vector_store.h"

namespace fast_vector::service {

struct PublicationConfig {
  bool enabled = false;
  std::uint64_t maximum_file_bytes = 1024ULL * 1024 * 1024;
  std::filesystem::path staging_directory = std::filesystem::temp_directory_path();
  DotProductKernel kernel = DotProductKernel::Scalar;
};

class PublicationBusyError : public std::runtime_error {
 public:
  PublicationBusyError() : std::runtime_error("snapshot preparation already in progress") {}
};

class PublicationDisabledError : public std::logic_error {
 public:
  PublicationDisabledError() : std::logic_error("snapshot publication is disabled") {}
};

[[nodiscard]] v1::SnapshotManifest read_snapshot_manifest(const std::filesystem::path& path);
void validate_manifest(const v1::SnapshotManifest& manifest, const std::string& index_type,
                       std::uint64_t maximum_bytes);
[[nodiscard]] std::unique_ptr<VectorIndex> load_verified_snapshot(
    const std::filesystem::path& path, const v1::SnapshotManifest& manifest,
    const std::string& index_type, DotProductKernel kernel, std::uint64_t maximum_bytes);
[[nodiscard]] SnapshotIdentity snapshot_identity(const v1::SnapshotManifest& manifest,
                                                 std::uint64_t generation);

/** Owns exactly one preparation slot. Queries never acquire its mutex. */
class SnapshotPublication {
 public:
  SnapshotPublication(std::shared_ptr<VectorStore> store, std::string index_type,
                      PublicationConfig config);
  [[nodiscard]] std::uint64_t begin(const v1::SnapshotManifest& manifest, std::uint64_t generation);
  void complete(const std::filesystem::path& path, std::uint64_t token);
  void fail(std::uint64_t token) noexcept;
  void rejected(const v1::SnapshotManifest& manifest, std::uint64_t generation);
  void activate(const std::string& snapshot_id, std::uint64_t generation);
  [[nodiscard]] v1::PublicationStatus status(StoreStats* store_stats = nullptr) const;
  [[nodiscard]] const PublicationConfig& config() const noexcept { return config_; }

 private:
  std::shared_ptr<VectorStore> store_;
  std::string index_type_;
  PublicationConfig config_;
  mutable std::mutex mutex_;
  v1::PublicationStatus status_;
  v1::SnapshotManifest manifest_;
  std::uint64_t generation_ = 0;
  std::uint64_t token_ = 0;
  std::chrono::steady_clock::time_point started_;
  std::shared_ptr<const ServingSnapshot> candidate_;
};

/** Private, exclusive staging file, removed on every exit path. No caller-controlled filename. */
class SnapshotStagingFile {
 public:
  explicit SnapshotStagingFile(const std::filesystem::path& directory);
  ~SnapshotStagingFile();
  SnapshotStagingFile(const SnapshotStagingFile&) = delete;
  SnapshotStagingFile& operator=(const SnapshotStagingFile&) = delete;
  [[nodiscard]] const std::filesystem::path& path() const noexcept { return path_; }

 private:
  std::filesystem::path path_;
};
}  // namespace fast_vector::service
