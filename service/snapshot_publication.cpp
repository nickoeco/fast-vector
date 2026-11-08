#include "service/snapshot_publication.h"

#include <google/protobuf/util/json_util.h>
#include <openssl/evp.h>
#include <sys/resource.h>
#include <unistd.h>

#include <array>
#include <fstream>
#include <iomanip>
#include <limits>
#include <sstream>

#include "fast_vector/flat_index_io.h"
#include "fast_vector/hnsw_index_io.h"

namespace fast_vector::service {
namespace {
bool canonical_id(const std::string& value) {
  return value.size() == 71 && value.compare(0, 7, "sha256:") == 0 &&
         value.find_first_not_of("0123456789abcdef", 7) == std::string::npos;
}
std::uint64_t elapsed(std::chrono::steady_clock::time_point start) {
  return static_cast<std::uint64_t>(std::chrono::duration_cast<std::chrono::microseconds>(
                                        std::chrono::steady_clock::now() - start)
                                        .count());
}
void identity_to_proto(const SnapshotIdentity& identity, v1::SnapshotIdentity* output) {
  output->set_snapshot_id(identity.snapshot_id);
  output->set_generation(identity.generation);
  output->set_embedding_revision(identity.embedding_revision);
}
std::string sha256_file(const std::filesystem::path& path) {
  std::ifstream input(path, std::ios::binary);
  if (!input) throw std::runtime_error("snapshot file unavailable");
  std::unique_ptr<EVP_MD_CTX, decltype(&EVP_MD_CTX_free)> digest(EVP_MD_CTX_new(),
                                                                 &EVP_MD_CTX_free);
  if (!digest || EVP_DigestInit_ex(digest.get(), EVP_sha256(), nullptr) != 1) {
    throw std::runtime_error("digest initialization failed");
  }
  std::array<char, 65536> buffer{};
  while (input.read(buffer.data(), buffer.size()) || input.gcount() != 0) {
    if (EVP_DigestUpdate(digest.get(), buffer.data(), static_cast<std::size_t>(input.gcount())) !=
        1) {
      throw std::runtime_error("digest update failed");
    }
  }
  if (!input.eof()) throw std::runtime_error("snapshot read failed");
  std::array<unsigned char, EVP_MAX_MD_SIZE> bytes{};
  unsigned length = 0;
  if (EVP_DigestFinal_ex(digest.get(), bytes.data(), &length) != 1) {
    throw std::runtime_error("digest completion failed");
  }
  std::ostringstream output;
  for (unsigned i = 0; i < length; ++i) {
    output << std::hex << std::setw(2) << std::setfill('0') << static_cast<unsigned>(bytes[i]);
  }
  return output.str();
}
}  // namespace

v1::SnapshotManifest read_snapshot_manifest(const std::filesystem::path& path) {
  if (std::filesystem::file_size(path) > 16384) throw std::invalid_argument("manifest too large");
  std::ifstream input(path);
  std::string json((std::istreambuf_iterator<char>(input)), std::istreambuf_iterator<char>());
  v1::SnapshotManifest manifest;
  if (!input || !google::protobuf::util::JsonStringToMessage(json, &manifest).ok()) {
    throw std::invalid_argument("invalid snapshot manifest JSON");
  }
  return manifest;
}

void validate_manifest(const v1::SnapshotManifest& manifest, const std::string& index_type,
                       std::uint64_t maximum_bytes) {
  if (manifest.ByteSizeLong() > 16384 || manifest.index_type() != index_type ||
      (index_type != "flat" && index_type != "hnsw") || manifest.dimension() == 0 ||
      manifest.dimension() > std::numeric_limits<std::size_t>::max() ||
      manifest.vector_count() > std::numeric_limits<std::size_t>::max() ||
      manifest.file_size() < 56 || manifest.file_size() > maximum_bytes ||
      manifest.sha256().size() != 64 || manifest.snapshot_id() != "sha256:" + manifest.sha256() ||
      manifest.embedding_revision().size() > 256 ||
      manifest.sha256().find_first_not_of("0123456789abcdef") != std::string::npos ||
      (index_type == "flat" ? manifest.format_version() != 1
                            : (manifest.format_version() < 1 || manifest.format_version() > 3))) {
    throw std::invalid_argument("invalid or incompatible snapshot manifest");
  }
}

std::unique_ptr<VectorIndex> load_verified_snapshot(const std::filesystem::path& path,
                                                    const v1::SnapshotManifest& manifest,
                                                    const std::string& index_type,
                                                    DotProductKernel kernel,
                                                    std::uint64_t maximum_bytes,
                                                    ResourceLimits limits) {
  validate_manifest(manifest, index_type, maximum_bytes);
  if (std::filesystem::file_size(path) != manifest.file_size() ||
      sha256_file(path) != manifest.sha256()) {
    throw std::invalid_argument("snapshot content does not match manifest");
  }
  std::ifstream header(path, std::ios::binary);
  std::array<unsigned char, 12> bytes{};
  header.read(reinterpret_cast<char*>(bytes.data()), bytes.size());
  const auto version = static_cast<std::uint32_t>(bytes[8]) |
                       (static_cast<std::uint32_t>(bytes[9]) << 8U) |
                       (static_cast<std::uint32_t>(bytes[10]) << 16U) |
                       (static_cast<std::uint32_t>(bytes[11]) << 24U);
  if (!header || version != manifest.format_version())
    throw std::invalid_argument("snapshot format version mismatch");
  std::unique_ptr<VectorIndex> index;
  if (index_type == "flat")
    index = std::make_unique<FlatIndex>(load_flat_index(path, kernel, limits));
  else
    index = std::make_unique<HnswIndex>(load_hnsw_index(path, kernel, limits));
  if (index->dimension() != manifest.dimension() || index->size() != manifest.vector_count()) {
    throw std::invalid_argument("snapshot shape does not match manifest");
  }
  return index;
}

SnapshotIdentity snapshot_identity(const v1::SnapshotManifest& manifest, std::uint64_t generation) {
  return {manifest.snapshot_id(), generation, manifest.embedding_revision()};
}

SnapshotPublication::SnapshotPublication(std::shared_ptr<VectorStore> store, std::string index_type,
                                         PublicationConfig config)
    : store_(std::move(store)), index_type_(std::move(index_type)), config_(std::move(config)) {
  if (!store_ || config_.maximum_file_bytes == 0 || (config_.enabled && !store_->read_only())) {
    throw std::invalid_argument("publication requires a read-only store and positive file limit");
  }
  status_.set_enabled(config_.enabled);
  if (config_.enabled && store_->serving_snapshot()->identity.generation == 0) {
    throw std::invalid_argument("publication requires a managed startup identity");
  }
  status_.set_state("SERVING");
  status_.set_last_result("NONE");
  if (store_->read_only())
    status_.set_highest_generation(store_->serving_snapshot()->identity.generation);
}

std::uint64_t SnapshotPublication::begin(const v1::SnapshotManifest& manifest,
                                         std::uint64_t generation) {
  if (!config_.enabled) throw PublicationDisabledError();
  validate_manifest(manifest, index_type_, config_.maximum_file_bytes);
  check_shape(store_->resource_limits(), manifest.dimension(), manifest.vector_count());
  std::lock_guard lock(mutex_);
  if (status_.state() == "PREPARING") throw PublicationBusyError();
  const auto current = store_->serving_snapshot();
  if (generation == 0 || generation <= current->identity.generation ||
      generation < status_.highest_generation() ||
      (generation == status_.highest_generation() &&
       manifest.SerializeAsString() != manifest_.SerializeAsString())) {
    throw SnapshotMismatchError();
  }
  if (token_ == std::numeric_limits<std::uint64_t>::max())
    throw std::length_error("prepare token exhausted");
  // Allocate status first: admission failure must not discard an existing READY candidate.
  auto next_manifest = manifest;
  auto next_status = status_;
  const auto started = std::chrono::steady_clock::now();
  next_status.set_highest_generation(generation);
  identity_to_proto(snapshot_identity(manifest, generation), next_status.mutable_candidate());
  *next_status.mutable_last_attempt() = next_status.candidate();
  next_status.set_state("PREPARING");
  next_status.set_last_result("PREPARING");
  next_status.set_prepare_us(0);
  next_status.set_activation_us(0);
  manifest_.Swap(&next_manifest);
  status_.Swap(&next_status);
  generation_ = generation;
  started_ = started;
  // Drop an older READY index before loading another one. Readers are independent.
  candidate_.reset();
  return ++token_;
}

void SnapshotPublication::complete(const std::filesystem::path& path, std::uint64_t token) {
  const auto loading =
      store_->memory_budget()->reserve(snapshot_loading_units(path, store_->resource_limits()));
  v1::SnapshotManifest manifest;
  std::uint64_t generation;
  {
    std::lock_guard lock(mutex_);
    if (token != token_ || status_.state() != "PREPARING") throw SnapshotMismatchError();
    manifest = manifest_;
    generation = generation_;
  }
  auto index = load_verified_snapshot(path, manifest, index_type_, config_.kernel,
                                      config_.maximum_file_bytes, store_->resource_limits());
  auto prepared =
      store_->prepare_snapshot(std::move(index), snapshot_identity(manifest, generation));
  std::lock_guard lock(mutex_);
  if (token != token_ || status_.state() != "PREPARING") throw SnapshotMismatchError();
  candidate_ = std::move(prepared);
  status_.set_prepare_us(elapsed(started_));
  status_.set_state("READY_TO_ACTIVATE");
  status_.set_last_result("READY_TO_ACTIVATE");
  identity_to_proto(snapshot_identity(manifest, generation), status_.mutable_last_attempt());
}

void SnapshotPublication::fail(std::uint64_t token) noexcept {
  try {
    std::lock_guard lock(mutex_);
    if (token != token_ || status_.state() != "PREPARING") return;
    candidate_.reset();
    status_.clear_candidate();
    status_.set_state("FAILED");
    status_.set_last_result("PREPARE_FAILED");
    status_.set_prepare_us(elapsed(started_));
    identity_to_proto(snapshot_identity(manifest_, generation_), status_.mutable_last_attempt());
  } catch (...) {
    // Failure telemetry must not replace the original error or disturb the serving pointer.
  }
}

void SnapshotPublication::rejected(const v1::SnapshotManifest& manifest, std::uint64_t generation) {
  std::lock_guard lock(mutex_);
  status_.set_last_result("REJECTED");
  // Rejected metadata is untrusted: never persist oversized or arbitrary request text.
  identity_to_proto(
      {canonical_id(manifest.snapshot_id()) ? manifest.snapshot_id() : "", generation,
       manifest.embedding_revision().size() <= 256 ? manifest.embedding_revision() : ""},
      status_.mutable_last_attempt());
  if (status_.state() == "SERVING" || status_.state() == "FAILED") status_.set_state("FAILED");
}

void SnapshotPublication::activate(const std::string& snapshot_id, std::uint64_t generation) {
  if (!config_.enabled) throw PublicationDisabledError();
  std::lock_guard lock(mutex_);
  const auto old = store_->serving_snapshot();
  const auto reject = [&] {
    status_.set_last_result("ACTIVATION_REJECTED");
    identity_to_proto({canonical_id(snapshot_id) ? snapshot_id : "", generation, {}},
                      status_.mutable_last_attempt());
    status_.set_activation_us(0);
    throw SnapshotMismatchError();
  };
  if (generation < status_.highest_generation()) reject();
  if (old->identity.snapshot_id == snapshot_id && old->identity.generation == generation) {
    status_.set_state("SERVING");
    status_.set_last_result("ALREADY_ACTIVE");
    status_.set_activation_us(0);
    identity_to_proto(old->identity, status_.mutable_last_attempt());
    return;
  }
  if (status_.state() != "READY_TO_ACTIVATE" || !candidate_ ||
      candidate_->identity.snapshot_id != snapshot_id ||
      candidate_->identity.generation != generation) {
    reject();
  }
  const auto start = std::chrono::steady_clock::now();
  store_->activate_snapshot(candidate_);
  status_.set_activation_us(elapsed(start));
  candidate_.reset();
  status_.mutable_last_attempt()->Swap(status_.mutable_candidate());
  status_.clear_candidate();
  status_.set_state("SERVING");
  status_.set_last_result("ACTIVATED");
  // old keeps reclamation outside the measured atomic activation interval.
}

v1::PublicationStatus SnapshotPublication::status(StoreStats* store_stats) const {
  std::lock_guard lock(mutex_);
  auto status = status_;
  const auto observed = store_->stats();
  if (store_stats) *store_stats = observed;
  identity_to_proto(observed.identity, status.mutable_current());
  rusage usage{};
  if (getrusage(RUSAGE_SELF, &usage) == 0)
    status.set_peak_rss_kib(static_cast<std::uint64_t>(usage.ru_maxrss));
  return status;
}

SnapshotStagingFile::SnapshotStagingFile(const std::filesystem::path& directory) {
  auto name = (directory / "fast-vector-prepare-XXXXXX").string();
  const int descriptor = mkstemp(name.data());
  if (descriptor < 0) throw std::runtime_error("cannot create snapshot staging file");
  try {
    path_ = name;
  } catch (...) {
    close(descriptor);
    unlink(name.c_str());
    throw;
  }
  close(descriptor);
}
SnapshotStagingFile::~SnapshotStagingFile() {
  std::error_code error;
  std::filesystem::remove(path_, error);
}
}  // namespace fast_vector::service
