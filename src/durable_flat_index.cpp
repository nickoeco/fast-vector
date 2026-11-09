#include "fast_vector/durable_flat_index.h"

#include <openssl/evp.h>

#include <algorithm>
#include <array>
#include <bit>
#include <chrono>
#include <cmath>
#include <cstring>
#include <fstream>
#include <limits>
#include <regex>
#include <stdexcept>
#include <unordered_map>
#include <unordered_set>
#include <utility>

#include "fast_vector/flat_index_io.h"

#if defined(__linux__)
#include <fcntl.h>
#include <sys/file.h>
#include <sys/stat.h>
#include <unistd.h>

#include <cerrno>
#include <system_error>
#endif

namespace fast_vector {

// Keep unchecked insertion private: WAL values have already passed finite/norm checks.
class WalIndexAccess {
 public:
  static void insert(FlatIndex& index, VectorId id, std::span<const float> values) {
    index.add_normalized(id, values);
  }
  static bool contains(const FlatIndex& index, VectorId id) {
    return index.id_to_row_.contains(id);
  }
  static bool matches(const FlatIndex& index, VectorId id, std::span<const float> values) {
    const auto entry = index.id_to_row_.find(id);
    return entry != index.id_to_row_.end() && values.size() == index.dimension_ &&
           std::memcmp(index.vectors_.data() + entry->second * index.dimension_, values.data(),
                       values.size_bytes()) == 0;
  }
};

#if defined(__linux__)
namespace {
using Bytes = std::vector<unsigned char>;
using Clock = std::chrono::steady_clock;
constexpr std::uint64_t kOffset = 14695981039346656037ULL;
constexpr std::uint64_t kPrime = 1099511628211ULL;
constexpr std::size_t kHeader = 40;
constexpr std::size_t kManifest = 72;
constexpr std::size_t kManifestV2 = 88;
constexpr std::size_t kManifestV3 = 168;
constexpr std::size_t kManifestV4 = 208;
constexpr std::size_t kSegmentHeader = 72;
constexpr std::size_t kDedupHeader = 56;
constexpr std::array<unsigned char, 8> kDedupMagic{'F', 'V', 'D', 'E', 'D', 'U', 'P', 0};
constexpr std::size_t kMaximumDimension = 1'000'000;
constexpr std::size_t kMaximumOperationId = 256;
constexpr std::array<unsigned char, 8> kWalMagic{'F', 'V', 'W', 'A', 'L', '0', '1', 0};
constexpr std::array<unsigned char, 8> kCheckpointMagic{'F', 'V', 'R', 'E', 'C', '0', '1', 0};

std::uint64_t elapsed(Clock::time_point start) {
  return static_cast<std::uint64_t>(
      std::chrono::duration_cast<std::chrono::microseconds>(Clock::now() - start).count());
}
[[noreturn]] void invalid(const char* message) {
  throw std::runtime_error(std::string("invalid recovery data: ") + message);
}
[[noreturn]] void system_failure(const char* message) {
  throw std::system_error(errno, std::generic_category(), message);
}
std::uint64_t hash(std::span<const unsigned char> bytes, std::uint64_t value = kOffset) {
  for (auto byte : bytes) {
    value ^= byte;
    value *= kPrime;
  }
  return value;
}
void put(Bytes& bytes, std::uint64_t value, std::size_t width = 8) {
  for (std::size_t i = 0; i < width; ++i) {
    bytes.push_back(value & 255U);
    value >>= 8U;
  }
}
std::uint64_t get(std::span<const unsigned char> bytes, std::size_t offset, std::size_t width = 8) {
  if (offset > bytes.size() || width > bytes.size() - offset) invalid("truncated field");
  std::uint64_t value = 0;
  for (std::size_t i = 0; i < width; ++i) value |= std::uint64_t{bytes[offset + i]} << (8 * i);
  return value;
}
void checksum(Bytes& bytes) { put(bytes, hash(bytes)); }
void verify_checksum(std::span<const unsigned char> bytes) {
  if (bytes.size() < 8 || get(bytes, bytes.size() - 8) != hash(bytes.first(bytes.size() - 8)))
    invalid("checksum mismatch");
}

class File {
 public:
  explicit File(int fd) : fd_(fd) {
    if (fd < 0) system_failure("open recovery file");
  }
  ~File() {
    if (fd_ >= 0) ::close(fd_);
  }
  File(const File&) = delete;
  File& operator=(const File&) = delete;
  File(File&& other) noexcept : fd_(std::exchange(other.fd_, -1)) {}
  int fd() const noexcept { return fd_; }

 private:
  int fd_;
};
void write_all(int fd, std::span<const unsigned char> bytes) {
  while (!bytes.empty()) {
    const auto count = ::write(fd, bytes.data(), bytes.size());
    if (count < 0 && errno == EINTR) continue;
    if (count <= 0) system_failure("append recovery file");
    bytes = bytes.subspan(static_cast<std::size_t>(count));
  }
}
void read_all(int fd, std::span<unsigned char> bytes, std::uint64_t offset) {
  while (!bytes.empty()) {
    const auto count = ::pread(fd, bytes.data(), bytes.size(), static_cast<off_t>(offset));
    if (count < 0 && errno == EINTR) continue;
    if (count < 0) system_failure("read recovery file");
    if (count == 0) invalid("unexpected end of file");
    bytes = bytes.subspan(static_cast<std::size_t>(count));
    offset += static_cast<std::uint64_t>(count);
  }
}
std::uint64_t file_size(int fd) {
  struct stat info {};
  if (fstat(fd, &info) != 0) system_failure("stat recovery file");
  if (!S_ISREG(info.st_mode) || info.st_size < 0) invalid("not a regular file");
  return static_cast<std::uint64_t>(info.st_size);
}
void sync(int fd, bool data_only = false) {
  int result;
  do {
    result = data_only ? fdatasync(fd) : fsync(fd);
  } while (result < 0 && errno == EINTR);
  if (result != 0) system_failure("sync recovery file");
}
void sync_directory(const std::filesystem::path& path) {
  File directory(::open(path.c_str(), O_RDONLY | O_DIRECTORY | O_CLOEXEC));
  sync(directory.fd());
}
void verify_private_directory(const std::filesystem::path& path) {
  struct stat info {};
  if (::stat(path.c_str(), &info) != 0) system_failure("stat WAL directory");
  if (!S_ISDIR(info.st_mode) || info.st_uid != geteuid() || (info.st_mode & (S_IWGRP | S_IWOTH)))
    throw std::invalid_argument(
        "WAL rotation requires an owned directory without group/other write permission");
}
class Temporary {
 public:
  explicit Temporary(const std::filesystem::path& directory) {
    auto name = (directory / ".recovery-XXXXXX").string();
    std::vector<char> buffer(name.begin(), name.end());
    buffer.push_back(0);
    File file(mkstemp(buffer.data()));
    try {
      path_ = buffer.data();
    } catch (...) {
      unlink(buffer.data());
      throw;
    }
  }
  ~Temporary() {
    // Cleanup must remain allocation-free during injected allocation failures.
    ::unlink(path_.c_str());
  }
  const std::filesystem::path& path() const { return path_; }

 private:
  std::filesystem::path path_;
};
std::uint64_t file_hash(const std::filesystem::path& path) {
  File file(::open(path.c_str(), O_RDONLY | O_CLOEXEC | O_NOFOLLOW));
  const auto size = file_size(file.fd());
  std::array<unsigned char, 65536> buffer{};
  auto value = kOffset;
  for (std::uint64_t offset = 0; offset < size;) {
    const auto count =
        static_cast<std::size_t>(std::min<std::uint64_t>(buffer.size(), size - offset));
    auto bytes = std::span(buffer).first(count);
    read_all(file.fd(), bytes, offset);
    value = hash(bytes, value);
    offset += count;
  }
  return value;
}
struct WalRecord {
  std::uint64_t lsn;
  VectorId id;
  std::string operation_id;
  std::vector<float> original;
  std::vector<float> normalized;
  std::uint64_t epoch = 0;
};
struct DedupEntry {
  std::uint64_t lsn;
  VectorId id;
  std::uint64_t offset;
  std::array<unsigned char, 32> digest;
  bool checkpoint = false;
};
using DedupTable = std::unordered_map<std::string, DedupEntry>;
struct EpochPolicy {
  std::uint64_t current = 0, oldest = 0, retained = 0, operations = 0, bytes = 0;
};
std::string operation_key(std::uint64_t epoch, std::string_view id) {
  std::string key(8, '\0');
  for (std::size_t i = 0; i < 8; ++i) {
    key[i] = static_cast<char>(epoch & 255U);
    epoch >>= 8;
  }
  key.append(id);
  return key;
}
std::uint64_t key_epoch(const std::string& key) {
  return get(std::span(reinterpret_cast<const unsigned char*>(key.data()), key.size()), 0);
}
std::uint64_t entry_bytes(const std::string& key, std::size_t dimension) {
  return resource_add(80 + key.size(), resource_multiply(dimension, 4));
}
void validate_policy(const EpochPolicy& policy) {
  if (policy.current == 0 || policy.retained == 0 || policy.retained > 64 ||
      policy.operations == 0 || policy.bytes == 0 ||
      policy.oldest !=
          (policy.current >= policy.retained ? policy.current - policy.retained + 1 : 0))
    invalid("dedup epoch policy");
  (void)resource_multiply(policy.retained, policy.operations);
  (void)resource_multiply(policy.retained, policy.bytes);
}
void validate_retained_table(const DedupTable& table, const EpochPolicy& policy,
                             std::size_t dimension, bool pruning = false) {
  if (!policy.current) return;
  validate_policy(policy);
  std::array<std::uint64_t, 64> counts{}, bytes{};
  for (const auto& [key, entry] : table) {
    (void)entry;
    const auto epoch = key_epoch(key);
    if (pruning && epoch < policy.oldest) continue;
    if (epoch < policy.oldest || epoch > policy.current)
      invalid("dedup entry epoch outside window");
    const auto slot = epoch % 64;
    bytes[slot] = resource_add(bytes[slot], entry_bytes(key, dimension));
    if (++counts[slot] > policy.operations || bytes[slot] > policy.bytes) throw ResourceExhausted();
  }
}
std::array<unsigned char, 32> request_digest(std::span<const unsigned char> bytes) {
  std::array<unsigned char, 32> result{};
  unsigned size = 0;
  if (EVP_Digest(bytes.data(), bytes.size(), result.data(), &size, EVP_sha256(), nullptr) != 1 ||
      size != 32)
    throw std::runtime_error("SHA-256 failed");
  return result;
}
Bytes original_bits(std::span<const float> values) {
  Bytes bytes;
  bytes.reserve(values.size_bytes());
  for (float value : values) put(bytes, std::bit_cast<std::uint32_t>(value), 4);
  return bytes;
}
Bytes read_request(const DedupEntry& entry, int wal_fd, int checkpoint_fd, std::size_t dimension) {
  const auto fd = entry.checkpoint ? checkpoint_fd : wal_fd;
  if (fd < 0) invalid("missing dedup source");
  Bytes bytes(dimension * 4);
  read_all(fd, bytes, entry.offset);
  if (request_digest(bytes) != entry.digest) invalid("dedup request digest mismatch");
  return bytes;
}
Bytes encode(const WalRecord& record) {
  const auto key =
      record.epoch ? operation_key(record.epoch, record.operation_id) : record.operation_id;
  Bytes bytes;
  bytes.reserve(48 + key.size() + record.original.size() * 8);
  put(bytes, 48 + key.size() + record.original.size() * 8);
  put(bytes, record.lsn);
  put(bytes, record.epoch ? 2 : 1, 4);
  put(bytes, key.size(), 4);
  put(bytes, record.id);
  put(bytes, record.original.size());
  bytes.insert(bytes.end(), key.begin(), key.end());
  for (const auto& values : {std::span(record.original), std::span(record.normalized)})
    for (float value : values) put(bytes, std::bit_cast<std::uint32_t>(value), 4);
  checksum(bytes);
  return bytes;
}

/** Validates fixed fields before allocating a record-sized buffer. */
class WalReader {
 public:
  WalReader(int fd, std::size_t dimension, std::uint64_t epoch = 0)
      : fd_(fd),
        dimension_(dimension),
        size_(file_size(fd)),
        epochs_(epoch != 0),
        expected_epoch_(epoch) {}
  bool next(WalRecord& record) {
    if (offset == size_) return false;
    const auto remaining = size_ - offset;
    if (remaining < 8) {
      tail = remaining;
      return false;
    }
    std::array<unsigned char, 48> prefix{};
    read_all(fd_, std::span(prefix).first(std::min<std::uint64_t>(remaining, prefix.size())),
             offset);
    const auto length = get(prefix, 0);
    const auto minimum = 49 + (epochs_ ? 8 : 0) + dimension_ * 8;
    const auto maximum = 48 + kMaximumOperationId + (epochs_ ? 8 : 0) + dimension_ * 8;
    if (length < minimum || length > maximum) invalid("record length outside configured bounds");
    // Validate every available fixed field even when the trailing payload is incomplete.
    if (remaining >= 16 && get(prefix, 8) != next_lsn) invalid("non-contiguous LSN");
    if (remaining >= 20 && get(prefix, 16, 4) != (epochs_ ? 2 : 1))
      invalid("unsupported operation");
    if (remaining >= 24) {
      const auto op_length = get(prefix, 20, 4);
      if (op_length < (epochs_ ? 9U : 1U) || op_length > kMaximumOperationId + (epochs_ ? 8 : 0) ||
          length != 48 + op_length + dimension_ * 8)
        invalid("invalid operation ID or record length");
    }
    if (remaining >= 40 && get(prefix, 32) != dimension_) invalid("dimension mismatch");
    if (epochs_ && remaining >= 48 && get(prefix, 40) != expected_epoch_)
      invalid("WAL epoch mismatch");
    if (length > remaining) {
      tail = remaining;
      return false;
    }
    Bytes bytes(static_cast<std::size_t>(length));
    read_all(fd_, bytes, offset);
    verify_checksum(bytes);
    record.lsn = get(bytes, 8);
    record.id = get(bytes, 24);
    const auto op_length = static_cast<std::size_t>(get(bytes, 20, 4));
    record.epoch = epochs_ ? get(bytes, 40) : 0;
    if (epochs_ && record.epoch == 0) invalid("zero epoch WAL record");
    record.operation_id.assign(reinterpret_cast<const char*>(bytes.data() + 40 + (epochs_ ? 8 : 0)),
                               op_length - (epochs_ ? 8 : 0));
    record.original.resize(dimension_);
    record.normalized.resize(dimension_);
    auto cursor = 40 + op_length;
    original_offset = offset + cursor;
    for (auto* values : {&record.original, &record.normalized})
      for (auto& value : *values) {
        value = std::bit_cast<float>(static_cast<std::uint32_t>(get(bytes, cursor, 4)));
        if (!std::isfinite(value)) invalid("non-finite vector");
        cursor += 4;
      }
    double norm = 0, original_norm = 0;
    for (auto value : record.normalized) norm += static_cast<double>(value) * value;
    for (auto value : record.original) original_norm += static_cast<double>(value) * value;
    if (std::abs(norm - 1.0) > 1e-3 || original_norm == 0) invalid("invalid vector norm");
    digest = hash(bytes, digest);
    offset += length;
    ++next_lsn;
    return true;
  }
  std::uint64_t offset = kHeader, next_lsn = 1, tail = 0, digest = kOffset;
  std::uint64_t original_offset = 0;

 private:
  int fd_;
  std::size_t dimension_;
  std::uint64_t size_;
  bool epochs_;
  std::uint64_t expected_epoch_;
};

class WalWriter {
 public:
  explicit WalWriter(const WalOptions& options, std::string name = "operations.wal")
      : file(
            ::open((options.directory / name).c_str(), O_RDWR | O_APPEND | O_CLOEXEC | O_NOFOLLOW)),
        name(std::move(name)),
        options_(options) {}
  void append(std::span<const unsigned char> bytes) {
    if (bytes.size() > options_.maximum_wal_bytes - offset)
      throw std::length_error("WAL size limit reached; rotation is not implemented");
    hit(WalBoundary::BeforeAppend);
    const auto half = bytes.size() / 2;
    write_all(file.fd(), bytes.first(half));
    hit(WalBoundary::AfterPartialAppend);
    write_all(file.fd(), bytes.subspan(half));
    hit(WalBoundary::AfterAppend);
    hit(WalBoundary::BeforeSync);
    const auto start = Clock::now();
    sync(file.fd(), true);
    sync_us = elapsed(start);
    hit(WalBoundary::AfterSync);
    digest = hash(bytes, digest);
    offset += bytes.size();
  }
  void hit(WalBoundary boundary) const {
    if (options_.fault_hook) options_.fault_hook(boundary);
  }
  File file;
  std::string name;
  std::uint64_t generation = 0;
  std::uint64_t offset = kHeader, digest = kOffset, sync_us = 0;

 private:
  const WalOptions& options_;
};

struct RecoveryPoint {
  std::uint64_t lsn = 0, size = 0, digest = 0, wal_offset = kHeader, wal_digest = 0;
  bool present = false;
  std::uint64_t dedup_size = 0, dedup_digest = 0;
  std::uint64_t generation = 0, active_header_digest = 0;
  std::string active_name{};
  EpochPolicy policy{};
};
std::filesystem::path checkpoint_path(const WalOptions& options, std::uint64_t lsn) {
  return options.directory / ("checkpoint-" + std::to_string(lsn) + ".fv");
}
std::filesystem::path dedup_path(const WalOptions& options, std::uint64_t lsn,
                                 std::uint64_t generation = 0) {
  return options.directory / ("dedup-" + std::to_string(lsn) +
                              (generation ? "-" + std::to_string(generation) : "") + ".fvd");
}

std::uint64_t segment_nonce(const std::string& name) {
  std::uint64_t value = 0;
  for (std::size_t i = 0; i < 6; ++i)
    value |= std::uint64_t{static_cast<unsigned char>(name[name.size() - 6 + i])} << (i * 8);
  return value;
}
Bytes segment_header(const WalOptions& options, std::uint64_t generation, std::uint64_t lsn,
                     const std::string& name, bool epochs) {
  Bytes bytes(kWalMagic.begin(), kWalMagic.end());
  put(bytes, epochs ? 3 : 2, 4);
  put(bytes, kSegmentHeader, 4);
  put(bytes, options.dimension);
  put(bytes, 48 + kMaximumOperationId + (epochs ? 8 : 0) + options.dimension * 8);
  put(bytes, resource_add(lsn, 1));
  put(bytes, generation);
  put(bytes, lsn);
  put(bytes, segment_nonce(name));
  checksum(bytes);
  return bytes;
}
std::unique_ptr<WalWriter> create_segment(const WalOptions& options, std::uint64_t generation,
                                          std::uint64_t lsn, bool epochs) {
  auto pattern = (options.directory / ("wal-" + std::to_string(generation) + "-XXXXXX")).string();
  std::vector<char> name(pattern.begin(), pattern.end());
  name.push_back(0);
  File file(mkstemp(name.data()));
  // Orphans are left for reference-based cleanup after a durable publication.
  const auto basename = std::filesystem::path(name.data()).filename().string();
  const auto bytes = segment_header(options, generation, lsn, basename, epochs);
  write_all(file.fd(), bytes);
  sync(file.fd());
  sync_directory(options.directory);
  auto result = std::make_unique<WalWriter>(options, basename);
  result->generation = generation;
  result->offset = kSegmentHeader;
  result->digest = hash(bytes);
  return result;
}
bool valid_segment_name(const std::string& name, std::uint64_t generation) {
  const auto prefix = "wal-" + std::to_string(generation) + "-";
  return name.starts_with(prefix) && name.size() == prefix.size() + 6 &&
         name.find_first_not_of("0123456789abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ",
                                prefix.size()) == std::string::npos;
}

class DedupCheckpoint {
 public:
  static std::unique_ptr<File> load(const WalOptions& options, const RecoveryPoint& point,
                                    const FlatIndex& index, DedupTable& table) {
    const auto path = dedup_path(options, point.lsn, point.generation);
    File source(::open(path.c_str(), O_RDONLY | O_CLOEXEC | O_NOFOLLOW));
    auto file = std::make_unique<File>(std::move(source));
    const bool epochs = point.policy.current != 0;
    const std::size_t header_size = epochs ? 64 : kDedupHeader;
    if (point.dedup_size < header_size ||
        point.dedup_size >
            (point.generation
                 ? options.maximum_checkpoint_bytes
                 : std::max<std::uint64_t>(kDedupHeader,
                                           resource_multiply(options.maximum_wal_bytes, 2))) ||
        file_size(file->fd()) != point.dedup_size || file_hash(path) != point.dedup_digest)
      invalid("dedup file size or checksum");
    Bytes header(header_size);
    read_all(file->fd(), header, 0);
    verify_checksum(header);
    if (!std::equal(kDedupMagic.begin(), kDedupMagic.end(), header.begin()) ||
        get(header, 8, 4) != (epochs ? 2 : 1) || get(header, 12, 4) != header_size ||
        get(header, 16) != options.dimension || (!epochs && get(header, 24) != point.lsn) ||
        get(header, 32) != point.wal_offset || get(header, 40) != point.wal_digest)
      invalid("dedup header or checkpoint boundary");
    const auto count = get(header, 24);
    if (count > point.lsn || (epochs && get(header, 48) != point.lsn) ||
        (epochs && count > resource_multiply(point.policy.retained, point.policy.operations)))
      invalid("dedup count");
    std::uint64_t offset = header_size, previous_lsn = 0;
    std::unordered_set<VectorId> ids;
    for (std::uint64_t row = 0; row < count; ++row) {
      if (offset > point.dedup_size || point.dedup_size - offset < 72)
        invalid("dedup record truncated");
      std::array<unsigned char, 72> fixed{};
      read_all(file->fd(), fixed, offset);
      const auto length = get(fixed, 0), key_size = get(fixed, 24, 4);
      const auto lsn = get(fixed, 8);
      if (key_size < (epochs ? 9U : 1U) || key_size > kMaximumOperationId + (epochs ? 8 : 0) ||
          (epochs ? (lsn <= previous_lsn || lsn > point.lsn) : lsn != row + 1) ||
          get(fixed, 28, 4) != 0 || get(fixed, 32) != options.dimension ||
          length != 80 + key_size + options.dimension * 4 || length > point.dedup_size - offset)
        invalid("dedup record fields");
      Bytes bytes(static_cast<std::size_t>(length));
      read_all(file->fd(), bytes, offset);
      verify_checksum(bytes);
      std::string key(reinterpret_cast<const char*>(bytes.data() + 72),
                      static_cast<std::size_t>(key_size));
      if (!epochs) key = operation_key(0, key);
      DedupEntry entry{lsn, get(bytes, 16), offset + 72 + key_size, {}, true};
      std::copy_n(bytes.begin() + 40, 32, entry.digest.begin());
      const auto raw =
          std::span<const unsigned char>(bytes).subspan(72 + key_size, options.dimension * 4);
      if (request_digest(raw) != entry.digest || !WalIndexAccess::contains(index, entry.id) ||
          !ids.insert(entry.id).second || !table.emplace(std::move(key), entry).second)
        invalid("dedup duplicate, digest or vector ID");
      double norm = 0;
      for (std::size_t i = 0; i < options.dimension; ++i) {
        const auto value = std::bit_cast<float>(static_cast<std::uint32_t>(get(raw, i * 4, 4)));
        if (!std::isfinite(value)) invalid("dedup non-finite input");
        norm += static_cast<double>(value) * value;
      }
      if (norm == 0) invalid("dedup zero input");
      offset += length;
      previous_lsn = lsn;
    }
    if (offset != point.dedup_size) invalid("dedup trailing data");
    validate_retained_table(table, point.policy, options.dimension);
    return file;
  }
  static void save(const WalOptions& options, std::uint64_t lsn, WalWriter& writer,
                   const DedupTable& table, int old_checkpoint_fd,
                   const std::filesystem::path& path, EpochPolicy policy) {
    File output(::open(path.c_str(), O_WRONLY | O_TRUNC | O_CLOEXEC));
    Bytes header(kDedupMagic.begin(), kDedupMagic.end());
    const bool epochs = policy.current != 0;
    put(header, epochs ? 2 : 1, 4);
    put(header, epochs ? 64 : kDedupHeader, 4);
    put(header, options.dimension);
    std::uint64_t count = 0;
    for (const auto& [key, entry] : table) {
      (void)entry;
      if (!epochs || key_epoch(key) >= policy.oldest) ++count;
    }
    put(header, count);
    put(header, writer.offset);
    put(header, writer.digest);
    if (epochs) put(header, lsn);
    checksum(header);
    write_all(output.fd(), header);
    std::vector<const DedupTable::value_type*> ordered;
    ordered.reserve(table.size());
    for (const auto& entry : table)
      if (!epochs || key_epoch(entry.first) >= policy.oldest) ordered.push_back(&entry);
    std::sort(ordered.begin(), ordered.end(),
              [](const auto* a, const auto* b) { return a->second.lsn < b->second.lsn; });
    std::uint64_t expected = 1;
    for (const auto* item : ordered) {
      const auto& [internal_key, entry] = *item;
      const auto key = std::string_view(internal_key).substr(epochs ? 0 : 8);
      if (!epochs && entry.lsn != expected++) invalid("dedup LSN sequence");
      auto raw = read_request(entry, writer.file.fd(), old_checkpoint_fd, options.dimension);
      Bytes bytes;
      bytes.reserve(80 + key.size() + raw.size());
      put(bytes, 80 + key.size() + raw.size());
      put(bytes, entry.lsn);
      put(bytes, entry.id);
      put(bytes, key.size(), 4);
      put(bytes, 0, 4);
      put(bytes, options.dimension);
      bytes.insert(bytes.end(), entry.digest.begin(), entry.digest.end());
      bytes.insert(bytes.end(), key.begin(), key.end());
      bytes.insert(bytes.end(), raw.begin(), raw.end());
      checksum(bytes);
      write_all(output.fd(), bytes);
    }
    if (!epochs && expected - 1 != lsn) invalid("dedup count mismatch");
    sync(output.fd());
  }
};

class RecoveryManager {
 public:
  static RecoveryPoint checkpoint(const WalOptions& options, FlatIndex& index) {
    const auto path = options.directory / "recovery.manifest";
    if (!std::filesystem::exists(path)) return {};
    File file(::open(path.c_str(), O_RDONLY | O_CLOEXEC | O_NOFOLLOW));
    const auto size = file_size(file.fd());
    if (size != kManifest && size != kManifestV2 && size != kManifestV3 && size != kManifestV4)
      invalid("recovery manifest length");
    Bytes bytes(static_cast<std::size_t>(size));
    read_all(file.fd(), bytes, 0);
    verify_checksum(bytes);
    if (!std::equal(kCheckpointMagic.begin(), kCheckpointMagic.end(), bytes.begin()) ||
        get(bytes, 8, 4) !=
            (size == kManifest ? 1 : (size == kManifestV2 ? 2 : (size == kManifestV3 ? 3 : 4))) ||
        get(bytes, 12, 4) != size || get(bytes, 16) != options.dimension)
      invalid("recovery manifest configuration");
    RecoveryPoint point{get(bytes, 24), get(bytes, 32), get(bytes, 40),
                        get(bytes, 48), get(bytes, 56), true};
    if (size >= kManifestV2) {
      point.dedup_size = get(bytes, 64);
      point.dedup_digest = get(bytes, 72);
      if (point.dedup_size < kDedupHeader) invalid("missing v2 dedup checkpoint");
    }
    if (size >= kManifestV3) {
      point.generation = get(bytes, 80);
      point.active_header_digest = get(bytes, 88);
      const auto first = bytes.begin() + 96, last = bytes.begin() + 160;
      const auto zero = std::find(first, last, 0);
      if (zero == last || std::any_of(zero, last, [](auto value) { return value != 0; }))
        invalid("segment name padding");
      point.active_name.assign(first, zero);
      if (point.generation == 0 || !valid_segment_name(point.active_name, point.generation))
        invalid("segment identity");
    }
    if (size == kManifestV4) {
      point.policy = {get(bytes, 160), get(bytes, 168), get(bytes, 176), get(bytes, 184),
                      get(bytes, 192)};
      validate_policy(point.policy);
    }
    if (point.wal_offset < kHeader ||
        (!point.generation && point.wal_offset > options.maximum_wal_bytes) ||
        point.wal_offset > static_cast<std::uint64_t>(std::numeric_limits<off_t>::max()) ||
        (!point.generation &&
         point.lsn > (point.wal_offset - kHeader) / (49 + options.dimension * 8)) ||
        point.size > (point.generation ? options.maximum_checkpoint_bytes
                                       : std::max<std::uint64_t>(56, options.maximum_wal_bytes)))
      invalid("recovery manifest bounds");
    if (point.generation &&
        resource_add(point.size, point.dedup_size) > options.maximum_checkpoint_bytes)
      throw ResourceExhausted();
    const auto snapshot = checkpoint_path(options, point.lsn);
    check_wal_shape(options.limits, options.dimension, point.lsn);
    const auto units = snapshot_loading_units(snapshot, options.limits);
    auto lease = options.loading_budget ? options.loading_budget->reserve(units) : nullptr;
    if (std::filesystem::file_size(snapshot) != point.size || file_hash(snapshot) != point.digest)
      invalid("checkpoint content mismatch");
    File snapshot_file(::open(snapshot.c_str(), O_RDONLY | O_CLOEXEC | O_NOFOLLOW));
    std::array<unsigned char, 56> header{};
    read_all(snapshot_file.fd(), header, 0);
    if (get(header, 24) != options.dimension || get(header, 32) != point.lsn)
      invalid("checkpoint header shape mismatch");
    index = load_flat_index(snapshot, options.kernel, options.limits);
    if (index.dimension() != options.dimension || index.size() != point.lsn)
      invalid("checkpoint shape mismatch");
    return point;
  }
};

class CheckpointManager {
 public:
  static void publish(const WalOptions& options, const FlatIndex& index, std::uint64_t lsn,
                      WalWriter& writer, bool& manifest_renamed, const DedupTable& table,
                      int old_checkpoint_fd, DedupTable& next, std::unique_ptr<File>& next_file,
                      std::uint64_t& next_size, std::unique_ptr<WalWriter>& next_writer,
                      RecoveryPoint& next_point, EpochPolicy policy) {
    const bool epochs = policy.current != 0;
    const bool rotate = epochs || options.rotate_on_checkpoint || writer.generation != 0;
    if (epochs) validate_retained_table(table, policy, options.dimension, true);
    if (rotate) {
      // Bound paired checkpoint output before writing either file.
      const auto vector_bytes =
          resource_multiply(lsn, resource_add(8, resource_multiply(options.dimension, 4)));
      std::uint64_t dedup_bytes = epochs ? 64 : kDedupHeader;
      for (const auto& [key, entry] : table) {
        (void)entry;
        if (epochs && key_epoch(key) < policy.oldest) continue;
        dedup_bytes =
            resource_add(dedup_bytes, resource_add(80 + key.size() - (epochs ? 0 : 8),
                                                   resource_multiply(options.dimension, 4)));
      }
      if (resource_add(resource_add(56, vector_bytes), dedup_bytes) >
          options.maximum_checkpoint_bytes)
        throw ResourceExhausted();
      next_writer = create_segment(options, resource_add(writer.generation, 1), lsn, epochs);
      writer.hit(WalBoundary::AfterWalCreated);
    }
    const auto generation = next_writer ? next_writer->generation : 0;
    Temporary temporary(options.directory);
    save_flat_index(index, temporary.path());
    const auto workspace = options.loading_budget
                               ? options.loading_budget->reserve(resource_add(
                                     snapshot_loading_units(temporary.path(), options.limits),
                                     resource_multiply(table.size(), 1024)))
                               : nullptr;
    const auto verified = load_flat_index(temporary.path(), options.kernel, options.limits);
    if (verified.size() != index.size()) invalid("checkpoint verification");
    File snapshot(::open(temporary.path().c_str(), O_RDONLY | O_CLOEXEC));
    sync(snapshot.fd());
    const auto digest = file_hash(temporary.path());
    const auto size = file_size(snapshot.fd());
    const auto destination = checkpoint_path(options, lsn);
    if (::link(temporary.path().c_str(), destination.c_str()) != 0) {
      if (errno != EEXIST) system_failure("publish immutable checkpoint");
      if (std::filesystem::file_size(destination) != size || file_hash(destination) != digest)
        invalid("existing checkpoint differs");
    }
    sync_directory(options.directory);
    Temporary dedup(options.directory);
    DedupCheckpoint::save(options, lsn, writer, table, old_checkpoint_fd, dedup.path(), policy);
    const auto dedup_size = std::filesystem::file_size(dedup.path());
    const auto dedup_digest = file_hash(dedup.path());
    const auto dedup_destination = dedup_path(options, lsn, generation);
    if (::link(dedup.path().c_str(), dedup_destination.c_str()) != 0) {
      if (errno != EEXIST) system_failure("publish dedup checkpoint");
      if (std::filesystem::file_size(dedup_destination) != dedup_size ||
          file_hash(dedup_destination) != dedup_digest)
        invalid("existing dedup checkpoint differs");
    }
    RecoveryPoint point{lsn,           size, digest,     writer.offset,
                        writer.digest, true, dedup_size, dedup_digest};
    point.generation = generation;
    point.policy = policy;
    if (next_writer) {
      point.active_name = next_writer->name;
      point.active_header_digest = next_writer->digest;
    }
    next_file = DedupCheckpoint::load(options, point, index, next);
    next_size = dedup_size;
    sync_directory(options.directory);
    Bytes bytes(kCheckpointMagic.begin(), kCheckpointMagic.end());
    put(bytes, epochs ? 4 : (rotate ? 3 : 2), 4);
    put(bytes, epochs ? kManifestV4 : (rotate ? kManifestV3 : kManifestV2), 4);
    put(bytes, options.dimension);
    put(bytes, lsn);
    put(bytes, size);
    put(bytes, digest);
    put(bytes, writer.offset);
    put(bytes, writer.digest);
    put(bytes, dedup_size);
    put(bytes, dedup_digest);
    if (rotate) {
      put(bytes, point.generation);
      put(bytes, point.active_header_digest);
      bytes.insert(bytes.end(), point.active_name.begin(), point.active_name.end());
      bytes.resize(160, 0);
    }
    if (epochs) {
      put(bytes, policy.current);
      put(bytes, policy.oldest);
      put(bytes, policy.retained);
      put(bytes, policy.operations);
      put(bytes, policy.bytes);
    }
    checksum(bytes);
    next_point = std::move(point);
    Temporary manifest(options.directory);
    File file(::open(manifest.path().c_str(), O_WRONLY | O_TRUNC | O_CLOEXEC));
    write_all(file.fd(), bytes);
    sync(file.fd());
    writer.hit(WalBoundary::BeforeCheckpointPublish);
    if (::rename(manifest.path().c_str(), (options.directory / "recovery.manifest").c_str()) != 0)
      system_failure("publish recovery manifest");
    manifest_renamed = true;
    writer.hit(WalBoundary::AfterCheckpointRename);
    sync_directory(options.directory);
  }
};
}  // namespace

class DurableFlatIndex::Impl {
 public:
  explicit Impl(WalOptions config)
      : options(std::move(config)),
        index(options.dimension, options.kernel),
        lock_file(open_lock(options)) {
    const auto start = Clock::now();
    if (::flock(lock_file.fd(), LOCK_EX | LOCK_NB) != 0) system_failure("lock WAL directory");
    sync_directory(options.directory);
    sync_directory(options.directory.has_parent_path() ? options.directory.parent_path() : ".");
    auto point = RecoveryManager::checkpoint(options, index);
    if (point.policy.current) {
      if (options.dedup_retention && (options.retained_epochs != point.policy.retained ||
                                      options.maximum_epoch_operations != point.policy.operations ||
                                      options.maximum_epoch_bytes != point.policy.bytes))
        throw std::invalid_argument("persisted dedup policy differs from requested policy");
      options.dedup_retention = true;
      options.retained_epochs = point.policy.retained;
      options.maximum_epoch_operations = point.policy.operations;
      options.maximum_epoch_bytes = point.policy.bytes;
    }
    if (point.generation) verify_private_directory(options.directory);
    if (!point.present) {
      // Never bootstrap an empty/obsolete legacy WAL after losing a segmented root.
      const bool legacy = std::filesystem::exists(options.directory / "operations.wal");
      static const std::regex segment("^wal-[0-9]+-[A-Za-z0-9]{6}$");
      for (const auto& entry : std::filesystem::directory_iterator(options.directory)) {
        if (!std::regex_match(entry.path().filename().string(), segment)) continue;
        if (!std::filesystem::is_regular_file(entry.symlink_status()) || !legacy ||
            entry.file_size() > kSegmentHeader)
          invalid("missing recovery manifest for segmented WAL");
      }
    }
    if (!point.generation) initialize();
    writer = std::make_unique<WalWriter>(options,
                                         point.generation ? point.active_name : "operations.wal");
    writer->generation = point.generation;
    if (file_size(writer->file.fd()) > options.maximum_wal_bytes) invalid("WAL exceeds size limit");
    Bytes header(point.generation ? kSegmentHeader : kHeader);
    read_all(writer->file.fd(), header, 0);
    verify_checksum(header);
    if (!std::equal(kWalMagic.begin(), kWalMagic.end(), header.begin()) ||
        get(header, 8, 4) != (point.policy.current ? 3 : (point.generation ? 2 : 1)) ||
        get(header, 12, 4) != header.size() || get(header, 16) != options.dimension ||
        get(header, 24) !=
            48 + kMaximumOperationId + (point.policy.current ? 8 : 0) + options.dimension * 8)
      invalid("WAL header configuration");
    if (point.generation &&
        (get(header, 32) != resource_add(point.lsn, 1) || get(header, 40) != point.generation ||
         get(header, 48) != point.lsn || get(header, 56) != segment_nonce(point.active_name) ||
         hash(header) != point.active_header_digest))
      invalid("WAL identity or start LSN");
    WalReader reader(writer->file.fd(), options.dimension, point.policy.current);
    reader.digest = hash(header);
    bool found_checkpoint = point.lsn == 0 && point.wal_offset == kHeader &&
                            (!point.present || point.wal_digest == reader.digest);
    WalRecord record{};
    std::unordered_set<VectorId> seen_ids;
    if (point.dedup_size) {
      if (!point.generation && point.wal_offset > file_size(writer->file.fd()))
        invalid("WAL shorter than checkpoint");
      dedup_file = DedupCheckpoint::load(options, point, index, dedup);
      reader.offset = point.generation ? kSegmentHeader : point.wal_offset;
      reader.next_lsn = point.lsn + 1;
      reader.digest = point.generation ? hash(header) : point.wal_digest;
      found_checkpoint = true;
      stats.applied_lsn = point.lsn;
    }
    for (const auto& [key, entry] : dedup) {
      (void)entry;
      key_bytes += key.size();
    }
    const auto scan_start = reader.offset;
    current_point = point;
    refresh_usage();
    while (reader.next(record)) {
      const auto key = operation_key(record.epoch, record.operation_id);
      if (point.policy.current) {
        if (record.epoch != point.policy.current) invalid("WAL epoch differs from active epoch");
        check_epoch_capacity(key);
      }
      check_wal_shape(options.limits, options.dimension, record.lsn);
      if (record.lsn <= point.lsn) {
        if (!WalIndexAccess::matches(index, record.id, record.normalized))
          invalid("checkpoint does not match WAL prefix");
      } else {
        WalIndexAccess::insert(index, record.id, record.normalized);
        ++stats.recovered_records;
      }
      if (!dedup
               .emplace(key, DedupEntry{record.lsn, record.id, reader.original_offset,
                                        request_digest(original_bits(record.original)), false})
               .second)
        invalid("duplicate operation ID in WAL");
      key_bytes += key.size();
      if (point.policy.current) {
        ++stats.current_epoch_operations;
        stats.current_epoch_bytes += entry_bytes(key, options.dimension);
      }
      if (!seen_ids.insert(record.id).second) invalid("duplicate vector ID in WAL");
      if (record.lsn == point.lsn) {
        if (reader.offset != point.wal_offset || reader.digest != point.wal_digest)
          invalid("checkpoint WAL boundary mismatch");
        found_checkpoint = true;
      }
      stats.applied_lsn = record.lsn;
    }
    if (!found_checkpoint) invalid("checkpoint exceeds complete WAL prefix");
    if (reader.tail) {
      if (ftruncate(writer->file.fd(), static_cast<off_t>(reader.offset)) != 0)
        system_failure("truncate incomplete WAL tail");
      sync(writer->file.fd());
    }
    // A complete but unacknowledged record may only have reached the page cache.
    // Synchronize the accepted prefix before a retry can receive a durable success.
    sync(writer->file.fd(), true);
    writer->offset = reader.offset;
    writer->digest = reader.digest;
    stats.enabled = true;
    stats.checkpoint_lsn = point.lsn;
    stats.discarded_tail_bytes = reader.tail;
    stats.recovery_us = elapsed(start);
    stats.wal_scan_bytes = reader.offset - scan_start;
    stats.dedup_file_bytes = point.dedup_size;
    current_point = std::move(point);
    refresh_usage();
    stats.wal_generation = writer->generation;
    if (writer->generation) cleanup();
    if (options.dedup_retention && !current_point.policy.current) (void)checkpoint(false);
    stats.recovery_us = elapsed(start);
  }
  static int open_lock(const WalOptions& options) {
    if (options.dimension == 0 || options.dimension > kMaximumDimension ||
        options.maximum_wal_bytes < kHeader ||
        options.maximum_wal_bytes > static_cast<std::uint64_t>(std::numeric_limits<off_t>::max()) ||
        options.directory.empty() || !std::filesystem::is_directory(options.directory))
      throw std::invalid_argument("WAL requires an existing directory and bounded configuration");
    if (((options.rotate_on_checkpoint || options.dedup_retention) &&
         options.maximum_wal_bytes < kSegmentHeader) ||
        options.maximum_checkpoint_bytes == 0)
      throw std::invalid_argument("invalid rotation limits");
    check_wal_shape(options.limits, options.dimension, 0);
    if (options.rotate_on_checkpoint || options.dedup_retention)
      verify_private_directory(options.directory);
    if (options.dedup_retention)
      validate_policy({1, options.retained_epochs == 1 ? 1U : 0U, options.retained_epochs,
                       options.maximum_epoch_operations, options.maximum_epoch_bytes});
    return ::open((options.directory / "writer.lock").c_str(),
                  O_CREAT | O_RDWR | O_CLOEXEC | O_NOFOLLOW, 0600);
  }
  void initialize() {
    const auto path = options.directory / "operations.wal";
    const int fd = ::open(path.c_str(), O_CREAT | O_EXCL | O_WRONLY | O_CLOEXEC | O_NOFOLLOW, 0600);
    if (fd < 0) {
      if (errno == EEXIST) return;
      system_failure("create WAL");
    }
    File file(fd);
    Bytes bytes(kWalMagic.begin(), kWalMagic.end());
    put(bytes, 1, 4);
    put(bytes, kHeader, 4);
    put(bytes, options.dimension);
    put(bytes, 48 + kMaximumOperationId + options.dimension * 8);
    checksum(bytes);
    write_all(file.fd(), bytes);
    sync(file.fd());
    sync_directory(options.directory);
  }
  CommitResult commit(const std::string& operation_id, VectorId id, std::span<const float> vector,
                      const ResourceLimits* admission, std::uint64_t epoch) {
    if (stats.recovery_required) throw RecoveryRequiredError();
    const auto start = Clock::now();
    if (operation_id.empty() || operation_id.size() > kMaximumOperationId ||
        vector.size() != options.dimension)
      throw std::invalid_argument("operation ID or dimension invalid");
    const auto& policy = current_point.policy;
    if ((!policy.current && epoch != 0) || epoch > policy.current)
      throw DedupEpochError("future or unsupported dedup epoch");
    if (policy.current && epoch < policy.oldest) throw DedupEpochError("dedup epoch expired");
    const auto key = operation_key(epoch, operation_id);
    const auto existing = dedup.find(key);
    if (existing != dedup.end()) {
      const auto& old = existing->second;
      auto raw = original_bits(vector);
      if (old.id != id || old.digest != request_digest(raw))
        throw std::invalid_argument("operation ID reused with different content");
      Bytes stored;
      try {
        stored = read_request(old, writer->file.fd(), dedup_file ? dedup_file->fd() : -1,
                              options.dimension);
      } catch (const std::bad_alloc&) {
        throw;
      } catch (...) {
        stats.recovery_required = true;
        throw;
      }
      if (raw != stored) throw std::invalid_argument("operation ID reused with different content");
      return {old.lsn, static_cast<std::size_t>(old.lsn), true};
    }
    if (policy.current && epoch != policy.current)
      throw DedupEpochError("closed dedup epoch has no matching operation");
    check_epoch_capacity(key);
    if (WalIndexAccess::contains(index, id))
      throw std::invalid_argument("vector ID already exists");
    check_wal_shape(options.limits, options.dimension, resource_add(index.size(), 1));
    if (admission) check_wal_shape(*admission, options.dimension, resource_add(index.size(), 1));
    if (stats.applied_lsn == std::numeric_limits<std::uint64_t>::max())
      throw std::length_error("LSN exhausted");
    WalRecord record{stats.applied_lsn + 1, id,
                     operation_id,          std::vector<float>(vector.begin(), vector.end()),
                     normalize_l2(vector),  epoch};
    auto bytes = encode(record);
    if (bytes.size() > options.maximum_wal_bytes - writer->offset) {
      if (!(options.rotate_on_checkpoint || writer->generation))
        throw std::length_error("WAL size limit reached");
      if (options.maximum_wal_bytes < kSegmentHeader ||
          bytes.size() > options.maximum_wal_bytes - kSegmentHeader)
        throw std::length_error("record exceeds WAL segment limit");
      (void)checkpoint(false);
    }
    // Allocate retry metadata before any append. All errors after this point fail closed.
    dedup.emplace(
        key, DedupEntry{record.lsn, id, writer->offset + 40 + operation_id.size() + (epoch ? 8 : 0),
                        request_digest(original_bits(record.original)), false});
    key_bytes += key.size();
    if (policy.current) {
      ++stats.current_epoch_operations;
      stats.current_epoch_bytes += entry_bytes(key, options.dimension);
    }
    try {
      writer->append(bytes);
      stats.last_sync_us = writer->sync_us;
      writer->hit(WalBoundary::BeforeApply);
      WalIndexAccess::insert(index, id, record.normalized);
      stats.applied_lsn = record.lsn;
      writer->hit(WalBoundary::AfterApply);
      stats.last_write_us = elapsed(start);
      writer->hit(WalBoundary::BeforeResponse);
      return {record.lsn, index.size(), false};
    } catch (...) {
      stats.recovery_required = true;
      throw;
    }
  }
  std::uint64_t checkpoint(bool advance = true) {
    if (stats.recovery_required) throw RecoveryRequiredError();
    const auto start = Clock::now();
    bool renamed = false;
    DedupTable next;
    std::unique_ptr<File> next_file;
    std::uint64_t next_size = 0;
    std::unique_ptr<WalWriter> next_writer;
    RecoveryPoint next_point;
    auto policy = current_point.policy;
    if (!policy.current && options.dedup_retention)
      policy = {1, options.retained_epochs == 1 ? 1U : 0U, options.retained_epochs,
                options.maximum_epoch_operations, options.maximum_epoch_bytes};
    else if (policy.current && advance) {
      policy.current = resource_add(policy.current, 1);
      policy.oldest = policy.current >= policy.retained ? policy.current - policy.retained + 1 : 0;
    }
    try {
      CheckpointManager::publish(options, index, stats.applied_lsn, *writer, renamed, dedup,
                                 dedup_file ? dedup_file->fd() : -1, next, next_file, next_size,
                                 next_writer, next_point, policy);
      if (next_writer) writer->hit(WalBoundary::BeforeWriterSwitch);
    } catch (...) {
      if (renamed) stats.recovery_required = true;
      throw;
    }
    dedup.swap(next);
    dedup_file.swap(next_file);
    if (next_writer) writer.swap(next_writer);
    current_point = std::move(next_point);
    refresh_usage();
    stats.wal_generation = writer->generation;
    stats.dedup_file_bytes = next_size;
    stats.checkpoint_lsn = stats.applied_lsn;
    try {
      if (writer->generation) writer->hit(WalBoundary::AfterWriterSwitch);
    } catch (...) {
      stats.recovery_required = true;
      throw;
    }
    next_writer.reset();
    next_file.reset();
    if (writer->generation) cleanup();
    stats.last_checkpoint_us = elapsed(start);
    return stats.checkpoint_lsn;
  }
  void refresh_usage() noexcept {
    const auto& policy = current_point.policy;
    stats.current_epoch = policy.current;
    stats.oldest_epoch = policy.oldest;
    stats.retained_epochs = policy.retained;
    stats.maximum_epoch_operations = policy.operations;
    stats.maximum_epoch_bytes = policy.bytes;
    stats.current_epoch_operations = 0;
    stats.current_epoch_bytes = 0;
    key_bytes = 0;
    for (const auto& [key, entry] : dedup) {
      (void)entry;
      key_bytes += key.size();
      if (policy.current && key_epoch(key) == policy.current) {
        ++stats.current_epoch_operations;
        stats.current_epoch_bytes += 80 + key.size() + options.dimension * 4;
      }
    }
  }
  void check_epoch_capacity(const std::string& key) const {
    const auto& policy = current_point.policy;
    if (policy.current && (stats.current_epoch_operations >= policy.operations ||
                           resource_add(stats.current_epoch_bytes,
                                        entry_bytes(key, options.dimension)) > policy.bytes))
      throw ResourceExhausted();
  }
  void cleanup() noexcept {
    stats.cleanup_pending = true;
    try {
      writer->hit(WalBoundary::BeforeWalCleanup);
      verify_private_directory(options.directory);
      const auto keep_index = checkpoint_path(options, current_point.lsn).filename().string();
      const auto keep_dedup =
          dedup_path(options, current_point.lsn, current_point.generation).filename().string();
      static const std::regex owned(
          "^(operations\\.wal|wal-[0-9]+-[A-Za-z0-9]{6}|checkpoint-[0-9]+\\.fv|dedup-[0-9]+(-[0-9]+"
          ")?\\.fvd|\\.recovery-[A-Za-z0-9]{6})$");
      for (const auto& entry : std::filesystem::directory_iterator(options.directory)) {
        const auto name = entry.path().filename().string();
        if (!std::regex_match(name, owned) || name == writer->name || name == keep_index ||
            name == keep_dedup)
          continue;
        if (!std::filesystem::is_regular_file(entry.symlink_status()))
          throw std::runtime_error("unsafe cleanup entry");
        if (::unlink(entry.path().c_str()) != 0)
          system_failure("remove obsolete recovery artifact");
        ++stats.cleaned_files;
        writer->hit(WalBoundary::AfterWalCleanup);
      }
      sync_directory(options.directory);
      stats.cleanup_pending = false;
    } catch (...) { /* The durable current root remains usable; cleanup is retryable. */
    }
  }
  WalOptions options;
  FlatIndex index;
  File lock_file;
  std::unique_ptr<WalWriter> writer;
  DedupTable dedup;
  std::unique_ptr<File> dedup_file;
  std::uint64_t key_bytes = 0;
  DurabilityStats stats;
  RecoveryPoint current_point;
};
#else
class DurableFlatIndex::Impl {
 public:
  explicit Impl(WalOptions options) : index(options.dimension) {
    throw std::runtime_error("WAL durability is supported only on Linux");
  }
  CommitResult commit(const std::string&, VectorId, std::span<const float>, const ResourceLimits*,
                      std::uint64_t) {
    throw RecoveryRequiredError();
  }
  std::uint64_t checkpoint() { throw RecoveryRequiredError(); }
  FlatIndex index;
  DurabilityStats stats;
};
#endif

DurableFlatIndex::DurableFlatIndex(WalOptions options)
    : impl_(std::make_unique<Impl>(std::move(options))) {}
DurableFlatIndex::~DurableFlatIndex() = default;
void DurableFlatIndex::add(VectorId, std::span<const float>) {
  throw std::invalid_argument("durable insertion requires a client operation ID");
}
CommitResult DurableFlatIndex::commit(const std::string& operation_id, VectorId id,
                                      std::span<const float> vector,
                                      const ResourceLimits* admission, std::uint64_t epoch) {
  return impl_->commit(operation_id, id, vector, admission, epoch);
}
std::uint64_t DurableFlatIndex::checkpoint() { return impl_->checkpoint(); }
DurabilityStats DurableFlatIndex::durability_stats() const noexcept {
  auto result = impl_->stats;
#if defined(__linux__)
  result.dedup_entries = impl_->dedup.size();
  result.active_wal_bytes = impl_->writer->offset;
  result.dedup_metadata_bytes = impl_->dedup.size() * sizeof(DedupEntry);
  result.dedup_metadata_bytes += impl_->key_bytes;
#endif
  return result;
}
std::size_t DurableFlatIndex::size() const noexcept { return impl_->index.size(); }
std::size_t DurableFlatIndex::dimension() const noexcept { return impl_->index.dimension(); }
std::vector<SearchResult> DurableFlatIndex::search(std::span<const float> query,
                                                   std::size_t k) const {
  return impl_->index.search(query, k);
}
std::vector<SearchResult> DurableFlatIndex::search(std::span<const float> query, std::size_t k,
                                                   const SearchContext& context) const {
  return impl_->index.search(query, k, context);
}
}  // namespace fast_vector
