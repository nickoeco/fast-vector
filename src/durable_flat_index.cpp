#include "fast_vector/durable_flat_index.h"

#include <algorithm>
#include <array>
#include <bit>
#include <chrono>
#include <cmath>
#include <cstring>
#include <fstream>
#include <limits>
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
    std::error_code ec;
    std::filesystem::remove(path_, ec);
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
};
struct DedupEntry {
  std::uint64_t lsn;
  VectorId id;
  std::vector<float> original;
};
Bytes encode(const WalRecord& record) {
  Bytes bytes;
  bytes.reserve(48 + record.operation_id.size() + record.original.size() * 8);
  put(bytes, 48 + record.operation_id.size() + record.original.size() * 8);
  put(bytes, record.lsn);
  put(bytes, 1, 4);
  put(bytes, record.operation_id.size(), 4);
  put(bytes, record.id);
  put(bytes, record.original.size());
  bytes.insert(bytes.end(), record.operation_id.begin(), record.operation_id.end());
  for (const auto& values : {std::span(record.original), std::span(record.normalized)})
    for (float value : values) put(bytes, std::bit_cast<std::uint32_t>(value), 4);
  checksum(bytes);
  return bytes;
}

/** Validates fixed fields before allocating a record-sized buffer. */
class WalReader {
 public:
  WalReader(int fd, std::size_t dimension) : fd_(fd), dimension_(dimension), size_(file_size(fd)) {}
  bool next(WalRecord& record) {
    if (offset == size_) return false;
    const auto remaining = size_ - offset;
    if (remaining < 8) {
      tail = remaining;
      return false;
    }
    std::array<unsigned char, 40> prefix{};
    read_all(fd_, std::span(prefix).first(std::min<std::uint64_t>(remaining, prefix.size())),
             offset);
    const auto length = get(prefix, 0);
    const auto minimum = 49 + dimension_ * 8;
    const auto maximum = 48 + kMaximumOperationId + dimension_ * 8;
    if (length < minimum || length > maximum) invalid("record length outside configured bounds");
    // Validate every available fixed field even when the trailing payload is incomplete.
    if (remaining >= 16 && get(prefix, 8) != next_lsn) invalid("non-contiguous LSN");
    if (remaining >= 20 && get(prefix, 16, 4) != 1) invalid("unsupported operation");
    if (remaining >= 24) {
      const auto op_length = get(prefix, 20, 4);
      if (op_length == 0 || op_length > kMaximumOperationId ||
          length != 48 + op_length + dimension_ * 8)
        invalid("invalid operation ID or record length");
    }
    if (remaining >= 40 && get(prefix, 32) != dimension_) invalid("dimension mismatch");
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
    record.operation_id.assign(reinterpret_cast<const char*>(bytes.data() + 40), op_length);
    record.original.resize(dimension_);
    record.normalized.resize(dimension_);
    auto cursor = 40 + op_length;
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

 private:
  int fd_;
  std::size_t dimension_;
  std::uint64_t size_;
};

class WalWriter {
 public:
  explicit WalWriter(const WalOptions& options)
      : file(::open((options.directory / "operations.wal").c_str(),
                    O_RDWR | O_APPEND | O_CLOEXEC | O_NOFOLLOW)),
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
  std::uint64_t offset = kHeader, digest = kOffset, sync_us = 0;

 private:
  const WalOptions& options_;
};

struct RecoveryPoint {
  std::uint64_t lsn = 0, size = 0, digest = 0, wal_offset = kHeader, wal_digest = 0;
  bool present = false;
};
std::filesystem::path checkpoint_path(const WalOptions& options, std::uint64_t lsn) {
  return options.directory / ("checkpoint-" + std::to_string(lsn) + ".fv");
}

class RecoveryManager {
 public:
  static RecoveryPoint checkpoint(const WalOptions& options, FlatIndex& index) {
    const auto path = options.directory / "recovery.manifest";
    if (!std::filesystem::exists(path)) return {};
    File file(::open(path.c_str(), O_RDONLY | O_CLOEXEC | O_NOFOLLOW));
    if (file_size(file.fd()) != kManifest) invalid("recovery manifest length");
    std::array<unsigned char, kManifest> bytes{};
    read_all(file.fd(), bytes, 0);
    verify_checksum(bytes);
    if (!std::equal(kCheckpointMagic.begin(), kCheckpointMagic.end(), bytes.begin()) ||
        get(bytes, 8, 4) != 1 || get(bytes, 12, 4) != kManifest ||
        get(bytes, 16) != options.dimension)
      invalid("recovery manifest configuration");
    RecoveryPoint point{get(bytes, 24), get(bytes, 32), get(bytes, 40),
                        get(bytes, 48), get(bytes, 56), true};
    if (point.wal_offset < kHeader || point.wal_offset > options.maximum_wal_bytes ||
        point.lsn > (point.wal_offset - kHeader) / (49 + options.dimension * 8) ||
        point.size > std::max<std::uint64_t>(56, options.maximum_wal_bytes))
      invalid("recovery manifest bounds");
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
                      WalWriter& writer, bool& manifest_renamed) {
    Temporary temporary(options.directory);
    save_flat_index(index, temporary.path());
    const auto verified = load_flat_index(temporary.path(), options.kernel);
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
    Bytes bytes(kCheckpointMagic.begin(), kCheckpointMagic.end());
    put(bytes, 1, 4);
    put(bytes, kManifest, 4);
    put(bytes, options.dimension);
    put(bytes, lsn);
    put(bytes, size);
    put(bytes, digest);
    put(bytes, writer.offset);
    put(bytes, writer.digest);
    checksum(bytes);
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
    initialize();
    writer = std::make_unique<WalWriter>(options);
    if (file_size(writer->file.fd()) > options.maximum_wal_bytes) invalid("WAL exceeds size limit");
    std::array<unsigned char, kHeader> header{};
    read_all(writer->file.fd(), header, 0);
    verify_checksum(header);
    if (!std::equal(kWalMagic.begin(), kWalMagic.end(), header.begin()) || get(header, 8, 4) != 1 ||
        get(header, 12, 4) != kHeader || get(header, 16) != options.dimension ||
        get(header, 24) != 48 + kMaximumOperationId + options.dimension * 8)
      invalid("WAL header configuration");
    const auto point = RecoveryManager::checkpoint(options, index);
    WalReader reader(writer->file.fd(), options.dimension);
    reader.digest = hash(header);
    bool found_checkpoint = point.lsn == 0 && point.wal_offset == kHeader &&
                            (!point.present || point.wal_digest == reader.digest);
    WalRecord record{};
    std::unordered_set<VectorId> seen_ids;
    while (reader.next(record)) {
      check_wal_shape(options.limits, options.dimension, record.lsn);
      if (record.lsn <= point.lsn) {
        if (!WalIndexAccess::matches(index, record.id, record.normalized))
          invalid("checkpoint does not match WAL prefix");
      } else {
        WalIndexAccess::insert(index, record.id, record.normalized);
        ++stats.recovered_records;
      }
      if (!dedup
               .emplace(record.operation_id,
                        DedupEntry{record.lsn, record.id, std::move(record.original)})
               .second)
        invalid("duplicate operation ID in WAL");
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
  }
  static int open_lock(const WalOptions& options) {
    if (options.dimension == 0 || options.dimension > kMaximumDimension ||
        options.maximum_wal_bytes < kHeader ||
        options.maximum_wal_bytes > static_cast<std::uint64_t>(std::numeric_limits<off_t>::max()) ||
        options.directory.empty() || !std::filesystem::is_directory(options.directory))
      throw std::invalid_argument("WAL requires an existing directory and bounded configuration");
    check_wal_shape(options.limits, options.dimension, 0);
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
                      const ResourceLimits* admission) {
    if (stats.recovery_required) throw RecoveryRequiredError();
    const auto start = Clock::now();
    if (operation_id.empty() || operation_id.size() > kMaximumOperationId ||
        vector.size() != options.dimension)
      throw std::invalid_argument("operation ID or dimension invalid");
    const auto existing = dedup.find(operation_id);
    if (existing != dedup.end()) {
      const auto& old = existing->second;
      if (old.id != id || old.original.size() != vector.size() ||
          std::memcmp(old.original.data(), vector.data(), vector.size_bytes()) != 0)
        throw std::invalid_argument("operation ID reused with different content");
      return {old.lsn, static_cast<std::size_t>(old.lsn), true};
    }
    if (WalIndexAccess::contains(index, id))
      throw std::invalid_argument("vector ID already exists");
    check_wal_shape(options.limits, options.dimension, resource_add(index.size(), 1));
    if (admission) check_wal_shape(*admission, options.dimension, resource_add(index.size(), 1));
    if (stats.applied_lsn == std::numeric_limits<std::uint64_t>::max())
      throw std::length_error("LSN exhausted");
    WalRecord record{stats.applied_lsn + 1, id, operation_id,
                     std::vector<float>(vector.begin(), vector.end()), normalize_l2(vector)};
    auto bytes = encode(record);
    if (bytes.size() > options.maximum_wal_bytes - writer->offset)
      throw std::length_error("WAL size limit reached");
    // Allocate retry metadata before any append. All errors after this point fail closed.
    dedup.emplace(operation_id, DedupEntry{record.lsn, id, std::move(record.original)});
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
  std::uint64_t checkpoint() {
    if (stats.recovery_required) throw RecoveryRequiredError();
    const auto start = Clock::now();
    bool renamed = false;
    try {
      CheckpointManager::publish(options, index, stats.applied_lsn, *writer, renamed);
    } catch (...) {
      if (renamed) stats.recovery_required = true;
      throw;
    }
    stats.checkpoint_lsn = stats.applied_lsn;
    stats.last_checkpoint_us = elapsed(start);
    return stats.checkpoint_lsn;
  }
  WalOptions options;
  FlatIndex index;
  File lock_file;
  std::unique_ptr<WalWriter> writer;
  std::unordered_map<std::string, DedupEntry> dedup;
  DurabilityStats stats;
};
#else
class DurableFlatIndex::Impl {
 public:
  explicit Impl(WalOptions options) : index(options.dimension) {
    throw std::runtime_error("WAL durability is supported only on Linux");
  }
  CommitResult commit(const std::string&, VectorId, std::span<const float>, const ResourceLimits*) {
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
                                      const ResourceLimits* admission) {
  return impl_->commit(operation_id, id, vector, admission);
}
std::uint64_t DurableFlatIndex::checkpoint() { return impl_->checkpoint(); }
DurabilityStats DurableFlatIndex::durability_stats() const noexcept { return impl_->stats; }
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
