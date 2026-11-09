#include <gtest/gtest.h>
#include <sys/wait.h>
#include <unistd.h>

#include <array>
#include <bit>
#include <cmath>
#include <csignal>
#include <filesystem>
#include <fstream>
#include <iterator>
#include <limits>
#include <string>
#include <vector>

#include "fast_vector/durable_flat_index.h"
#include "fast_vector/flat_index_io.h"
#include "fast_vector/vector_store.h"

namespace {
using namespace fast_vector;
using Bytes = std::vector<unsigned char>;
const std::vector<float> kVector{3.1F, 7.2F, -2.3F};
class Directory {
 public:
  Directory() {
    std::array<char, 64> name{};
    const std::string pattern = "/tmp/fast-vector-wal-XXXXXX";
    std::copy(pattern.begin(), pattern.end(), name.begin());
    const auto* result = mkdtemp(name.data());
    if (!result) throw std::runtime_error("mkdtemp failed");
    path = result;
  }
  ~Directory() {
    std::error_code ec;
    std::filesystem::remove_all(path, ec);
  }
  WalOptions options() const { return {.directory = path, .dimension = 3, .fault_hook = {}}; }
  std::filesystem::path path;
};
Bytes read(const std::filesystem::path& path) {
  std::ifstream input(path, std::ios::binary);
  return {std::istreambuf_iterator<char>(input), {}};
}
void write(const std::filesystem::path& path, const Bytes& bytes) {
  std::ofstream output(path, std::ios::binary | std::ios::trunc);
  output.write(reinterpret_cast<const char*>(bytes.data()),
               static_cast<std::streamsize>(bytes.size()));
}
void put(Bytes& bytes, std::size_t offset, std::uint64_t value, std::size_t width = 8) {
  for (std::size_t i = 0; i < width; ++i) {
    bytes.at(offset + i) = value & 255U;
    value >>= 8;
  }
}
std::uint64_t hash(const Bytes& bytes, std::size_t begin, std::size_t end) {
  std::uint64_t value = 14695981039346656037ULL;
  for (auto i = begin; i < end; ++i) {
    value ^= bytes[i];
    value *= 1099511628211ULL;
  }
  return value;
}
void checksum(Bytes& bytes, std::size_t begin = 0) {
  put(bytes, bytes.size() - 8, hash(bytes, begin, bytes.size() - 8));
}

TEST(DurableFlatIndexTest, RecoversExactBitsAndPersistentRetryResultWithoutCheckpoint) {
  Directory directory;
  Bytes before;
  {
    DurableFlatIndex index(directory.options());
    EXPECT_THROW(index.add(1, kVector), std::invalid_argument);
    const auto first = index.commit("first", 91, kVector);
    EXPECT_EQ(first.lsn, 1);
    EXPECT_EQ(first.index_size, 1);
    EXPECT_FALSE(first.replayed);
    (void)index.commit("second", 19, std::vector<float>{0, 1, 0});
    EXPECT_THROW((void)index.commit("first", 91, std::vector<float>{3.1F, 7.2F, 2.3F}),
                 std::invalid_argument);
    EXPECT_THROW((void)index.commit("first", 92, kVector), std::invalid_argument);
    EXPECT_THROW((void)index.commit("other", 91, kVector), std::invalid_argument);
    EXPECT_THROW((void)index.commit("", 92, kVector), std::invalid_argument);
    EXPECT_THROW((void)index.commit("zero", 92, std::vector<float>{0, 0, 0}),
                 std::invalid_argument);
    EXPECT_THROW((void)index.commit("nan", 92, std::vector<float>{NAN, 0, 0}),
                 std::invalid_argument);
    EXPECT_THROW((void)index.commit("dimension", 92, std::vector<float>{1}), std::invalid_argument);
    EXPECT_EQ(index.checkpoint(), 2);
    before = read(directory.path / "checkpoint-2.fv");
    // Delete the recovery point to exercise WAL-only reconstruction of the same float bits.
    std::filesystem::remove(directory.path / "recovery.manifest");
  }
  DurableFlatIndex recovered(directory.options());
  EXPECT_EQ(recovered.size(), 2);
  EXPECT_EQ(recovered.durability_stats().recovered_records, 2);
  const auto retry = recovered.commit("first", 91, kVector);
  EXPECT_TRUE(retry.replayed);
  EXPECT_EQ(retry.lsn, 1);
  EXPECT_EQ(retry.index_size, 1);
  EXPECT_EQ(recovered.checkpoint(), 2);
  EXPECT_EQ(read(directory.path / "checkpoint-2.fv"), before);
  const auto results = recovered.search(kVector, 2);
  EXPECT_EQ(results.at(0).id, 91);
  EXPECT_NEAR(results[0].score, 1, 1e-6);
}

TEST(DurableFlatIndexTest, CheckpointReplaysSuffixAndRestoresHistoricalIdempotency) {
  Directory directory;
  {
    DurableFlatIndex index(directory.options());
    (void)index.commit("prefix", 10, kVector);
    EXPECT_EQ(index.checkpoint(), 1);
    (void)index.commit("suffix", 20, std::vector<float>{1, 0, 0});
  }
  DurableFlatIndex recovered(directory.options());
  const auto stats = recovered.durability_stats();
  EXPECT_EQ(stats.checkpoint_lsn, 1);
  EXPECT_EQ(stats.applied_lsn, 2);
  EXPECT_EQ(stats.recovered_records, 1);
  EXPECT_EQ(recovered.size(), 2);
  EXPECT_TRUE(recovered.commit("prefix", 10, kVector).replayed);
  EXPECT_TRUE(recovered.commit("suffix", 20, std::vector<float>{1, 0, 0}).replayed);
  EXPECT_THROW((void)recovered.commit("prefix", 10, std::vector<float>{1, 0, 0}),
               std::invalid_argument);
}

TEST(DurableFlatIndexTest, IncompleteLastRecordIsTruncatedAndCanBeReplaced) {
  Directory directory;
  Bytes complete;
  std::size_t prefix = 0;
  {
    DurableFlatIndex index(directory.options());
    (void)index.commit("one", 1, kVector);
    prefix = read(directory.path / "operations.wal").size();
    (void)index.commit("two", 2, kVector);
    complete = read(directory.path / "operations.wal");
  }
  for (std::size_t cut = prefix; cut < complete.size(); ++cut) {
    SCOPED_TRACE(cut);
    write(directory.path / "operations.wal", Bytes(complete.begin(), complete.begin() + cut));
    DurableFlatIndex index(directory.options());
    EXPECT_EQ(index.size(), 1);
    EXPECT_EQ(index.durability_stats().discarded_tail_bytes, cut - prefix);
    EXPECT_EQ(std::filesystem::file_size(directory.path / "operations.wal"), prefix);
    EXPECT_EQ(index.commit("replacement", 3, kVector).lsn, 2);
  }
}

TEST(DurableFlatIndexTest, RejectsCompleteCorruptionIllegalFieldsAndRepeatedLsn) {
  Directory directory;
  Bytes good;
  std::size_t second = 0;
  {
    DurableFlatIndex index(directory.options());
    (void)index.commit("one", 1, kVector);
    second = read(directory.path / "operations.wal").size();
    (void)index.commit("two", 2, kVector);
    good = read(directory.path / "operations.wal");
  }
  const auto path = directory.path / "operations.wal";
  auto corrupt = good;
  corrupt.back() ^= 1;
  write(path, corrupt);
  EXPECT_THROW(DurableFlatIndex{directory.options()}, std::runtime_error);
  for (const auto [offset, value, width] : std::vector<std::array<std::uint64_t, 3>>{
           {second, 0, 8},
           {second, std::numeric_limits<std::uint64_t>::max(), 8},
           {second + 8, 1, 8},
           {second + 16, 2, 4},
           {second + 20, 257, 4},
           {second + 32, 4, 8},
           {second + 24, 1, 8},
           {second + 43, std::bit_cast<std::uint32_t>(INFINITY), 4},
           {second + 55, std::bit_cast<std::uint32_t>(0.F), 4}}) {
    SCOPED_TRACE(offset);
    corrupt = good;
    put(corrupt, offset, value, width);
    checksum(corrupt, second);
    write(path, corrupt);
    EXPECT_THROW(DurableFlatIndex{directory.options()}, std::exception);
  }
  corrupt = good;
  corrupt[0] ^= 1;
  checksum(corrupt);
  write(path, corrupt);
  EXPECT_THROW(DurableFlatIndex{directory.options()}, std::runtime_error);
  write(path, good);
  auto wrong = directory.options();
  wrong.dimension = 4;
  EXPECT_THROW(DurableFlatIndex{wrong}, std::runtime_error);
}

TEST(DurableFlatIndexTest, CorruptManifestCheckpointAndDedupAreRejected) {
  Directory directory;
  {
    DurableFlatIndex index(directory.options());
    (void)index.commit("one", 1, kVector);
    (void)index.checkpoint();
  }
  const auto manifest = directory.path / "recovery.manifest";
  const auto good_manifest = read(manifest);
  auto bad = good_manifest;
  bad.back() ^= 1;
  write(manifest, bad);
  EXPECT_THROW(DurableFlatIndex{directory.options()}, std::runtime_error);
  bad = good_manifest;
  put(bad, 24, 2);
  checksum(bad);
  write(manifest, bad);
  EXPECT_THROW(DurableFlatIndex{directory.options()}, std::exception);
  write(manifest, good_manifest);
  const auto checkpoint = directory.path / "checkpoint-1.fv";
  const auto good_checkpoint = read(checkpoint);
  bad = good_checkpoint;
  bad.back() ^= 1;
  write(checkpoint, bad);
  EXPECT_THROW(DurableFlatIndex{directory.options()}, std::runtime_error);
  write(checkpoint, good_checkpoint);
  const auto wal = directory.path / "dedup-1.fvd";
  bad = read(wal);
  bad.back() ^= 1;
  write(wal, bad);
  EXPECT_THROW(DurableFlatIndex{directory.options()}, std::runtime_error);
}

TEST(DurableFlatIndexTest, AnyFailureAfterAppendAdmissionStopsWritesUntilRecovery) {
  for (auto boundary :
       {WalBoundary::BeforeAppend, WalBoundary::AfterPartialAppend, WalBoundary::AfterAppend,
        WalBoundary::BeforeSync, WalBoundary::AfterSync, WalBoundary::BeforeApply,
        WalBoundary::AfterApply, WalBoundary::BeforeResponse}) {
    Directory directory;
    auto options = directory.options();
    options.fault_hook = [boundary](WalBoundary actual) {
      if (actual == boundary) throw std::bad_alloc();
    };
    {
      DurableFlatIndex index(options);
      EXPECT_THROW((void)index.commit("one", 1, kVector), std::bad_alloc);
      EXPECT_TRUE(index.durability_stats().recovery_required);
      EXPECT_THROW((void)index.commit("two", 2, kVector), RecoveryRequiredError);
      EXPECT_THROW((void)index.checkpoint(), RecoveryRequiredError);
      if (boundary == WalBoundary::BeforeResponse || boundary == WalBoundary::AfterApply)
        EXPECT_EQ(index.search(kVector, 1).at(0).id, 1);
      else
        EXPECT_TRUE(index.search(kVector, 1).empty());
    }
    DurableFlatIndex recovered(directory.options());
    const auto exists =
        boundary != WalBoundary::BeforeAppend && boundary != WalBoundary::AfterPartialAppend;
    EXPECT_EQ(recovered.size(), exists ? 1 : 0);
    const auto retry = recovered.commit("one", 1, kVector);
    EXPECT_EQ(retry.lsn, 1);
    EXPECT_EQ(retry.replayed, exists);
  }
}

TEST(DurableFlatIndexTest, CheckpointFailureRetainsOldChainAndAfterRenameFailsClosed) {
  Directory directory;
  bool fail = false;
  auto boundary = WalBoundary::BeforeCheckpointPublish;
  auto options = directory.options();
  options.fault_hook = [&](WalBoundary actual) {
    if (fail && actual == boundary) throw std::runtime_error("injected publication failure");
  };
  {
    DurableFlatIndex index(options);
    (void)index.commit("one", 1, kVector);
    (void)index.checkpoint();
    const auto old = read(directory.path / "recovery.manifest");
    (void)index.commit("two", 2, kVector);
    fail = true;
    EXPECT_THROW((void)index.checkpoint(), std::runtime_error);
    EXPECT_EQ(read(directory.path / "recovery.manifest"), old);
    EXPECT_FALSE(index.durability_stats().recovery_required);
    (void)index.commit("three", 3, kVector);
  }
  {
    DurableFlatIndex recovered(directory.options());
    EXPECT_EQ(recovered.size(), 3);
    EXPECT_EQ(recovered.durability_stats().checkpoint_lsn, 1);
  }
  boundary = WalBoundary::AfterCheckpointRename;
  {
    DurableFlatIndex recovered(options);
    EXPECT_THROW((void)recovered.checkpoint(), std::runtime_error);
    EXPECT_TRUE(recovered.durability_stats().recovery_required);
    EXPECT_THROW((void)recovered.commit("four", 4, kVector), RecoveryRequiredError);
  }
  DurableFlatIndex recovered(directory.options());
  EXPECT_EQ(recovered.size(), 3);
  EXPECT_EQ(recovered.durability_stats().checkpoint_lsn, 3);
}

TEST(DurableFlatIndexTest, LocksDirectoryAndRejectsConfigurationBeforeLargeAllocations) {
  Directory directory;
  DurableFlatIndex writer(directory.options());
  EXPECT_THROW(DurableFlatIndex{directory.options()}, std::system_error);
  auto options = directory.options();
  options.dimension = 1'000'001;
  EXPECT_THROW(DurableFlatIndex{options}, std::invalid_argument);
  options = directory.options();
  options.maximum_wal_bytes = 39;
  EXPECT_THROW(DurableFlatIndex{options}, std::invalid_argument);
}

TEST(DurableFlatIndexTest, SyncIoFailureHasUncertainOutcomeAndBlocksFurtherWrites) {
  Directory directory;
  auto options = directory.options();
  options.fault_hook = [](WalBoundary boundary) {
    if (boundary == WalBoundary::BeforeSync)
      throw std::system_error(std::make_error_code(std::errc::io_error),
                              "injected fdatasync failure");
  };
  {
    DurableFlatIndex index(options);
    EXPECT_THROW((void)index.commit("one", 1, kVector), std::system_error);
    EXPECT_TRUE(index.durability_stats().recovery_required);
    EXPECT_EQ(index.size(), 0);
    EXPECT_THROW((void)index.commit("two", 2, kVector), RecoveryRequiredError);
  }
  DurableFlatIndex recovered(directory.options());
  EXPECT_EQ(recovered.size(), 1);
  EXPECT_TRUE(recovered.commit("one", 1, kVector).replayed);
}

TEST(DurableFlatIndexTest, EmptyCheckpointAndBoundedOperationIdsAreRecoverable) {
  Directory directory;
  {
    auto options = directory.options();
    options.maximum_wal_bytes = 40;
    DurableFlatIndex index(options);
    EXPECT_EQ(index.checkpoint(), 0);
  }
  {
    DurableFlatIndex index(directory.options());
    EXPECT_EQ(index.size(), 0);
    EXPECT_EQ(index.durability_stats().checkpoint_lsn, 0);
    EXPECT_THROW((void)index.commit(std::string(257, 'x'), 1, kVector), std::invalid_argument);
    EXPECT_EQ(index.commit(std::string(256, 'x'), 1, kVector).lsn, 1);
  }
  DurableFlatIndex recovered(directory.options());
  EXPECT_TRUE(recovered.commit(std::string(256, 'x'), 1, kVector).replayed);
}

TEST(DurableFlatIndexTest, DuplicateOperationAndAvailableInvalidLsnInPartialTailAreRejected) {
  Directory directory;
  std::size_t second = 0;
  {
    DurableFlatIndex index(directory.options());
    (void)index.commit("one", 1, kVector);
    second = read(directory.path / "operations.wal").size();
    (void)index.commit("two", 2, kVector);
  }
  auto bytes = read(directory.path / "operations.wal");
  const auto original = bytes;
  std::copy_n("one", 3, bytes.begin() + second + 40);
  checksum(bytes, second);
  write(directory.path / "operations.wal", bytes);
  EXPECT_THROW(DurableFlatIndex{directory.options()}, std::runtime_error);
  bytes = original;
  bytes.resize(second + 16);
  put(bytes, second + 8, 1);
  write(directory.path / "operations.wal", bytes);
  EXPECT_THROW(DurableFlatIndex{directory.options()}, std::runtime_error);
}

TEST(DurableFlatIndexTest, WalLimitDoesNotPoisonWriterAndSignedZeroRetriesAreBitExact) {
  Directory directory;
  auto options = directory.options();
  options.maximum_wal_bytes = 115;
  DurableFlatIndex index(options);
  (void)index.commit("one", 1, std::vector<float>{1, 0, 0});
  EXPECT_THROW((void)index.commit("one", 1, std::vector<float>{1, -0.F, 0}), std::invalid_argument);
  EXPECT_THROW((void)index.commit("two", 2, kVector), std::length_error);
  EXPECT_FALSE(index.durability_stats().recovery_required);
  EXPECT_TRUE(index.commit("one", 1, std::vector<float>{1, 0, 0}).replayed);
}

TEST(DurableFlatIndexTest, VectorStoreSerializesCommitAndRejectsBatchBypass) {
  Directory directory;
  ResourceLimits limits;
  limits.maximum_vectors = 1;
  VectorStore store(std::make_unique<DurableFlatIndex>(directory.options()), 1000, false, {},
                    limits);
  EXPECT_THROW(store.add(1, kVector), std::invalid_argument);
  EXPECT_EQ(store.add(1, kVector, "one").lsn, 1);
  EXPECT_TRUE(store.add(1, kVector, "one").replayed);
  EXPECT_THROW((void)store.add(2, kVector, "two"), ResourceExhausted);
  EXPECT_FALSE(store.durability_stats().recovery_required);
  EXPECT_EQ(store.stats().inserted_vectors, 1);
  const std::vector<VectorRecord> batch{{2, kVector}};
  EXPECT_THROW(store.add_batch(batch), std::logic_error);
  EXPECT_EQ(store.checkpoint(), 1);
  EXPECT_TRUE(store.durability_stats().enabled);
  EXPECT_EQ(store.search(kVector, 10).size(), 1);
}

TEST(DurableFlatIndexTest, SigkillAtCommitBoundariesPreservesAllAcknowledgedOperations) {
  for (auto boundary :
       {WalBoundary::BeforeAppend, WalBoundary::AfterPartialAppend, WalBoundary::AfterAppend,
        WalBoundary::BeforeSync, WalBoundary::AfterSync, WalBoundary::BeforeApply,
        WalBoundary::AfterApply, WalBoundary::BeforeResponse}) {
    Directory directory;
    int pipe_fd[2];
    ASSERT_EQ(pipe(pipe_fd), 0);
    const auto pid = fork();
    ASSERT_GE(pid, 0);
    if (pid == 0) {
      close(pipe_fd[0]);
      bool armed = false;
      auto options = directory.options();
      options.fault_hook = [&](WalBoundary actual) {
        if (armed && actual == boundary) {
          kill(getpid(), SIGKILL);
          _exit(99);
        }
      };
      DurableFlatIndex index(options);
      (void)index.commit("acknowledged", 1, kVector);
      const char ack = 'A';
      if (::write(pipe_fd[1], &ack, 1) != 1) _exit(98);
      armed = true;
      (void)index.commit("uncertain", 2, kVector);
      _exit(97);
    }
    close(pipe_fd[1]);
    char ack = 0;
    EXPECT_EQ(::read(pipe_fd[0], &ack, 1), 1);
    close(pipe_fd[0]);
    int status = 0;
    ASSERT_EQ(waitpid(pid, &status, 0), pid);
    ASSERT_TRUE(WIFSIGNALED(status));
    EXPECT_EQ(WTERMSIG(status), SIGKILL);
    EXPECT_EQ(ack, 'A');
    DurableFlatIndex recovered(directory.options());
    EXPECT_TRUE(recovered.commit("acknowledged", 1, kVector).replayed);
    const auto pending = recovered.commit("uncertain", 2, kVector);
    EXPECT_EQ(pending.lsn, 2);
    EXPECT_EQ(recovered.size(), 2);
  }
}
TEST(DurableFlatIndexTest, V2SkipsOldWalPrefixAndRetriesFromDedupFile) {
  Directory directory;
  std::size_t boundary = 0;
  {
    DurableFlatIndex index(directory.options());
    (void)index.commit("prefix", 1, kVector);
    (void)index.checkpoint();
    boundary = std::filesystem::file_size(directory.path / "operations.wal");
    (void)index.commit("suffix", 2, kVector);
  }
  auto bytes = read(directory.path / "operations.wal");
  bytes[boundary - 1] ^= 1;
  write(directory.path / "operations.wal", bytes);
  DurableFlatIndex index(directory.options());
  EXPECT_EQ(index.durability_stats().wal_scan_bytes, bytes.size() - boundary);
  EXPECT_EQ(index.durability_stats().recovered_records, 1);
  EXPECT_TRUE(index.commit("prefix", 1, kVector).replayed);
  EXPECT_TRUE(index.commit("suffix", 2, kVector).replayed);
  (void)index.checkpoint();
  EXPECT_EQ(index.durability_stats().dedup_entries, 2);
}

TEST(DurableFlatIndexTest, LegacyManifestScansWalAndUpgradesOnCheckpoint) {
  Directory directory;
  {
    DurableFlatIndex index(directory.options());
    (void)index.commit("one", 1, kVector);
    (void)index.checkpoint();
  }
  auto manifest = read(directory.path / "recovery.manifest");
  manifest.resize(72);
  put(manifest, 8, 1, 4);
  put(manifest, 12, 72, 4);
  checksum(manifest);
  write(directory.path / "recovery.manifest", manifest);
  {
    DurableFlatIndex index(directory.options());
    EXPECT_EQ(index.durability_stats().wal_scan_bytes,
              std::filesystem::file_size(directory.path / "operations.wal") - 40);
    EXPECT_TRUE(index.commit("one", 1, kVector).replayed);
    (void)index.checkpoint();
  }
  DurableFlatIndex index(directory.options());
  EXPECT_EQ(index.durability_stats().wal_scan_bytes, 0);
  EXPECT_TRUE(index.commit("one", 1, kVector).replayed);
}

TEST(DurableFlatIndexTest, CorruptDedupStructureIsRejectedEvenWithUpdatedChecksums) {
  Directory directory;
  {
    DurableFlatIndex index(directory.options());
    (void)index.commit("one", 1, kVector);
    (void)index.checkpoint();
  }
  const auto file = directory.path / "dedup-1.fvd";
  const auto good = read(file), root = read(directory.path / "recovery.manifest");
  for (const auto [offset, value, width] :
       std::vector<std::array<std::uint64_t, 3>>{{56, UINT64_MAX, 8},
                                                 {64, 2, 8},
                                                 {72, 999, 8},
                                                 {80, 257, 4},
                                                 {84, 1, 4},
                                                 {88, 4, 8},
                                                 {96, 0, 8}}) {
    auto bytes = good;
    put(bytes, offset, value, width);
    checksum(bytes, 56);
    write(file, bytes);
    auto manifest = root;
    put(manifest, 72, hash(bytes, 0, bytes.size()));
    checksum(manifest);
    write(directory.path / "recovery.manifest", manifest);
    EXPECT_THROW(DurableFlatIndex{directory.options()}, std::exception);
  }
}

TEST(DurableFlatIndexTest, MetadataDoesNotRetainDimensionSizedVectorsAndRetryRemainsBitExact) {
  std::uint64_t small = 0;
  for (const auto dimension : {3U, 4096U}) {
    Directory directory;
    auto options = directory.options();
    options.dimension = dimension;
    const std::vector<float> values(dimension, 1);
    {
      DurableFlatIndex index(options);
      (void)index.commit("one", 1, values);
      const auto metadata = index.durability_stats().dedup_metadata_bytes;
      if (dimension == 3)
        small = metadata;
      else
        EXPECT_EQ(metadata, small);
      (void)index.checkpoint();
      EXPECT_EQ(index.durability_stats().dedup_metadata_bytes, metadata);
    }
    DurableFlatIndex index(options);
    EXPECT_TRUE(index.commit("one", 1, values).replayed);
    auto changed = values;
    changed.back() = 2;
    EXPECT_THROW((void)index.commit("one", 1, changed), std::invalid_argument);
    EXPECT_EQ(index.durability_stats().dedup_metadata_bytes, small);
  }
}

TEST(DurableFlatIndexTest, RetryDetectsPostLoadDiskCorruptionAndFailsClosed) {
  Directory directory;
  DurableFlatIndex index(directory.options());
  (void)index.commit("one", 1, kVector);
  (void)index.checkpoint();
  auto bytes = read(directory.path / "dedup-1.fvd");
  bytes[56 + 72 + 3] ^= 1;
  write(directory.path / "dedup-1.fvd", bytes);
  EXPECT_THROW((void)index.commit("one", 1, kVector), std::runtime_error);
  EXPECT_TRUE(index.durability_stats().recovery_required);
  EXPECT_THROW((void)index.commit("two", 2, kVector), RecoveryRequiredError);
}
TEST(DurableFlatIndexTest, SigkillDuringPairedCheckpointKeepsAcknowledgedWritesRecoverable) {
  for (const auto boundary :
       {WalBoundary::BeforeCheckpointPublish, WalBoundary::AfterCheckpointRename}) {
    Directory directory;
    int descriptors[2];
    ASSERT_EQ(pipe(descriptors), 0);
    const auto pid = fork();
    ASSERT_GE(pid, 0);
    if (pid == 0) {
      close(descriptors[0]);
      bool armed = false;
      auto options = directory.options();
      options.fault_hook = [&](WalBoundary actual) {
        if (armed && boundary == actual) {
          kill(getpid(), SIGKILL);
          _exit(91);
        }
      };
      DurableFlatIndex index(options);
      (void)index.commit("one", 1, kVector);
      (void)index.checkpoint();
      (void)index.commit("two", 2, kVector);
      const char ack = '2';
      if (::write(descriptors[1], &ack, 1) != 1) _exit(92);
      armed = true;
      (void)index.checkpoint();
      _exit(93);
    }
    close(descriptors[1]);
    char ack = 0;
    ASSERT_EQ(::read(descriptors[0], &ack, 1), 1);
    close(descriptors[0]);
    int status = 0;
    ASSERT_EQ(waitpid(pid, &status, 0), pid);
    ASSERT_TRUE(WIFSIGNALED(status));
    ASSERT_EQ(WTERMSIG(status), SIGKILL);
    EXPECT_EQ(ack, '2');
    DurableFlatIndex recovered(directory.options());
    EXPECT_EQ(recovered.size(), 2);
    EXPECT_TRUE(recovered.commit("one", 1, kVector).replayed);
    EXPECT_TRUE(recovered.commit("two", 2, kVector).replayed);
    EXPECT_EQ(recovered.durability_stats().checkpoint_lsn,
              boundary == WalBoundary::BeforeCheckpointPublish ? 1 : 2);
  }
}
std::filesystem::path active_wal(const Directory& directory) {
  const auto root = read(directory.path / "recovery.manifest");
  if (root.size() != 168) throw std::runtime_error("expected segmented manifest");
  return directory.path /
         std::string(root.begin() + 96, std::find(root.begin() + 96, root.begin() + 160, 0));
}
TEST(WalRotationTest, AutomaticRotationKeepsLsnRetryAndOnlyCurrentRecoveryArtifacts) {
  Directory directory;
  auto options = directory.options();
  options.rotate_on_checkpoint = true;
  options.maximum_wal_bytes = 150;
  {
    DurableFlatIndex index(options);
    for (std::uint64_t id = 1; id <= 8; ++id)
      EXPECT_EQ(index.commit("op" + std::to_string(id), id, kVector).lsn, id);
    EXPECT_GT(index.durability_stats().wal_generation, 0);
    EXPECT_LE(index.durability_stats().active_wal_bytes, 150);
    EXPECT_FALSE(std::filesystem::exists(directory.path / "operations.wal"));
    const auto generation = index.durability_stats().wal_generation;
    for (std::uint64_t id = 1; id <= 8; ++id)
      EXPECT_TRUE(index.commit("op" + std::to_string(id), id, kVector).replayed);
    EXPECT_EQ(index.durability_stats().wal_generation, generation);
    const auto before = index.search(kVector, 8);
    EXPECT_EQ(index.checkpoint(), 8);
    EXPECT_EQ(index.durability_stats().active_wal_bytes, 72);
    EXPECT_EQ(index.checkpoint(), 8);  // Repeated same-LSN checkpoints must have distinct segments.
    EXPECT_EQ(index.search(kVector, 8), before);
    EXPECT_FALSE(index.durability_stats().cleanup_pending);
    EXPECT_EQ(std::distance(std::filesystem::directory_iterator(directory.path), {}), 5);
  }
  options.rotate_on_checkpoint = false;  // Persisted mode must remain authoritative.
  DurableFlatIndex recovered(options);
  EXPECT_EQ(recovered.size(), 8);
  EXPECT_EQ(recovered.durability_stats().wal_scan_bytes, 0);
  EXPECT_EQ(recovered.commit("next", 9, kVector).lsn, 9);
  EXPECT_EQ(recovered.checkpoint(), 9);
}
TEST(WalRotationTest, CheckpointAndSingleRecordCapsFailBeforeNewWrite) {
  Directory directory;
  auto options = directory.options();
  options.rotate_on_checkpoint = true;
  options.maximum_wal_bytes = 150;
  options.maximum_checkpoint_bytes = 100;
  DurableFlatIndex index(options);
  (void)index.commit("one", 1, kVector);
  EXPECT_THROW((void)index.commit("two", 2, kVector), ResourceExhausted);
  EXPECT_FALSE(index.durability_stats().recovery_required);
  EXPECT_TRUE(index.commit("one", 1, kVector).replayed);
  EXPECT_EQ(index.durability_stats().applied_lsn, 1);
  EXPECT_THROW((void)index.commit(std::string(256, 'x'), 3, kVector), std::length_error);
}
TEST(WalRotationTest, PartialActiveTailAndWrongSegmentIdentityAreHandledExplicitly) {
  Directory directory;
  auto options = directory.options();
  options.rotate_on_checkpoint = true;
  std::filesystem::path path;
  Bytes complete;
  {
    DurableFlatIndex index(options);
    (void)index.commit("one", 1, kVector);
    (void)index.checkpoint();
    path = active_wal(directory);
    (void)index.commit("two", 2, kVector);
    complete = read(path);
  }
  auto partial = complete;
  partial.resize(complete.size() - 1);
  write(path, partial);
  {
    DurableFlatIndex index(options);
    EXPECT_EQ(index.size(), 1);
    EXPECT_EQ(index.durability_stats().discarded_tail_bytes, partial.size() - 72);
    EXPECT_EQ(index.commit("two", 2, kVector).lsn, 2);
  }
  auto wrong = read(path);
  put(wrong, 32, 1);
  put(wrong, 64, hash(wrong, 0, 64));
  write(path, wrong);
  EXPECT_THROW(DurableFlatIndex{options}, std::runtime_error);
  write(path, complete);
  std::filesystem::remove(path);
  EXPECT_THROW(DurableFlatIndex{options}, std::system_error);
  EXPECT_FALSE(std::filesystem::exists(directory.path / "operations.wal"));
}
TEST(WalRotationTest, CleanupFailureIsObservableAndRetriedAtStartupWithoutRemovingOtherFiles) {
  Directory directory;
  bool fail = true;
  auto options = directory.options();
  options.rotate_on_checkpoint = true;
  options.fault_hook = [&](WalBoundary boundary) {
    if (fail && boundary == WalBoundary::BeforeWalCleanup)
      throw std::runtime_error("cleanup failure");
  };
  {
    DurableFlatIndex index(options);
    (void)index.commit("one", 1, kVector);
    (void)index.checkpoint();
    EXPECT_TRUE(index.durability_stats().cleanup_pending);
    EXPECT_FALSE(index.durability_stats().recovery_required);
    EXPECT_TRUE(std::filesystem::exists(directory.path / "operations.wal"));
    EXPECT_EQ(index.commit("two", 2, kVector).lsn, 2);
  }
  write(directory.path / "notes.txt", Bytes{1, 2, 3});
  write(directory.path / "wal-user-backup", Bytes{4});
  fail = false;
  DurableFlatIndex recovered(options);
  EXPECT_EQ(recovered.size(), 2);
  EXPECT_FALSE(recovered.durability_stats().cleanup_pending);
  EXPECT_FALSE(std::filesystem::exists(directory.path / "operations.wal"));
  EXPECT_TRUE(std::filesystem::exists(directory.path / "notes.txt"));
  EXPECT_TRUE(std::filesystem::exists(directory.path / "wal-user-backup"));
  EXPECT_TRUE(recovered.commit("one", 1, kVector).replayed);
}
TEST(WalRotationTest, SigkillAcrossPublicationSwitchAndCleanupPreservesAcknowledgedWrites) {
  for (const auto boundary : {WalBoundary::AfterWalCreated, WalBoundary::BeforeCheckpointPublish,
                              WalBoundary::AfterCheckpointRename, WalBoundary::BeforeWriterSwitch,
                              WalBoundary::AfterWriterSwitch, WalBoundary::BeforeWalCleanup,
                              WalBoundary::AfterWalCleanup}) {
    Directory directory;
    int descriptors[2];
    ASSERT_EQ(pipe(descriptors), 0);
    const auto pid = fork();
    ASSERT_GE(pid, 0);
    if (pid == 0) {
      close(descriptors[0]);
      bool armed = false;
      auto options = directory.options();
      options.rotate_on_checkpoint = true;
      options.fault_hook = [&](WalBoundary actual) {
        if (armed && actual == boundary) {
          kill(getpid(), SIGKILL);
          _exit(94);
        }
      };
      DurableFlatIndex index(options);
      (void)index.commit("one", 1, kVector);
      (void)index.checkpoint();
      (void)index.commit("two", 2, kVector);
      const char ack = '2';
      if (::write(descriptors[1], &ack, 1) != 1) _exit(95);
      armed = true;
      (void)index.checkpoint();
      _exit(96);
    }
    close(descriptors[1]);
    char ack = 0;
    ASSERT_EQ(::read(descriptors[0], &ack, 1), 1);
    close(descriptors[0]);
    int status = 0;
    ASSERT_EQ(waitpid(pid, &status, 0), pid);
    ASSERT_TRUE(WIFSIGNALED(status));
    EXPECT_EQ(WTERMSIG(status), SIGKILL);
    EXPECT_EQ(ack, '2');
    DurableFlatIndex recovered(directory.options());
    EXPECT_EQ(recovered.size(), 2);
    EXPECT_TRUE(recovered.commit("one", 1, kVector).replayed);
    EXPECT_TRUE(recovered.commit("two", 2, kVector).replayed);
    EXPECT_EQ(recovered.commit("three", 3, kVector).lsn, 3);
    EXPECT_EQ(std::distance(std::filesystem::directory_iterator(directory.path), {}), 5);
  }
}
TEST(WalRotationTest, RejectsSharedCleanupDirectoryAndFailsClosedAtWriterSwitch) {
  {
    Directory directory;
    auto options = directory.options();
    options.rotate_on_checkpoint = true;
    std::filesystem::permissions(directory.path, std::filesystem::perms::group_write,
                                 std::filesystem::perm_options::add);
    EXPECT_THROW(DurableFlatIndex{options}, std::invalid_argument);
  }
  for (const auto boundary : {WalBoundary::BeforeWriterSwitch, WalBoundary::AfterWriterSwitch}) {
    Directory directory;
    auto options = directory.options();
    options.rotate_on_checkpoint = true;
    options.fault_hook = [&](WalBoundary actual) {
      if (actual == boundary) throw std::runtime_error("switch failure");
    };
    {
      DurableFlatIndex index(options);
      (void)index.commit("one", 1, kVector);
      EXPECT_THROW((void)index.checkpoint(), std::runtime_error);
      EXPECT_TRUE(index.durability_stats().recovery_required);
      EXPECT_THROW((void)index.commit("two", 2, kVector), RecoveryRequiredError);
    }
    DurableFlatIndex recovered(directory.options());
    EXPECT_TRUE(recovered.commit("one", 1, kVector).replayed);
    EXPECT_EQ(recovered.commit("two", 2, kVector).lsn, 2);
  }
}
TEST(WalRotationTest, CheckpointVerificationHonorsConfiguredDimensionLimit) {
  Directory directory;
  auto options = directory.options();
  options.rotate_on_checkpoint = true;
  options.dimension = 70000;
  options.limits.maximum_dimension = 70000;
  const std::vector<float> values(options.dimension, 1);
  {
    DurableFlatIndex index(options);
    (void)index.commit("wide", 1, values);
    EXPECT_EQ(index.checkpoint(), 1);
  }
  DurableFlatIndex recovered(options);
  EXPECT_TRUE(recovered.commit("wide", 1, values).replayed);
  EXPECT_EQ(recovered.dimension(), 70000);
}
TEST(WalRotationTest, MissingManifestNeverBootstrapsAnEmptyReplacement) {
  {
    Directory orphan;
    auto options = orphan.options();
    options.rotate_on_checkpoint = true;
    options.fault_hook = [](WalBoundary boundary) {
      if (boundary == WalBoundary::BeforeCheckpointPublish)
        throw std::runtime_error("initial publication failed");
    };
    {
      DurableFlatIndex index(options);
      (void)index.commit("one", 1, kVector);
      EXPECT_THROW((void)index.checkpoint(), std::runtime_error);
      EXPECT_FALSE(index.durability_stats().recovery_required);
      EXPECT_EQ(index.commit("two", 2, kVector).lsn, 2);
    }
    DurableFlatIndex recovered(orphan.options());
    EXPECT_TRUE(recovered.commit("one", 1, kVector).replayed);
    EXPECT_TRUE(recovered.commit("two", 2, kVector).replayed);
  }
  Directory directory;
  auto options = directory.options();
  options.rotate_on_checkpoint = true;
  {
    DurableFlatIndex index(options);
    (void)index.commit("one", 1, kVector);
    (void)index.checkpoint();
  }
  std::filesystem::remove(directory.path / "recovery.manifest");
  EXPECT_THROW(DurableFlatIndex{directory.options()}, std::runtime_error);
  EXPECT_FALSE(std::filesystem::exists(directory.path / "operations.wal"));
}
}  // namespace
