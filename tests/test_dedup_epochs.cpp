#include <gtest/gtest.h>
#include <sys/wait.h>
#include <unistd.h>

#include <algorithm>
#include <array>
#include <csignal>
#include <filesystem>
#include <fstream>
#include <iterator>

#include "fast_vector/durable_flat_index.h"
#include "fast_vector/vector_store.h"

namespace {
using namespace fast_vector;
const std::vector<float> values{1, 2, 3};
class Directory {
 public:
  Directory() {
    std::string pattern = "/tmp/fast-vector-epochs-XXXXXX";
    auto* result = mkdtemp(pattern.data());
    if (!result) throw std::runtime_error("mkdtemp failed");
    path = result;
  }
  ~Directory() {
    std::error_code error;
    std::filesystem::remove_all(path, error);
  }
  WalOptions options(bool bounded = true) const {
    WalOptions result{.directory = path, .dimension = 3, .fault_hook = {}};
    result.dedup_retention = bounded;
    result.maximum_epoch_operations = 10;
    result.maximum_epoch_bytes = 2048;
    return result;
  }
  std::filesystem::path path;
};
using Bytes = std::vector<unsigned char>;
Bytes read(const std::filesystem::path& path) {
  std::ifstream input(path, std::ios::binary);
  return {std::istreambuf_iterator<char>(input), {}};
}
void write(const std::filesystem::path& path, const Bytes& bytes) {
  std::ofstream output(path, std::ios::binary | std::ios::trunc);
  output.write(reinterpret_cast<const char*>(bytes.data()), bytes.size());
}
void put(Bytes& bytes, std::size_t offset, std::uint64_t value) {
  for (std::size_t i = 0; i < 8; ++i) {
    bytes.at(offset + i) = value & 255U;
    value >>= 8;
  }
}
std::uint64_t hash(const Bytes& bytes, std::size_t begin, std::size_t end) {
  std::uint64_t result = 14695981039346656037ULL;
  for (auto i = begin; i < end; ++i) {
    result ^= bytes[i];
    result *= 1099511628211ULL;
  }
  return result;
}
TEST(DedupEpochTest, WindowPersistsAndExpiredOperationsNeverBecomeNewWrites) {
  Directory directory;
  {
    DurableFlatIndex index(directory.options());
    EXPECT_EQ(index.durability_stats().current_epoch, 1);
    EXPECT_EQ(index.commit("same", 1, values, nullptr, 1).lsn, 1);
    (void)index.checkpoint();
    EXPECT_TRUE(index.commit("same", 1, values, nullptr, 1).replayed);
    EXPECT_THROW((void)index.commit("new", 99, values, nullptr, 1), DedupEpochError);
    EXPECT_THROW((void)index.commit("new", 99, values, nullptr, 3), DedupEpochError);
    EXPECT_THROW((void)index.commit("new", 99, values), DedupEpochError);
    EXPECT_EQ(index.commit("same", 2, values, nullptr, 2).lsn, 2);
    EXPECT_THROW((void)index.commit("same", 3, values, nullptr, 2), std::invalid_argument);
    (void)index.checkpoint();  // current=3, window=[1,3]
    EXPECT_TRUE(index.commit("same", 1, values, nullptr, 1).replayed);
    (void)index.checkpoint();  // current=4, window=[2,4]
    EXPECT_EQ(index.durability_stats().oldest_epoch, 2);
    EXPECT_EQ(index.durability_stats().dedup_entries, 1);
    EXPECT_THROW((void)index.commit("same", 1, values, nullptr, 1), DedupEpochError);
    EXPECT_THROW((void)index.commit("unknown", 99, values, nullptr, 1), DedupEpochError);
    EXPECT_EQ(index.size(), 2);
    EXPECT_EQ(index.search(values, 10).size(), 2);
  }
  DurableFlatIndex recovered(directory.options(false));
  EXPECT_EQ(recovered.durability_stats().current_epoch, 4);
  EXPECT_TRUE(recovered.commit("same", 2, values, nullptr, 2).replayed);
  EXPECT_THROW((void)recovered.commit("same", 1, values, nullptr, 1), DedupEpochError);
  EXPECT_EQ(recovered.commit("next", 3, values, nullptr, 4).lsn, 3);
}
TEST(DedupEpochTest, CountAndByteCapsPreserveLsnAndAllowExistingRetries) {
  for (const bool byte_limit : {false, true}) {
    Directory directory;
    auto options = directory.options();
    options.maximum_epoch_operations = byte_limit ? 10 : 1;
    options.maximum_epoch_bytes = byte_limit ? 103 : 2048;  // 80 + 8 + 3 + 12
    DurableFlatIndex index(options);
    (void)index.commit("one", 1, values, nullptr, 1);
    EXPECT_THROW((void)index.commit("two", 2, values, nullptr, 1), ResourceExhausted);
    EXPECT_TRUE(index.commit("one", 1, values, nullptr, 1).replayed);
    EXPECT_FALSE(index.durability_stats().recovery_required);
    EXPECT_EQ(index.durability_stats().applied_lsn, 1);
    (void)index.checkpoint();
    EXPECT_EQ(index.commit("two", 2, values, nullptr, 2).lsn, 2);
  }
}
TEST(DedupEpochTest, AutomaticWalRotationDoesNotCloseCurrentEpoch) {
  Directory directory;
  auto options = directory.options();
  options.maximum_wal_bytes = 170;
  {
    DurableFlatIndex index(options);
    for (std::uint64_t id = 1; id <= 5; ++id)
      (void)index.commit("op" + std::to_string(id), id, values, nullptr, 1);
    EXPECT_GT(index.durability_stats().wal_generation, 1);
    EXPECT_EQ(index.durability_stats().current_epoch, 1);
    EXPECT_EQ(index.durability_stats().current_epoch_operations, 5);
  }
  DurableFlatIndex recovered(directory.options(false));
  EXPECT_EQ(recovered.durability_stats().current_epoch_operations, 5);
  for (std::uint64_t id = 1; id <= 5; ++id)
    EXPECT_TRUE(recovered.commit("op" + std::to_string(id), id, values, nullptr, 1).replayed);
}
TEST(DedupEpochTest, LegacyEpochZeroMigrationIsExplicitAndQuotaChecked) {
  Directory directory;
  {
    DurableFlatIndex legacy(directory.options(false));
    (void)legacy.commit("legacy-one", 1, values);
    (void)legacy.commit("legacy-two", 2, values);
    (void)legacy.checkpoint();
  }
  auto small = directory.options();
  small.maximum_epoch_operations = 1;
  EXPECT_THROW(DurableFlatIndex{small}, ResourceExhausted);
  {
    DurableFlatIndex legacy(directory.options(false));
    EXPECT_TRUE(legacy.commit("legacy-one", 1, values).replayed);
  }
  {
    DurableFlatIndex index(directory.options());
    EXPECT_TRUE(index.commit("legacy-one", 1, values).replayed);
    EXPECT_THROW((void)index.commit("missing", 99, values), DedupEpochError);
    (void)index.checkpoint();
    (void)index.checkpoint();  // epoch 3 expires epoch 0
    EXPECT_THROW((void)index.commit("legacy-one", 1, values), DedupEpochError);
    EXPECT_EQ(index.durability_stats().dedup_entries, 0);
    EXPECT_EQ(index.size(), 2);
  }
  auto mismatch = directory.options();
  mismatch.retained_epochs = 4;
  EXPECT_THROW(DurableFlatIndex{mismatch}, std::invalid_argument);
  DurableFlatIndex recovered(directory.options(false));
  EXPECT_EQ(recovered.durability_stats().oldest_epoch, 1);
}
TEST(DedupEpochTest, FailedPublicationDoesNotAdvanceOrExpireTheWindow) {
  Directory directory;
  bool fail = false;
  auto options = directory.options();
  options.retained_epochs = 1;
  options.fault_hook = [&](WalBoundary boundary) {
    if (fail && boundary == WalBoundary::BeforeCheckpointPublish)
      throw std::runtime_error("publish failed");
  };
  {
    DurableFlatIndex index(options);
    (void)index.commit("one", 1, values, nullptr, 1);
    fail = true;
    EXPECT_THROW((void)index.checkpoint(), std::runtime_error);
    EXPECT_EQ(index.durability_stats().current_epoch, 1);
    EXPECT_TRUE(index.commit("one", 1, values, nullptr, 1).replayed);
    EXPECT_FALSE(index.durability_stats().recovery_required);
  }
  DurableFlatIndex recovered(directory.options(false));
  EXPECT_TRUE(recovered.commit("one", 1, values, nullptr, 1).replayed);
  (void)recovered.checkpoint();
  EXPECT_EQ(recovered.durability_stats().dedup_entries, 0);
  EXPECT_THROW((void)recovered.commit("one", 1, values, nullptr, 1), DedupEpochError);
}
TEST(DedupEpochTest, CorruptPolicyAndWalEpochAreRejectedWithRecomputedChecksums) {
  Directory directory;
  {
    DurableFlatIndex index(directory.options());
    (void)index.commit("one", 1, values, nullptr, 1);
  }
  const auto root = directory.path / "recovery.manifest";
  const auto original = read(root);
  auto corrupt = original;
  put(corrupt, 168, 2);
  put(corrupt, 200, hash(corrupt, 0, 200));
  write(root, corrupt);
  EXPECT_THROW(DurableFlatIndex{directory.options(false)}, std::runtime_error);
  write(root, original);
  const auto wal =
      directory.path / std::string(original.begin() + 96,
                                   std::find(original.begin() + 96, original.begin() + 160, 0));
  auto bytes = read(wal);
  put(bytes, 72 + 40, 2);
  put(bytes, bytes.size() - 8, hash(bytes, 72, bytes.size() - 8));
  write(wal, bytes);
  EXPECT_THROW(DurableFlatIndex{directory.options(false)}, std::runtime_error);
  bytes.resize(72 + 48);
  write(wal, bytes);
  EXPECT_THROW(DurableFlatIndex{directory.options(false)}, std::runtime_error);
}
TEST(DedupEpochTest, SigkillAroundPublicationKeepsDurableExpiryFence) {
  for (const auto boundary :
       {WalBoundary::BeforeCheckpointPublish, WalBoundary::AfterCheckpointRename,
        WalBoundary::BeforeWriterSwitch, WalBoundary::AfterWalCleanup}) {
    Directory directory;
    int pipe_fd[2];
    ASSERT_EQ(pipe(pipe_fd), 0);
    const auto pid = fork();
    ASSERT_GE(pid, 0);
    if (pid == 0) {
      close(pipe_fd[0]);
      bool armed = false;
      auto options = directory.options();
      options.retained_epochs = 1;
      options.fault_hook = [&](WalBoundary actual) {
        if (armed && boundary == actual) {
          kill(getpid(), SIGKILL);
          _exit(90);
        }
      };
      DurableFlatIndex index(options);
      (void)index.commit("one", 1, values, nullptr, 1);
      const char ack = 'A';
      if (::write(pipe_fd[1], &ack, 1) != 1) _exit(91);
      armed = true;
      (void)index.checkpoint();
      _exit(92);
    }
    close(pipe_fd[1]);
    char ack = 0;
    ASSERT_EQ(::read(pipe_fd[0], &ack, 1), 1);
    close(pipe_fd[0]);
    int status = 0;
    ASSERT_EQ(waitpid(pid, &status, 0), pid);
    ASSERT_TRUE(WIFSIGNALED(status));
    EXPECT_EQ(WTERMSIG(status), SIGKILL);
    EXPECT_EQ(ack, 'A');
    DurableFlatIndex recovered(directory.options(false));
    EXPECT_EQ(recovered.size(), 1);
    if (boundary == WalBoundary::BeforeCheckpointPublish)
      EXPECT_TRUE(recovered.commit("one", 1, values, nullptr, 1).replayed);
    else
      EXPECT_THROW((void)recovered.commit("one", 1, values, nullptr, 1), DedupEpochError);
  }
}
TEST(DedupEpochTest, RetainedMetadataAndFileSizeStayBoundedAcrossManyEpochs) {
  Directory directory;
  auto options = directory.options();
  options.maximum_epoch_operations = 2;
  std::uint64_t stable_metadata = 0, stable_file = 0;
  {
    DurableFlatIndex index(options);
    for (std::uint64_t epoch = 1; epoch <= 20; ++epoch) {
      (void)index.commit("op-A", epoch * 2 - 1, values, nullptr, epoch);
      (void)index.commit("op-B", epoch * 2, values, nullptr, epoch);
      (void)index.checkpoint();
      const auto stats = index.durability_stats();
      EXPECT_LE(stats.dedup_entries, options.retained_epochs * options.maximum_epoch_operations);
      EXPECT_LE(stats.dedup_file_bytes, 64 + options.retained_epochs * options.maximum_epoch_bytes);
      if (epoch == 3) {
        stable_metadata = stats.dedup_metadata_bytes;
        stable_file = stats.dedup_file_bytes;
      }
      if (epoch > 3) {
        EXPECT_EQ(stats.dedup_metadata_bytes, stable_metadata);
        EXPECT_EQ(stats.dedup_file_bytes, stable_file);
      }
    }
    EXPECT_EQ(index.size(), 40);
    EXPECT_EQ(index.search(values, 100).size(), 40);
  }
  DurableFlatIndex recovered(directory.options(false));
  EXPECT_EQ(recovered.durability_stats().current_epoch, 21);
  EXPECT_EQ(recovered.durability_stats().dedup_entries, 4);
  EXPECT_THROW((void)recovered.commit("op-A", 1, values, nullptr, 1), DedupEpochError);
  EXPECT_TRUE(recovered.commit("op-A", 39, values, nullptr, 20).replayed);
}
TEST(DedupEpochTest, SidecarEpochOutsideWindowIsRejectedRatherThanSilentlyPruned) {
  Directory directory;
  {
    DurableFlatIndex index(directory.options());
    (void)index.commit("one", 1, values, nullptr, 1);
    (void)index.checkpoint();
    (void)index.checkpoint();
  }
  const auto root = directory.path / "recovery.manifest";
  const auto original_root = read(root);
  std::filesystem::path sidecar;
  for (const auto& entry : std::filesystem::directory_iterator(directory.path))
    if (entry.path().extension() == ".fvd") sidecar = entry.path();
  const auto original = read(sidecar);
  for (const auto epoch : {0U, 4U}) {
    auto bytes = original;
    put(bytes, 64 + 72, epoch);
    put(bytes, bytes.size() - 8, hash(bytes, 64, bytes.size() - 8));
    write(sidecar, bytes);
    auto manifest = original_root;
    put(manifest, 72, hash(bytes, 0, bytes.size()));
    put(manifest, 200, hash(manifest, 0, 200));
    write(root, manifest);
    EXPECT_THROW(DurableFlatIndex{directory.options(false)}, std::runtime_error);
  }
}
TEST(DedupEpochTest, CheckpointExpectedEpochRejectsDuplicateAdvanceUnderStoreLock) {
  Directory directory;
  VectorStore store(std::make_unique<DurableFlatIndex>(directory.options()));
  EXPECT_THROW((void)store.checkpoint(), DedupEpochError);
  EXPECT_EQ(store.add(1, values, "one", 1).lsn, 1);
  const auto result = store.checkpoint_result(1);
  EXPECT_EQ(result.lsn, 1);
  EXPECT_EQ(result.current_epoch, 2);
  EXPECT_EQ(result.oldest_epoch, 0);
  EXPECT_THROW((void)store.checkpoint(1), DedupEpochError);
  EXPECT_EQ(store.durability_stats().current_epoch, 2);
  EXPECT_TRUE(store.add(1, values, "one", 1).replayed);
}
}  // namespace
