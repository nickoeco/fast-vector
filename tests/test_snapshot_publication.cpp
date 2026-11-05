#include <gtest/gtest.h>
#include <openssl/evp.h>
#include <unistd.h>

#include <atomic>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <sstream>

#include "fast_vector/flat_index_io.h"
#include "fast_vector/hnsw_index_io.h"
#include "service/snapshot_publication.h"

namespace {
using namespace fast_vector;
using namespace fast_vector::service;

class Artifacts {
 public:
  Artifacts() {
    static std::atomic<unsigned> sequence{0};
    directory = std::filesystem::temp_directory_path() /
                ("fast-vector-publication-test-" + std::to_string(getpid()) + "-" +
                 std::to_string(sequence++));
    std::filesystem::create_directory(directory);
  }
  ~Artifacts() {
    std::error_code error;
    std::filesystem::remove_all(directory, error);
  }
  v1::SnapshotManifest save(bool hnsw, const std::string& name, VectorId id) {
    const auto file = directory / name;
    if (hnsw) {
      HnswIndex index({.dimension = 2, .max_connections = 2, .ef_construction = 8});
      index.add(id, std::vector<float>{1, 0});
      save_hnsw_index(index, file);
    } else {
      FlatIndex index(2);
      index.add(id, std::vector<float>{1, 0});
      save_flat_index(index, file);
    }
    v1::SnapshotManifest manifest;
    manifest.set_index_type(hnsw ? "hnsw" : "flat");
    manifest.set_format_version(hnsw ? 3 : 1);
    manifest.set_dimension(2);
    manifest.set_vector_count(1);
    manifest.set_file_size(std::filesystem::file_size(file));
    manifest.set_embedding_revision("embed-r1");
    digest(file, manifest);
    return manifest;
  }
  static void digest(const std::filesystem::path& path, v1::SnapshotManifest& manifest) {
    std::ifstream input(path, std::ios::binary);
    const std::string bytes((std::istreambuf_iterator<char>(input)),
                            std::istreambuf_iterator<char>());
    unsigned char output[EVP_MAX_MD_SIZE];
    unsigned size = 0;
    ASSERT_EQ(EVP_Digest(bytes.data(), bytes.size(), output, &size, EVP_sha256(), nullptr), 1);
    std::ostringstream sha;
    for (unsigned i = 0; i < size; ++i)
      sha << std::hex << std::setw(2) << std::setfill('0') << static_cast<unsigned>(output[i]);
    manifest.set_sha256(sha.str());
    manifest.set_snapshot_id("sha256:" + sha.str());
  }
  std::shared_ptr<VectorStore> store(const std::string& path,
                                     const v1::SnapshotManifest& manifest) {
    return std::make_shared<VectorStore>(
        load_verified_snapshot(directory / path, manifest, manifest.index_type(),
                               DotProductKernel::Scalar, 1024 * 1024),
        10, true, snapshot_identity(manifest, 1));
  }
  std::filesystem::path directory;
};

TEST(SnapshotPublicationTest, ValidatesDigestShapeFormatAndFullIndexForBothTypes) {
  Artifacts files;
  for (const bool hnsw : {false, true}) {
    const auto manifest = files.save(hnsw, "index", 10);
    auto load = [&](const v1::SnapshotManifest& descriptor) {
      return load_verified_snapshot(files.directory / "index", descriptor, manifest.index_type(),
                                    DotProductKernel::Scalar, 1024 * 1024);
    };
    EXPECT_EQ(load(manifest)->size(), 1U);
    auto bad = manifest;
    bad.set_vector_count(2);
    EXPECT_THROW((void)load(bad), std::invalid_argument);
    bad = manifest;
    bad.set_dimension(3);
    EXPECT_THROW((void)load(bad), std::invalid_argument);
    bad = manifest;
    bad.set_file_size(manifest.file_size() + 1);
    EXPECT_THROW((void)load(bad), std::invalid_argument);
    bad = manifest;
    bad.set_format_version(99);
    EXPECT_THROW((void)load(bad), std::invalid_argument);
    bad = manifest;
    bad.set_sha256(std::string(64, '0'));
    bad.set_snapshot_id("sha256:" + bad.sha256());
    EXPECT_THROW((void)load(bad), std::invalid_argument);
    bad = manifest;
    bad.set_embedding_revision(std::string(257, 'x'));
    EXPECT_THROW((void)load(bad), std::invalid_argument);
    bad = manifest;
    bad.GetReflection()->MutableUnknownFields(&bad)->AddLengthDelimited(9999,
                                                                        std::string(17000, 'x'));
    EXPECT_THROW((void)load(bad), std::invalid_argument);
    EXPECT_THROW(validate_manifest(manifest, hnsw ? "flat" : "hnsw", 1024 * 1024),
                 std::invalid_argument);
    EXPECT_THROW(validate_manifest(manifest, manifest.index_type(), 1), std::invalid_argument);
    {
      std::fstream output(files.directory / "index",
                          std::ios::binary | std::ios::in | std::ios::out);
      output.put('X');
    }
    EXPECT_THROW((void)load(manifest), std::invalid_argument);
    bad = manifest;
    Artifacts::digest(files.directory / "index", bad);
    EXPECT_THROW((void)load(bad), std::runtime_error);
  }
}

TEST(SnapshotPublicationTest, PreparesActivatesAndRollsBackWithHigherGeneration) {
  Artifacts files;
  for (const bool hnsw : {false, true}) {
    const auto first = files.save(hnsw, "a", 10);
    auto second = files.save(hnsw, "b", 20);
    second.set_embedding_revision("embed-r2");
    auto store = files.store("a", first);
    SnapshotPublication publication(store, first.index_type(), {.enabled = true});
    auto token = publication.begin(second, 2);
    EXPECT_EQ(publication.status().state(), "PREPARING");
    EXPECT_THROW((void)publication.begin(second, 3), PublicationBusyError);
    publication.complete(files.directory / "b", token);
    EXPECT_EQ(publication.status().state(), "READY_TO_ACTIVATE");
    EXPECT_EQ(store->search(std::vector<float>{1, 0}, 1)[0].id, 10U);
    EXPECT_THROW(publication.activate("wrong", 2), SnapshotMismatchError);
    publication.activate(second.snapshot_id(), 2);
    EXPECT_EQ(publication.status().last_attempt().snapshot_id(), second.snapshot_id());
    publication.activate(second.snapshot_id(), 2);
    EXPECT_EQ(store->stats().snapshot_version, 2U);
    EXPECT_EQ(store->stats().identity.embedding_revision, "embed-r2");
    SearchContext expected;
    expected.expected_snapshot_id = first.snapshot_id();
    EXPECT_THROW((void)store->search(std::vector<float>{1, 0}, 1, expected), SnapshotMismatchError);
    EXPECT_THROW((void)store->search({}, 0, expected), SnapshotMismatchError);
    token = publication.begin(first, 3);
    publication.complete(files.directory / "a", token);
    publication.activate(first.snapshot_id(), 3);
    const auto result = store->search(std::vector<float>{1, 0}, 1, expected);
    EXPECT_EQ(result.neighbors[0].id, 10U);
    EXPECT_EQ(result.identity.snapshot_id, first.snapshot_id());
    EXPECT_EQ(result.identity.generation, 3U);
    EXPECT_EQ(result.identity.embedding_revision, "embed-r1");
    EXPECT_THROW((void)publication.begin(second, 2), SnapshotMismatchError);
    EXPECT_THROW(store->replace_index(std::make_unique<FlatIndex>(2)), SnapshotMismatchError);
    EXPECT_EQ(publication.status().last_result(), "ACTIVATED");
  }
}

TEST(SnapshotPublicationTest, FailedPrepareRetainsFenceAndLateFailureCannotEraseNewCandidate) {
  Artifacts files;
  const auto a = files.save(false, "a", 10);
  const auto b = files.save(false, "b", 20);
  auto store = files.store("a", a);
  SnapshotPublication publication(store, "flat", {.enabled = true});
  const auto failed = publication.begin(b, 2);
  EXPECT_THROW(publication.complete(files.directory / "a", failed), std::invalid_argument);
  publication.fail(failed);
  EXPECT_EQ(publication.status().state(), "FAILED");
  EXPECT_EQ(publication.status().highest_generation(), 2U);
  EXPECT_EQ(store->stats().identity.generation, 1U);
  EXPECT_THROW(publication.activate(a.snapshot_id(), 1), SnapshotMismatchError);
  EXPECT_THROW((void)publication.begin(a, 2), SnapshotMismatchError);
  const auto retry = publication.begin(b, 2);
  publication.complete(files.directory / "b", retry);
  const auto newer = publication.begin(a, 3);
  publication.fail(retry);
  EXPECT_EQ(publication.status().state(), "PREPARING");
  EXPECT_EQ(publication.status().candidate().generation(), 3U);
  publication.complete(files.directory / "a", newer);
  EXPECT_THROW(publication.activate(b.snapshot_id(), 2), SnapshotMismatchError);
  publication.activate(a.snapshot_id(), 3);
  EXPECT_EQ(publication.status().current().generation(), 3U);
}

TEST(SnapshotPublicationTest, DisabledPublicationAndPrivateStagingCleanup) {
  Artifacts files;
  const auto a = files.save(false, "a", 10);
  SnapshotPublication disabled(files.store("a", a), "flat", {});
  EXPECT_THROW((void)disabled.begin(a, 2), PublicationDisabledError);
  EXPECT_THROW(disabled.activate(a.snapshot_id(), 2), PublicationDisabledError);
  auto rejected = a;
  rejected.set_snapshot_id(std::string(4096, 'x'));
  rejected.set_embedding_revision(std::string(4096, 'x'));
  disabled.rejected(rejected, 9);
  EXPECT_TRUE(disabled.status().last_attempt().snapshot_id().empty());
  EXPECT_TRUE(disabled.status().last_attempt().embedding_revision().empty());
  EXPECT_EQ(disabled.status().current().generation(), 1U);
  std::filesystem::path path;
  {
    SnapshotStagingFile temporary(files.directory);
    path = temporary.path();
    EXPECT_TRUE(std::filesystem::is_regular_file(path));
    EXPECT_EQ(std::filesystem::status(path).permissions() & std::filesystem::perms::others_all,
              std::filesystem::perms::none);
  }
  EXPECT_FALSE(std::filesystem::exists(path));
}
}  // namespace
