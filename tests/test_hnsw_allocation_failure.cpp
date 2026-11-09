#include <gtest/gtest.h>

#include <atomic>
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <iterator>
#include <new>
#include <vector>

#include "fast_vector/durable_flat_index.h"
#include "fast_vector/flat_index.h"
#include "fast_vector/hnsw_index_io.h"
#include "fast_vector/vector_store.h"

namespace {
thread_local std::ptrdiff_t allocations_left = -1;

class FailAllocationsAfter {
 public:
  explicit FailAllocationsAfter(const std::ptrdiff_t count) { allocations_left = count; }
  ~FailAllocationsAfter() { allocations_left = -1; }
  FailAllocationsAfter(const FailAllocationsAfter&) = delete;
  FailAllocationsAfter& operator=(const FailAllocationsAfter&) = delete;
};
}  // namespace

// Only this dedicated test executable replaces allocation. Production code has no test hook.
// Failure stays armed after the first rejection, so rollback must work without allocating.
void* operator new(const std::size_t size) {
  if (allocations_left == 0) {
    throw std::bad_alloc();
  }
  if (allocations_left > 0) {
    --allocations_left;
  }
  if (void* result = std::malloc(size == 0 ? 1 : size)) {
    return result;
  }
  throw std::bad_alloc();
}
void* operator new[](const std::size_t size) { return ::operator new(size); }
void* operator new(const std::size_t size, const std::nothrow_t&) noexcept {
  try {
    return ::operator new(size);
  } catch (...) {
    return nullptr;
  }
}
void* operator new[](const std::size_t size, const std::nothrow_t& tag) noexcept {
  return ::operator new(size, tag);
}
void operator delete(void* pointer) noexcept { std::free(pointer); }
void operator delete[](void* pointer) noexcept { std::free(pointer); }
void operator delete(void* pointer, std::size_t) noexcept { std::free(pointer); }
void operator delete[](void* pointer, std::size_t) noexcept { std::free(pointer); }
void operator delete(void* pointer, const std::nothrow_t&) noexcept { std::free(pointer); }
void operator delete[](void* pointer, const std::nothrow_t&) noexcept { std::free(pointer); }

namespace {

class Snapshot {
 public:
  Snapshot() {
    static std::atomic<unsigned> sequence{0};
    path_ = std::filesystem::temp_directory_path() /
            ("fast-vector-failure-" + std::to_string(sequence.fetch_add(1)) + "-" +
             std::to_string(reinterpret_cast<std::uintptr_t>(this)) + ".fvh");
  }
  ~Snapshot() {
    std::error_code error;
    std::filesystem::remove(path_, error);
  }
  std::vector<char> bytes(const fast_vector::HnswIndex& index) const {
    fast_vector::save_hnsw_index(index, path_);
    std::ifstream input(path_, std::ios::binary);
    return {std::istreambuf_iterator<char>(input), std::istreambuf_iterator<char>()};
  }

 private:
  std::filesystem::path path_;
};

void verify_add_failures(const fast_vector::HnswIndex& baseline) {
  Snapshot snapshot;
  const std::vector<float> query{1, 2, 3};
  const auto before = snapshot.bytes(baseline);
  const auto neighbors = baseline.search(query, 10);
  const auto ids = baseline.live_ids();
  auto control = baseline;
  control.add(999, query);
  const auto expected = snapshot.bytes(control);
  std::size_t failures = 0;
  bool completed = false;
  for (std::ptrdiff_t point = 0; point < 10000; ++point) {
    auto subject = baseline;
    bool rejected = false;
    try {
      const FailAllocationsAfter failure(point);
      subject.add(999, query);
    } catch (const std::bad_alloc&) {
      rejected = true;
    }
    SCOPED_TRACE(point);
    if (!rejected) {
      EXPECT_EQ(snapshot.bytes(subject), expected);
      completed = true;
      break;
    }
    ++failures;
    EXPECT_EQ(snapshot.bytes(subject), before);
    EXPECT_EQ(subject.live_ids(), ids);
    EXPECT_EQ(subject.search(query, 10), neighbors);
    // The failed ID must not remain reserved; retry must consume the original RNG state.
    subject.add(999, query);
    EXPECT_EQ(snapshot.bytes(subject), expected);
  }
  EXPECT_TRUE(completed);
  EXPECT_GT(failures, 0U);
  std::cout << "Verified add allocation failure points: " << failures << '\n';
}

TEST(HnswAllocationFailureTest, FirstInsertionHasStrongLogicalGuarantee) {
  verify_add_failures(fast_vector::HnswIndex({.dimension = 3}));
}

TEST(HnswAllocationFailureTest, PruningAndTombstonesHaveStrongLogicalGuarantee) {
  for (const auto selection : {fast_vector::HnswNeighborSelection::Simple,
                               fast_vector::HnswNeighborSelection::Heuristic}) {
    fast_vector::HnswIndex index({.dimension = 3,
                                  .max_connections = 2,
                                  .ef_construction = 12,
                                  .random_seed = 17,
                                  .neighbor_selection = selection});
    for (fast_vector::VectorId id = 0; id < 32; ++id) {
      index.add(id,
                std::vector<float>{static_cast<float>(id % 11 + 1), static_cast<float>(id % 7 + 2),
                                   static_cast<float>(id % 5 + 3)});
    }
    ASSERT_TRUE(index.erase(2));
    ASSERT_TRUE(index.erase(7));
    verify_add_failures(index);
  }
}

TEST(VectorStoreAllocationFailureTest, ControlBlockFailurePreservesPublishedSnapshot) {
  for (const bool hnsw : {false, true}) {
    const auto make_index = [hnsw]() -> std::unique_ptr<fast_vector::VectorIndex> {
      if (hnsw) {
        return std::make_unique<fast_vector::HnswIndex>(
            fast_vector::HnswConfig{.dimension = 3, .max_connections = 2, .ef_construction = 8});
      }
      return std::make_unique<fast_vector::FlatIndex>(3);
    };
    auto original = make_index();
    original->add(10, std::vector<float>{1, 0, 0});
    fast_vector::VectorStore store(std::move(original), 10, true);
    for (const std::ptrdiff_t point : {0, 1}) {
      auto replacement = make_index();
      replacement->add(20, std::vector<float>{1, 0, 0});
      bool rejected = false;
      try {
        const FailAllocationsAfter failure(point);
        store.replace_index(std::move(replacement));
      } catch (const std::bad_alloc&) {
        rejected = true;
      }
      ASSERT_TRUE(rejected);
      EXPECT_EQ(store.stats().snapshot_version, 1U);
    }
    EXPECT_EQ(store.stats().index_size, 1U);
    const auto results = store.search(std::vector<float>{1, 0, 0}, 1);
    ASSERT_EQ(results.size(), 1U);
    EXPECT_EQ(results[0].id, 10U);
    auto retry = make_index();
    retry->add(30, std::vector<float>{1, 0, 0});
    store.replace_index(std::move(retry));
    EXPECT_EQ(store.search(std::vector<float>{1, 0, 0}, 1)[0].id, 30U);
  }
}

TEST(HnswAllocationFailureTest, RebuildFailureLeavesSourceUnchanged) {
  fast_vector::HnswIndex source({.dimension = 3, .max_connections = 2, .ef_construction = 8});
  for (fast_vector::VectorId id = 1; id <= 6; ++id) {
    source.add(id, std::vector<float>{static_cast<float>(id), 2, 3});
  }
  ASSERT_TRUE(source.erase(2));
  ASSERT_TRUE(source.erase(5));
  Snapshot snapshot;
  const auto before = snapshot.bytes(source);
  const auto expected = snapshot.bytes(source.rebuild(123));
  bool completed = false;
  std::size_t failures = 0;
  for (std::ptrdiff_t point = 0; point < 10000; ++point) {
    bool rejected = false;
    try {
      const FailAllocationsAfter failure(point);
      (void)source.rebuild(123);
    } catch (const std::bad_alloc&) {
      rejected = true;
    }
    SCOPED_TRACE(point);
    EXPECT_EQ(snapshot.bytes(source), before);
    if (!rejected) {
      completed = true;
      break;
    }
    ++failures;
  }
  EXPECT_TRUE(completed);
  EXPECT_GT(failures, 0U);
  EXPECT_EQ(snapshot.bytes(source.rebuild(123)), expected);
  std::cout << "Verified rebuild allocation failure points: " << failures << '\n';
}

#if defined(__linux__)
TEST(WalAllocationFailureTest, EveryPostSyncAllocationFailureRequiresRecovery) {
  std::size_t failures = 0;
  bool completed = false;
  for (std::ptrdiff_t point = 0; point < 32; ++point) {
    auto directory =
        std::filesystem::temp_directory_path() /
        ("fast-vector-wal-allocation-" +
         std::to_string(reinterpret_cast<std::uintptr_t>(&failures)) + "-" + std::to_string(point));
    ASSERT_TRUE(std::filesystem::create_directory(directory));
    fast_vector::WalOptions options{.directory = directory, .dimension = 3, .fault_hook = {}};
    const std::vector<float> values{1, 2, 3};
    bool rejected = false;
    {
      auto injected = options;
      injected.fault_hook = [point](fast_vector::WalBoundary boundary) {
        if (boundary == fast_vector::WalBoundary::BeforeApply) allocations_left = point;
        if (boundary == fast_vector::WalBoundary::AfterApply) allocations_left = -1;
      };
      fast_vector::DurableFlatIndex index(injected);
      try {
        (void)index.commit("operation", 42, values);
      } catch (const std::bad_alloc&) {
        rejected = true;
      }
      allocations_left = -1;
      if (rejected) {
        EXPECT_TRUE(index.durability_stats().recovery_required);
        EXPECT_EQ(index.size(), 0);
        EXPECT_THROW((void)index.commit("next", 43, values), fast_vector::RecoveryRequiredError);
      } else {
        EXPECT_EQ(index.size(), 1);
      }
    }
    {
      fast_vector::DurableFlatIndex recovered(options);
      EXPECT_EQ(recovered.size(), 1);
      EXPECT_TRUE(recovered.commit("operation", 42, values).replayed);
      EXPECT_EQ(recovered.search(values, 1).at(0).id, 42);
    }
    std::filesystem::remove_all(directory);
    if (!rejected) {
      completed = true;
      break;
    }
    ++failures;
  }
  EXPECT_TRUE(completed);
  EXPECT_GT(failures, 0);
  std::cout << "Verified post-sync allocation failure points: " << failures << '\n';
}
void verify_checkpoint_allocation_failures(bool rotate) {
  const auto directory = std::filesystem::temp_directory_path() /
                         ("fast-vector-dedup-allocation-" +
                          std::to_string(reinterpret_cast<std::uintptr_t>(&allocations_left)) +
                          "-" + std::to_string(rotate));
  ASSERT_TRUE(std::filesystem::create_directory(directory));
  const fast_vector::WalOptions options{
      .directory = directory, .dimension = 3, .fault_hook = {}, .rotate_on_checkpoint = rotate};
  const std::vector<float> values{1, 2, 3};
  auto root_bytes = [&] {
    std::ifstream input(directory / "recovery.manifest", std::ios::binary);
    return std::vector<char>(std::istreambuf_iterator<char>(input), {});
  };
  const auto descriptor_count = [] {
    std::size_t count = 0;
    for (const auto& entry : std::filesystem::directory_iterator("/proc/self/fd")) {
      (void)entry;
      ++count;
    }
    return count;
  };
  bool completed = false;
  std::size_t failures = 0;
  {
    fast_vector::DurableFlatIndex index(options);
    (void)index.commit("one", 1, values);
    (void)index.checkpoint();
    (void)index.commit("two", 2, values);
    const auto before = root_bytes();
    const auto before_descriptors = descriptor_count();
    for (std::ptrdiff_t point = 0; point < 512; ++point) {
      bool rejected = false;
      try {
        const FailAllocationsAfter failure(point);
        (void)index.checkpoint();
      } catch (const std::bad_alloc&) {
        rejected = true;
      }
      // iostreams may translate a failed buffer allocation into an I/O exception.
      catch (const std::exception&) {
        rejected = true;
      }
      if (!rejected) {
        completed = true;
        break;
      }
      ++failures;
      EXPECT_EQ(root_bytes(), before);
      EXPECT_EQ(descriptor_count(), before_descriptors);
      EXPECT_FALSE(index.durability_stats().recovery_required);
      EXPECT_EQ(index.size(), 2);
      EXPECT_TRUE(index.commit("one", 1, values).replayed);
      EXPECT_TRUE(index.commit("two", 2, values).replayed);
    }
  }
  EXPECT_TRUE(completed);
  EXPECT_GT(failures, 0);
  {
    fast_vector::DurableFlatIndex recovered(options);
    EXPECT_EQ(recovered.durability_stats().wal_scan_bytes, 0);
    EXPECT_TRUE(recovered.commit("one", 1, values).replayed);
    EXPECT_TRUE(recovered.commit("two", 2, values).replayed);
  }
  std::filesystem::remove_all(directory);
  std::cout << "Verified checkpoint allocation failure points (rotation=" << rotate
            << "): " << failures << '\n';
}
TEST(WalAllocationFailureTest, PairedCheckpointAllocationFailuresPreserveOldRootAndRetrySources) {
  verify_checkpoint_allocation_failures(false);
}
TEST(WalAllocationFailureTest, RotatingCheckpointAllocationFailuresPreserveOldRootAndRetrySources) {
  verify_checkpoint_allocation_failures(true);
}
TEST(WalAllocationFailureTest, EpochExpiryPublicationAllocationFailuresNeverExpireOldRoot) {
  const auto directory = std::filesystem::temp_directory_path() /
                         ("fast-vector-epoch-allocation-" +
                          std::to_string(reinterpret_cast<std::uintptr_t>(&allocations_left)));
  ASSERT_TRUE(std::filesystem::create_directory(directory));
  fast_vector::WalOptions options{.directory = directory, .dimension = 3, .fault_hook = {}};
  options.dedup_retention = true;
  options.retained_epochs = 1;
  const std::vector<float> values{1, 2, 3};
  auto root_bytes = [&] {
    std::ifstream input(directory / "recovery.manifest", std::ios::binary);
    return std::vector<char>(std::istreambuf_iterator<char>(input), {});
  };
  auto descriptors = [] {
    std::size_t count = 0;
    for (const auto& entry : std::filesystem::directory_iterator("/proc/self/fd")) {
      (void)entry;
      ++count;
    }
    return count;
  };
  bool completed = false;
  std::size_t failures = 0;
  {
    fast_vector::DurableFlatIndex index(options);
    (void)index.commit("one", 1, values, nullptr, 1);
    (void)index.commit("two", 2, values, nullptr, 1);
    const auto before = root_bytes();
    const auto before_fds = descriptors();
    for (std::ptrdiff_t point = 0; point < 512; ++point) {
      bool rejected = false;
      try {
        const FailAllocationsAfter failure(point);
        (void)index.checkpoint();
      } catch (const std::exception&) {
        rejected = true;
      }
      if (!rejected) {
        completed = true;
        break;
      }
      ++failures;
      EXPECT_EQ(root_bytes(), before);
      EXPECT_EQ(descriptors(), before_fds);
      EXPECT_EQ(index.durability_stats().current_epoch, 1);
      EXPECT_FALSE(index.durability_stats().recovery_required);
      EXPECT_TRUE(index.commit("one", 1, values, nullptr, 1).replayed);
      EXPECT_TRUE(index.commit("two", 2, values, nullptr, 1).replayed);
    }
  }
  EXPECT_TRUE(completed);
  EXPECT_GT(failures, 0);
  {
    fast_vector::DurableFlatIndex recovered(options);
    EXPECT_EQ(recovered.size(), 2);
    EXPECT_EQ(recovered.durability_stats().current_epoch, 2);
    EXPECT_EQ(recovered.durability_stats().dedup_entries, 0);
    EXPECT_THROW((void)recovered.commit("one", 1, values, nullptr, 1),
                 fast_vector::DedupEpochError);
    EXPECT_EQ(recovered.commit("three", 3, values, nullptr, 2).lsn, 3);
  }
  std::filesystem::remove_all(directory);
  std::cout << "Verified expiry checkpoint allocation failure points: " << failures << '\n';
}
#endif

}  // namespace
