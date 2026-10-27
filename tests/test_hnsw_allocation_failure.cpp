#include <gtest/gtest.h>

#include <atomic>
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <iterator>
#include <new>
#include <vector>

#include "fast_vector/hnsw_index_io.h"

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

}  // namespace
