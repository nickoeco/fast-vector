#include <gtest/gtest.h>

#include <algorithm>
#include <atomic>
#include <bit>
#include <chrono>
#include <cstddef>
#include <cstdint>
#include <filesystem>
#include <fstream>
#include <functional>
#include <iterator>
#include <limits>
#include <stdexcept>
#include <string>
#include <system_error>
#include <tuple>
#include <vector>

#include "fast_vector/hnsw_index_io.h"

namespace {

class TemporaryIndexFile {
 public:
  TemporaryIndexFile() {
    static std::atomic<unsigned int> sequence{0};
    const auto stamp = std::chrono::steady_clock::now().time_since_epoch().count();
    path_ = std::filesystem::temp_directory_path() /
            ("fast-vector-hnsw-test-" + std::to_string(stamp) + "-" +
             std::to_string(sequence.fetch_add(1)) + ".fvh");
  }
  ~TemporaryIndexFile() {
    std::error_code error;
    std::filesystem::remove(path_, error);
  }
  TemporaryIndexFile(const TemporaryIndexFile&) = delete;
  TemporaryIndexFile& operator=(const TemporaryIndexFile&) = delete;
  [[nodiscard]] const std::filesystem::path& path() const noexcept { return path_; }

 private:
  std::filesystem::path path_;
};

constexpr std::size_t kHeaderSize = 112;
constexpr std::size_t kChecksumOffset = 104;

std::vector<std::uint8_t> read_bytes(const std::filesystem::path& path) {
  std::ifstream input(path, std::ios::binary);
  return {std::istreambuf_iterator<char>(input), std::istreambuf_iterator<char>()};
}

void write_bytes(const std::filesystem::path& path, const std::vector<std::uint8_t>& bytes) {
  std::ofstream output(path, std::ios::binary | std::ios::trunc);
  output.write(reinterpret_cast<const char*>(bytes.data()),
               static_cast<std::streamsize>(bytes.size()));
  ASSERT_TRUE(output);
}

std::uint64_t get_integer(const std::vector<std::uint8_t>& bytes, const std::size_t offset,
                          const std::size_t width) {
  std::uint64_t value = 0;
  for (std::size_t i = 0; i < width; ++i) {
    value |= static_cast<std::uint64_t>(bytes.at(offset + i)) << (8U * i);
  }
  return value;
}

void put_integer(std::vector<std::uint8_t>& bytes, const std::size_t offset,
                 const std::size_t width, std::uint64_t value) {
  for (std::size_t i = 0; i < width; ++i) {
    bytes.at(offset + i) = static_cast<std::uint8_t>(value & 0xFFU);
    value >>= 8U;
  }
}

void update_checksum(std::vector<std::uint8_t>& bytes) {
  std::uint64_t checksum = 14695981039346656037ULL;
  for (std::size_t i = 0; i < bytes.size(); ++i) {
    if (i >= kChecksumOffset && i < kChecksumOffset + 8) {
      continue;
    }
    checksum ^= bytes[i];
    checksum *= 1099511628211ULL;
  }
  put_integer(bytes, kChecksumOffset, 8, checksum);
}

fast_vector::HnswIndex make_index(const fast_vector::HnswNeighborSelection selection) {
  fast_vector::HnswIndex index({
      .dimension = 3,
      .max_connections = 4,
      .ef_construction = 16,
      .ef_search = 20,
      .random_seed = 17,
      .neighbor_selection = selection,
  });
  for (fast_vector::VectorId id = 1; id <= 100; ++id) {
    index.add(id,
              std::vector<float>{static_cast<float>(id % 23 + 1), static_cast<float>(id % 17 + 2),
                                 static_cast<float>(id % 11 + 3)});
  }
  return index;
}

void expect_rejected(const fast_vector::HnswIndex& index,
                     const std::function<void(std::vector<std::uint8_t>&)>& mutate) {
  TemporaryIndexFile file;
  fast_vector::save_hnsw_index(index, file.path());
  auto bytes = read_bytes(file.path());
  mutate(bytes);
  update_checksum(bytes);
  write_bytes(file.path(), bytes);
  EXPECT_THROW((void)fast_vector::load_hnsw_index(file.path()), std::runtime_error);
}

struct EdgeLocation {
  std::size_t node;
  std::size_t layer;
  std::size_t degree_offset;
  std::size_t neighbor_offset;
  std::uint32_t degree;
};

std::vector<EdgeLocation> locate_edges(const std::vector<std::uint8_t>& bytes) {
  const std::size_t count = get_integer(bytes, 32, 8);
  const std::size_t dimension = get_integer(bytes, 24, 8);
  std::size_t offset = kHeaderSize + count * 8 + count * dimension * 4;
  std::vector<EdgeLocation> locations;
  for (std::size_t node = 0; node < count; ++node) {
    const std::uint32_t level = static_cast<std::uint32_t>(get_integer(bytes, offset, 4));
    offset += 4;
    for (std::size_t layer = 0; layer <= level; ++layer) {
      const std::size_t degree_offset = offset;
      const auto degree = static_cast<std::uint32_t>(get_integer(bytes, offset, 4));
      offset += 4;
      locations.push_back({node, layer, degree_offset, offset, degree});
      offset += static_cast<std::size_t>(degree) * 4;
    }
  }
  return locations;
}

std::size_t graph_end(const std::vector<std::uint8_t>& bytes) {
  const auto edges = locate_edges(bytes);
  const auto& last = edges.back();
  return last.neighbor_offset + static_cast<std::size_t>(last.degree) * 4;
}

TEST(HnswIndexIoTest, RoundTripPreservesQueriesConfigurationAndGraphStatistics) {
  for (const auto selection : {fast_vector::HnswNeighborSelection::Simple,
                               fast_vector::HnswNeighborSelection::Heuristic}) {
    TemporaryIndexFile file;
    const auto original = make_index(selection);
    fast_vector::save_hnsw_index(original, file.path());
    const auto loaded = fast_vector::load_hnsw_index(file.path());
    EXPECT_EQ(loaded.size(), original.size());
    EXPECT_EQ(loaded.dimension(), original.dimension());
    EXPECT_EQ(loaded.config().max_connections, original.config().max_connections);
    EXPECT_EQ(loaded.config().ef_construction, original.config().ef_construction);
    EXPECT_EQ(loaded.config().ef_search, original.config().ef_search);
    EXPECT_EQ(loaded.config().random_seed, original.config().random_seed);
    EXPECT_EQ(loaded.config().neighbor_selection, selection);
    const auto before = original.stats();
    const auto after = loaded.stats();
    EXPECT_EQ(after.node_count, before.node_count);
    EXPECT_EQ(after.max_level, before.max_level);
    EXPECT_EQ(after.directed_edge_count, before.directed_edge_count);
    EXPECT_EQ(after.maximum_node_degree, before.maximum_node_degree);
    for (const std::vector<float>& query :
         {std::vector<float>{1.0F, 2.0F, 3.0F}, std::vector<float>{4.0F, 1.0F, 7.0F}}) {
      EXPECT_EQ(loaded.search(query, 10), original.search(query, 10));
    }
  }
}

TEST(HnswIndexIoTest, RoundTripSupportsEmptyIndex) {
  TemporaryIndexFile file;
  fast_vector::HnswIndex empty({.dimension = 7});
  fast_vector::save_hnsw_index(empty, file.path());
  auto loaded = fast_vector::load_hnsw_index(file.path());
  EXPECT_EQ(loaded.size(), 0U);
  EXPECT_EQ(loaded.dimension(), 7U);
  loaded.add(1, std::vector<float>{1, 0, 0, 0, 0, 0, 0});
  EXPECT_EQ(loaded.size(), 1U);
}

TEST(HnswIndexIoTest, RoundTripSupportsOneNode) {
  TemporaryIndexFile file;
  fast_vector::HnswIndex original({.dimension = 2});
  original.add(77, std::vector<float>{3.0F, 4.0F});
  fast_vector::save_hnsw_index(original, file.path());
  const auto loaded = fast_vector::load_hnsw_index(file.path());
  EXPECT_EQ(loaded.search(std::vector<float>{3.0F, 4.0F}, 1),
            original.search(std::vector<float>{3.0F, 4.0F}, 1));
  EXPECT_EQ(loaded.stats().directed_edge_count, 0U);
}

TEST(HnswIndexIoTest, RestoredRngStateReproducesFutureInsertions) {
  for (const auto selection : {fast_vector::HnswNeighborSelection::Simple,
                               fast_vector::HnswNeighborSelection::Heuristic}) {
    TemporaryIndexFile file;
    auto original = make_index(selection);
    fast_vector::save_hnsw_index(original, file.path());
    auto loaded = fast_vector::load_hnsw_index(file.path());
    for (fast_vector::VectorId id = 101; id <= 140; ++id) {
      const std::vector<float> vector{static_cast<float>(id % 31 + 1),
                                      static_cast<float>(id % 19 + 2),
                                      static_cast<float>(id % 13 + 3)};
      original.add(id, vector);
      loaded.add(id, vector);
    }
    fast_vector::save_hnsw_index(original, file.path());
    const auto expected = read_bytes(file.path());
    fast_vector::save_hnsw_index(loaded, file.path());
    EXPECT_EQ(read_bytes(file.path()), expected);
  }
}

TEST(HnswIndexIoTest, LoadsLegacyVersionOneSnapshot) {
  TemporaryIndexFile file;
  const auto original = make_index(fast_vector::HnswNeighborSelection::Heuristic);
  fast_vector::save_hnsw_index(original, file.path());
  auto bytes = read_bytes(file.path());
  bytes.resize(graph_end(bytes));
  put_integer(bytes, 8, 4, 1);
  put_integer(bytes, 96, 8, bytes.size() - kHeaderSize);
  update_checksum(bytes);
  write_bytes(file.path(), bytes);
  const auto loaded = fast_vector::load_hnsw_index(file.path());
  EXPECT_EQ(loaded.stats().directed_edge_count, original.stats().directed_edge_count);
  EXPECT_EQ(loaded.search(std::vector<float>{1, 2, 3}, 10),
            original.search(std::vector<float>{1, 2, 3}, 10));
}

TEST(HnswIndexIoTest, RejectsCorruptRngStateWithMatchingChecksum) {
  const auto index = make_index(fast_vector::HnswNeighborSelection::Simple);
  TemporaryIndexFile file;
  fast_vector::save_hnsw_index(index, file.path());
  const auto offset = graph_end(read_bytes(file.path()));
  expect_rejected(index, [=](auto& bytes) { put_integer(bytes, offset, 4, 0); });
  expect_rejected(index, [=](auto& bytes) { put_integer(bytes, offset, 4, 16385); });
  expect_rejected(index, [=](auto& bytes) { bytes.at(offset + 4) = 'x'; });
  expect_rejected(index, [=](auto& bytes) { bytes.back() = 'x'; });
}

TEST(HnswIndexIoTest, RejectsCorruptHeaderAndPayloadFields) {
  const auto index = make_index(fast_vector::HnswNeighborSelection::Heuristic);
  for (const auto& [offset, width, value] :
       {std::tuple<std::size_t, std::size_t, std::uint64_t>{0, 1, 0},
        {8, 4, 3},
        {12, 4, 1},
        {16, 4, 0},
        {20, 4, 0},
        {24, 8, 0},
        {24, 8, std::numeric_limits<std::uint64_t>::max()},
        {32, 8, std::numeric_limits<std::uint64_t>::max()},
        {40, 8, 1},
        {48, 8, 1},
        {56, 8, 0},
        {72, 4, 2},
        {76, 4, 1},
        {80, 8, 64},
        {88, 8, 100},
        {96, 8, 1}}) {
    expect_rejected(index, [=](auto& bytes) { put_integer(bytes, offset, width, value); });
  }
  expect_rejected(index, [](auto& bytes) { put_integer(bytes, kHeaderSize, 8, 2); });
  expect_rejected(index,
                  [](auto& bytes) { put_integer(bytes, kHeaderSize + 100 * 8, 4, 0x7FC00000U); });
  expect_rejected(index, [](auto& bytes) {
    put_integer(bytes, kHeaderSize + 100 * 8, 4, std::bit_cast<std::uint32_t>(0.25F));
  });
}

TEST(HnswIndexIoTest, RejectsCorruptGraphFieldsEvenWithMatchingChecksum) {
  const auto index = make_index(fast_vector::HnswNeighborSelection::Simple);
  TemporaryIndexFile file;
  fast_vector::save_hnsw_index(index, file.path());
  const auto bytes = read_bytes(file.path());
  const auto edges = locate_edges(bytes);
  const auto first_edge = std::find_if(edges.begin(), edges.end(),
                                       [](const EdgeLocation& edge) { return edge.degree >= 2; });
  ASSERT_NE(first_edge, edges.end());
  const std::size_t node_level_offset = kHeaderSize + 100 * 8 + 100 * 3 * 4;
  expect_rejected(index, [=](auto& modified) { put_integer(modified, node_level_offset, 4, 64); });
  expect_rejected(index,
                  [=](auto& modified) { put_integer(modified, first_edge->degree_offset, 4, 9); });
  expect_rejected(
      index, [=](auto& modified) { put_integer(modified, first_edge->neighbor_offset, 4, 100); });
  expect_rejected(index, [=](auto& modified) {
    put_integer(modified, first_edge->neighbor_offset, 4, first_edge->node);
  });
  expect_rejected(index, [=](auto& modified) {
    const std::uint64_t neighbor = get_integer(modified, first_edge->neighbor_offset, 4);
    put_integer(modified, first_edge->neighbor_offset + 4, 4, neighbor);
  });

  auto upper_edge = std::find_if(edges.begin(), edges.end(), [](const EdgeLocation& edge) {
    return edge.layer > 0 && edge.degree > 0;
  });
  ASSERT_NE(upper_edge, edges.end());
  std::vector<std::uint32_t> levels(100, 0);
  for (const auto& edge : edges) {
    levels[edge.node] = std::max(levels[edge.node], static_cast<std::uint32_t>(edge.layer));
  }
  const auto low_node = std::find_if(levels.begin(), levels.end(), [&](const std::uint32_t level) {
    return level < upper_edge->layer;
  });
  ASSERT_NE(low_node, levels.end());
  expect_rejected(index, [](auto& modified) { put_integer(modified, 80, 8, 0); });
  expect_rejected(index, [=](auto& modified) {
    put_integer(modified, 88, 8, static_cast<std::size_t>(low_node - levels.begin()));
  });
  expect_rejected(index, [=](auto& modified) {
    put_integer(modified, upper_edge->neighbor_offset, 4,
                static_cast<std::size_t>(low_node - levels.begin()));
  });
}

TEST(HnswIndexIoTest, RejectsTruncationTrailingBytesAndChecksumMismatch) {
  const auto index = make_index(fast_vector::HnswNeighborSelection::Simple);
  TemporaryIndexFile file;
  fast_vector::save_hnsw_index(index, file.path());
  const auto original = read_bytes(file.path());
  auto modified = original;
  modified.pop_back();
  write_bytes(file.path(), modified);
  EXPECT_THROW((void)fast_vector::load_hnsw_index(file.path()), std::runtime_error);
  modified = original;
  modified.push_back(0);
  write_bytes(file.path(), modified);
  EXPECT_THROW((void)fast_vector::load_hnsw_index(file.path()), std::runtime_error);
  modified = original;
  modified[64] ^= 1U;
  write_bytes(file.path(), modified);
  EXPECT_THROW((void)fast_vector::load_hnsw_index(file.path()), std::runtime_error);
  modified = original;
  modified[kHeaderSize] ^= 1U;
  write_bytes(file.path(), modified);
  EXPECT_THROW((void)fast_vector::load_hnsw_index(file.path()), std::runtime_error);
}

#if defined(__linux__)
TEST(HnswIndexIoTest, PublishesWithoutOverwriteAndAtomicallyReplacesOnRequest) {
  TemporaryIndexFile file;
  const auto original = make_index(fast_vector::HnswNeighborSelection::Simple);
  fast_vector::publish_hnsw_index(original, file.path());
  const auto initial_bytes = read_bytes(file.path());
  EXPECT_EQ(fast_vector::load_hnsw_index(file.path()).size(), original.size());

  auto replacement = make_index(fast_vector::HnswNeighborSelection::Heuristic);
  replacement.add(101, std::vector<float>{1, 2, 3});
  EXPECT_THROW(fast_vector::publish_hnsw_index(replacement, file.path()), std::system_error);
  EXPECT_EQ(read_bytes(file.path()), initial_bytes);

  fast_vector::publish_hnsw_index(replacement, file.path(), true);
  const auto restored = fast_vector::load_hnsw_index(file.path());
  EXPECT_EQ(restored.size(), replacement.size());
  EXPECT_EQ(restored.stats().directed_edge_count, replacement.stats().directed_edge_count);
  EXPECT_NE(read_bytes(file.path()), initial_bytes);

  const std::string prefix = file.path().filename().string() + ".tmp-";
  for (const auto& entry : std::filesystem::directory_iterator(file.path().parent_path())) {
    EXPECT_FALSE(entry.path().filename().string().starts_with(prefix));
  }
}

TEST(HnswIndexIoTest, FailedPublishDoesNotCreateDestination) {
  TemporaryIndexFile file;
  const auto index = make_index(fast_vector::HnswNeighborSelection::Simple);
  const auto missing_parent = file.path() / "missing";
  EXPECT_THROW(fast_vector::publish_hnsw_index(index, missing_parent), std::system_error);
  EXPECT_FALSE(std::filesystem::exists(file.path()));
}
#endif

}  // namespace
