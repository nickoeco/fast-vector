#include <charconv>
#include <chrono>
#include <cstddef>
#include <cstdint>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <limits>
#include <optional>
#include <random>
#include <stdexcept>
#include <string>
#include <string_view>
#include <vector>

#if defined(__linux__)
#include <unistd.h>
#endif

#include "fast_vector/hnsw_index_io.h"

namespace {

using Clock = std::chrono::steady_clock;

struct Options {
  std::size_t vector_count = 1'000;
  std::size_t dimension = 128;
  std::size_t max_connections = 16;
  std::size_t ef_construction = 100;
  std::size_t ef_search = 50;
  std::uint32_t data_seed = 20250908U;
  std::uint64_t graph_seed = 42;
};

std::uint64_t parse_unsigned(const std::string_view value, const std::string& option) {
  std::uint64_t result = 0;
  const auto [end, error] = std::from_chars(value.data(), value.data() + value.size(), result);
  if (error != std::errc{} || end != value.data() + value.size()) {
    throw std::invalid_argument("invalid value for " + option);
  }
  return result;
}

std::size_t parse_positive_size(const std::string_view value, const std::string& option) {
  const std::uint64_t result = parse_unsigned(value, option);
  if (result == 0 || result > std::numeric_limits<std::size_t>::max()) {
    throw std::invalid_argument(option + " must be a positive platform-sized integer");
  }
  return static_cast<std::size_t>(result);
}

Options parse_arguments(const int argc, char* argv[]) {
  Options options;
  for (int i = 1; i < argc; i += 2) {
    if (i + 1 >= argc) {
      throw std::invalid_argument("every benchmark option requires a value");
    }
    const std::string option = argv[i];
    const std::string_view value = argv[i + 1];
    if (option == "--vectors") {
      options.vector_count = parse_positive_size(value, option);
    } else if (option == "--dimension") {
      options.dimension = parse_positive_size(value, option);
    } else if (option == "--m") {
      options.max_connections = parse_positive_size(value, option);
    } else if (option == "--ef-construction") {
      options.ef_construction = parse_positive_size(value, option);
    } else if (option == "--ef-search") {
      options.ef_search = parse_positive_size(value, option);
    } else if (option == "--data-seed") {
      const std::uint64_t seed = parse_unsigned(value, option);
      if (seed > std::numeric_limits<std::uint32_t>::max()) {
        throw std::invalid_argument("--data-seed must fit uint32_t");
      }
      options.data_seed = static_cast<std::uint32_t>(seed);
    } else if (option == "--graph-seed") {
      options.graph_seed = parse_unsigned(value, option);
    } else {
      throw std::invalid_argument("unknown option: " + option);
    }
  }
  return options;
}

std::optional<std::uint64_t> resident_set_size_bytes() {
#if defined(__linux__)
  std::ifstream statm("/proc/self/statm");
  std::uint64_t total_pages = 0;
  std::uint64_t resident_pages = 0;
  if (!(statm >> total_pages >> resident_pages)) {
    return std::nullopt;
  }
  const long page_size = sysconf(_SC_PAGESIZE);
  if (page_size <= 0) {
    return std::nullopt;
  }
  return resident_pages * static_cast<std::uint64_t>(page_size);
#else
  return std::nullopt;
#endif
}

std::string format_bytes(const std::optional<std::uint64_t> bytes) {
  return bytes ? std::to_string(*bytes) : "unavailable";
}

class TemporarySnapshot {
 public:
  TemporarySnapshot() {
    const auto stamp = Clock::now().time_since_epoch().count();
    path_ = std::filesystem::temp_directory_path() /
            ("fast-vector-hnsw-benchmark-" + std::to_string(stamp) + ".fvh");
  }
  ~TemporarySnapshot() {
    std::error_code error;
    std::filesystem::remove(path_, error);
  }
  TemporarySnapshot(const TemporarySnapshot&) = delete;
  TemporarySnapshot& operator=(const TemporarySnapshot&) = delete;
  [[nodiscard]] const std::filesystem::path& path() const noexcept { return path_; }

 private:
  std::filesystem::path path_;
};

}  // namespace

int main(const int argc, char* argv[]) {
  try {
    const Options options = parse_arguments(argc, argv);
    fast_vector::HnswIndex original({
        .dimension = options.dimension,
        .max_connections = options.max_connections,
        .ef_construction = options.ef_construction,
        .ef_search = options.ef_search,
        .random_seed = options.graph_seed,
        .neighbor_selection = fast_vector::HnswNeighborSelection::Heuristic,
    });
    std::mt19937 generator(options.data_seed);
    std::uniform_real_distribution<float> distribution(-1.0F, 1.0F);
    std::vector<float> values(options.dimension);
    const auto build_start = Clock::now();
    for (std::size_t node = 0; node < options.vector_count; ++node) {
      for (float& value : values) {
        value = distribution(generator);
      }
      original.add(static_cast<fast_vector::VectorId>(node + 1), values);
    }
    const auto build_end = Clock::now();

    TemporarySnapshot snapshot;
    const auto save_start = Clock::now();
    fast_vector::save_hnsw_index(original, snapshot.path());
    const auto save_end = Clock::now();
    const auto rss_before_load = resident_set_size_bytes();
    const auto load_start = Clock::now();
    fast_vector::HnswIndex loaded = fast_vector::load_hnsw_index(snapshot.path());
    const auto load_end = Clock::now();
    const auto rss_after_load = resident_set_size_bytes();

    std::vector<float> query(options.dimension, 0.0F);
    query.front() = 1.0F;
    if (original.search(query, 10) != loaded.search(query, 10)) {
      throw std::runtime_error("loaded index returned different search results");
    }
    const auto before = original.stats();
    const auto after = loaded.stats();
    if (before.node_count != after.node_count || before.max_level != after.max_level ||
        before.directed_edge_count != after.directed_edge_count ||
        before.maximum_node_degree != after.maximum_node_degree) {
      throw std::runtime_error("loaded graph statistics differ");
    }

#if defined(NDEBUG)
    constexpr std::string_view kBuildMode = "Release-like (NDEBUG defined)";
#else
    constexpr std::string_view kBuildMode = "Debug-like (NDEBUG not defined)";
#endif
    std::optional<std::uint64_t> approximate_load_rss_delta;
    if (rss_before_load && rss_after_load && *rss_after_load >= *rss_before_load) {
      approximate_load_rss_delta = *rss_after_load - *rss_before_load;
    }
    std::cout << std::fixed << std::setprecision(3) << "Build mode: " << kBuildMode << '\n'
              << "Vector count: " << options.vector_count << '\n'
              << "Dimension: " << options.dimension << '\n'
              << "M: " << options.max_connections << '\n'
              << "efConstruction: " << options.ef_construction << '\n'
              << "efSearch: " << options.ef_search << '\n'
              << "Data seed: " << options.data_seed << '\n'
              << "Graph seed: " << options.graph_seed << '\n'
              << "HNSW build time (ms): "
              << std::chrono::duration<double, std::milli>(build_end - build_start).count() << '\n'
              << "Snapshot save time (ms): "
              << std::chrono::duration<double, std::milli>(save_end - save_start).count() << '\n'
              << "Snapshot load time (ms): "
              << std::chrono::duration<double, std::milli>(load_end - load_start).count() << '\n'
              << "Snapshot file size (bytes): " << std::filesystem::file_size(snapshot.path())
              << '\n'
              << "RSS before load (bytes): " << format_bytes(rss_before_load) << '\n'
              << "RSS after load (bytes): " << format_bytes(rss_after_load) << '\n'
              << "Approximate load RSS delta (bytes): " << format_bytes(approximate_load_rss_delta)
              << '\n'
              << "Graph directed edges: " << after.directed_edge_count << '\n'
              << "Round-trip query and graph stats: match\n";
  } catch (const std::exception& error) {
    std::cerr << "HNSW persistence benchmark failed: " << error.what() << '\n';
    return 1;
  }
  return 0;
}
