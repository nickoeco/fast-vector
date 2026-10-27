#include <charconv>
#include <chrono>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <optional>
#include <stdexcept>
#include <string>
#include <string_view>

#if defined(__linux__)
#include <sys/resource.h>
#include <unistd.h>
#endif

#include "fast_vector/hnsw_index_io.h"

namespace {
using Clock = std::chrono::steady_clock;

struct Options {
  std::filesystem::path input;
  std::filesystem::path output;
  std::optional<std::uint64_t> seed;
  bool overwrite = false;
};

Options parse_arguments(const int argc, char* argv[]) {
  Options options;
  for (int i = 1; i < argc; i += 2) {
    if (i + 1 >= argc) {
      throw std::invalid_argument("every rebuild option requires a value");
    }
    const std::string_view key = argv[i];
    const std::string_view value = argv[i + 1];
    if (key == "--input") {
      options.input = value;
    } else if (key == "--output") {
      options.output = value;
    } else if (key == "--seed") {
      std::uint64_t seed = 0;
      const auto [end, error] = std::from_chars(value.data(), value.data() + value.size(), seed);
      if (error != std::errc{} || end != value.data() + value.size()) {
        throw std::invalid_argument("invalid rebuild seed");
      }
      options.seed = seed;
    } else if (key == "--overwrite" && (value == "true" || value == "false")) {
      options.overwrite = value == "true";
    } else {
      throw std::invalid_argument("unknown rebuild option or invalid value: " + std::string(key));
    }
  }
  if (options.input.empty() || options.output.empty()) {
    throw std::invalid_argument("--input and --output are required");
  }
  if (std::filesystem::weakly_canonical(options.input) ==
          std::filesystem::weakly_canonical(options.output) ||
      (std::filesystem::exists(options.output) &&
       std::filesystem::equivalent(options.input, options.output))) {
    throw std::invalid_argument("rebuild output must be different from the input snapshot");
  }
  return options;
}

std::string rss_bytes() {
#if defined(__linux__)
  std::ifstream input("/proc/self/statm");
  std::uint64_t total = 0;
  std::uint64_t resident = 0;
  const auto page_size = sysconf(_SC_PAGESIZE);
  if (input >> total >> resident && page_size > 0) {
    return std::to_string(resident * static_cast<std::uint64_t>(page_size));
  }
#endif
  return "unavailable";
}

std::string peak_rss_bytes() {
#if defined(__linux__)
  struct rusage usage {};
  if (getrusage(RUSAGE_SELF, &usage) == 0) {
    return std::to_string(static_cast<std::uint64_t>(usage.ru_maxrss) * 1024);
  }
#endif
  return "unavailable";
}

double milliseconds(const Clock::time_point start, const Clock::time_point end) {
  return std::chrono::duration<double, std::milli>(end - start).count();
}
}  // namespace

int main(const int argc, char* argv[]) {
  try {
    const auto options = parse_arguments(argc, argv);
    const auto load_start = Clock::now();
    const auto source = fast_vector::load_hnsw_index(options.input);
    const auto load_end = Clock::now();
    const auto loaded_rss = rss_bytes();
    const auto seed = options.seed.value_or(source.config().random_seed);
    const auto rebuild_start = Clock::now();
    const auto rebuilt = source.rebuild(seed);
    const auto rebuild_end = Clock::now();
    const auto rebuilt_rss = rss_bytes();
    const auto before = source.stats();
    const auto after = rebuilt.stats();
    if (source.live_ids() != rebuilt.live_ids() || after.deleted_count != 0 ||
        after.node_count != before.live_count) {
      throw std::runtime_error("rebuilt index failed live-ID/count verification");
    }
    const auto publish_start = Clock::now();
    fast_vector::publish_hnsw_index(rebuilt, options.output, options.overwrite);
    const auto publish_end = Clock::now();
    std::cout << std::fixed << std::setprecision(3) << "Input physical nodes: " << before.node_count
              << '\n'
              << "Input live vectors: " << before.live_count << '\n'
              << "Input tombstones: " << before.deleted_count << '\n'
              << "Output physical nodes: " << after.node_count << '\n'
              << "Output live vectors: " << after.live_count << '\n'
              << "Output tombstones: " << after.deleted_count << '\n'
              << "Rebuild seed: " << seed << '\n'
              << "Load time (ms): " << milliseconds(load_start, load_end) << '\n'
              << "Rebuild time (ms): " << milliseconds(rebuild_start, rebuild_end) << '\n'
              << "Validated publication time (ms): " << milliseconds(publish_start, publish_end)
              << '\n'
              << "Input file size (bytes): " << std::filesystem::file_size(options.input) << '\n'
              << "Output file size (bytes): " << std::filesystem::file_size(options.output) << '\n'
              << "RSS after load (bytes): " << loaded_rss << '\n'
              << "RSS with source and rebuilt indexes (bytes): " << rebuilt_rss << '\n'
              << "RSS after publication validation (bytes): " << rss_bytes() << '\n'
              << "Peak process RSS (bytes): " << peak_rss_bytes() << '\n'
              << "Live-ID verification: match\n";
  } catch (const std::exception& error) {
    std::cerr << "HNSW rebuild failed: " << error.what() << '\n'
              << "Usage: fast_vector_rebuild_hnsw --input SOURCE --output DESTINATION "
                 "[--seed UINT64] [--overwrite true|false]\n";
    return 1;
  }
  return 0;
}
