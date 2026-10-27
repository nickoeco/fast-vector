#include <algorithm>
#include <charconv>
#include <chrono>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <limits>
#include <numeric>
#include <optional>
#include <random>
#include <stdexcept>
#include <string>
#include <string_view>
#include <thread>
#include <unordered_set>
#include <vector>

#if defined(__linux__)
#include <unistd.h>
#endif

#include "fast_vector/flat_index.h"
#include "fast_vector/hnsw_index.h"
#include "fast_vector/hnsw_index_io.h"

namespace {

using Clock = std::chrono::steady_clock;

struct BenchmarkConfig {
  std::size_t vector_count = 10'000;
  std::size_t dimension = 128;
  std::size_t query_count = 1'000;
  std::size_t top_k = 10;
  std::size_t warmup_count = 20;
  std::size_t max_connections = 16;
  std::size_t ef_construction = 200;
  std::vector<std::size_t> ef_search_values{10, 50, 100, 200};
  std::uint32_t data_seed = 20250908U;
  std::uint64_t hnsw_seed = 42;
  std::size_t delete_percent = 0;
  bool rebuild = false;
  fast_vector::DotProductKernel kernel = fast_vector::DotProductKernel::Scalar;
  fast_vector::HnswNeighborSelection neighbor_selection =
      fast_vector::HnswNeighborSelection::Heuristic;
};

struct QueryMetrics {
  double average_us;
  double p50_us;
  double p95_us;
  double p99_us;
  double qps;
  double mean_recall;
  double minimum_recall;
  std::size_t short_queries;
};

std::uint64_t parse_unsigned(const std::string_view text, const std::string& option) {
  std::uint64_t value = 0;
  const auto [end, error] = std::from_chars(text.data(), text.data() + text.size(), value);
  if (error != std::errc{} || end != text.data() + text.size()) {
    throw std::invalid_argument("invalid value for " + option);
  }
  return value;
}

std::size_t parse_positive_size(const std::string_view text, const std::string& option) {
  const std::uint64_t value = parse_unsigned(text, option);
  if (value == 0 || value > std::numeric_limits<std::size_t>::max()) {
    throw std::invalid_argument(option + " must be a positive platform-sized integer");
  }
  return static_cast<std::size_t>(value);
}

std::vector<std::size_t> parse_ef_values(const std::string_view text) {
  std::vector<std::size_t> values;
  std::size_t begin = 0;
  while (begin <= text.size()) {
    const std::size_t comma = text.find(',', begin);
    const std::size_t length =
        comma == std::string_view::npos ? text.size() - begin : comma - begin;
    values.push_back(parse_positive_size(text.substr(begin, length), "--ef-search"));
    if (comma == std::string_view::npos) {
      break;
    }
    begin = comma + 1;
  }
  std::sort(values.begin(), values.end());
  values.erase(std::unique(values.begin(), values.end()), values.end());
  return values;
}

BenchmarkConfig parse_arguments(const int argc, char* argv[]) {
  BenchmarkConfig config;
  for (int i = 1; i < argc; i += 2) {
    if (i + 1 >= argc) {
      throw std::invalid_argument("every benchmark option requires a value");
    }
    const std::string option = argv[i];
    const std::string_view value = argv[i + 1];
    if (option == "--vectors") {
      config.vector_count = parse_positive_size(value, option);
    } else if (option == "--dimension") {
      config.dimension = parse_positive_size(value, option);
    } else if (option == "--queries") {
      config.query_count = parse_positive_size(value, option);
    } else if (option == "--k") {
      config.top_k = parse_positive_size(value, option);
    } else if (option == "--warmup") {
      config.warmup_count = parse_positive_size(value, option);
    } else if (option == "--m") {
      config.max_connections = parse_positive_size(value, option);
    } else if (option == "--ef-construction") {
      config.ef_construction = parse_positive_size(value, option);
    } else if (option == "--ef-search") {
      config.ef_search_values = parse_ef_values(value);
    } else if (option == "--seed") {
      const std::uint64_t seed = parse_unsigned(value, option);
      if (seed > std::numeric_limits<std::uint32_t>::max()) {
        throw std::invalid_argument("--seed must fit in uint32_t");
      }
      config.data_seed = static_cast<std::uint32_t>(seed);
    } else if (option == "--hnsw-seed") {
      config.hnsw_seed = parse_unsigned(value, option);
    } else if (option == "--delete-percent") {
      const auto percent = parse_unsigned(value, option);
      if (percent > 99) {
        throw std::invalid_argument("--delete-percent must be between 0 and 99");
      }
      config.delete_percent = static_cast<std::size_t>(percent);
    } else if (option == "--rebuild" && (value == "true" || value == "false")) {
      config.rebuild = value == "true";
    } else if (option == "--kernel") {
      if (value == "scalar") {
        config.kernel = fast_vector::DotProductKernel::Scalar;
      } else if (value == "auto") {
        config.kernel = fast_vector::DotProductKernel::AutoVectorized;
      } else if (value == "avx2") {
        config.kernel = fast_vector::DotProductKernel::Avx2;
      } else {
        throw std::invalid_argument("--kernel must be scalar, auto, or avx2");
      }
    } else if (option == "--neighbor-selection") {
      if (value == "simple") {
        config.neighbor_selection = fast_vector::HnswNeighborSelection::Simple;
      } else if (value == "heuristic") {
        config.neighbor_selection = fast_vector::HnswNeighborSelection::Heuristic;
      } else {
        throw std::invalid_argument("--neighbor-selection must be simple or heuristic");
      }
    } else {
      throw std::invalid_argument("unknown benchmark option: " + option);
    }
  }
  return config;
}

std::vector<float> random_vector(std::mt19937& generator, const std::size_t dimension) {
  std::uniform_real_distribution<float> distribution(-1.0F, 1.0F);
  std::vector<float> values(dimension);
  for (float& value : values) {
    value = distribution(generator);
  }
  return values;
}

double percentile(const std::vector<double>& sorted_values, const double fraction) {
  const auto rank =
      static_cast<std::size_t>(std::ceil(fraction * static_cast<double>(sorted_values.size())));
  return sorted_values[std::max<std::size_t>(1, rank) - 1];
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

std::uint64_t non_negative_difference(const std::optional<std::uint64_t> after,
                                      const std::optional<std::uint64_t> before) {
  if (!after || !before || *after < *before) {
    return 0;
  }
  return *after - *before;
}

double recall(const std::vector<fast_vector::SearchResult>& truth,
              const std::vector<fast_vector::SearchResult>& approximate) {
  std::unordered_set<fast_vector::VectorId> truth_ids;
  truth_ids.reserve(truth.size());
  for (const auto& result : truth) {
    truth_ids.insert(result.id);
  }
  std::size_t matches = 0;
  for (const auto& result : approximate) {
    matches += truth_ids.contains(result.id) ? 1U : 0U;
  }
  return truth.empty() ? 1.0 : static_cast<double>(matches) / static_cast<double>(truth.size());
}

template <typename SearchFunction>
QueryMetrics measure_queries(const std::vector<std::vector<float>>& queries,
                             const std::vector<std::vector<fast_vector::SearchResult>>& truth,
                             const std::unordered_set<fast_vector::VectorId>& deleted_ids,
                             const std::size_t warmup_count, SearchFunction&& search,
                             double& checksum) {
  for (std::size_t i = 0; i < warmup_count; ++i) {
    const auto results = search(queries[i % queries.size()]);
    if (!results.empty()) {
      checksum += results.front().score;
    }
  }

  std::vector<double> latencies_us;
  std::vector<double> recalls;
  latencies_us.reserve(queries.size());
  recalls.reserve(queries.size());
  std::size_t short_queries = 0;
  for (std::size_t i = 0; i < queries.size(); ++i) {
    const auto start = Clock::now();
    const auto results = search(queries[i]);
    const auto end = Clock::now();
    if (!results.empty()) {
      checksum += results.front().score;
    }
    // Validate outside the timed region so integrity checks do not inflate query latency.
    std::unordered_set<fast_vector::VectorId> returned_ids;
    for (const auto& result : results) {
      if (deleted_ids.contains(result.id) || !returned_ids.insert(result.id).second) {
        throw std::runtime_error("HNSW returned a deleted or duplicate ID");
      }
    }
    short_queries += results.size() < truth[i].size() ? 1U : 0U;
    latencies_us.push_back(std::chrono::duration<double, std::micro>(end - start).count());
    recalls.push_back(recall(truth[i], results));
  }
  const double total_us = std::accumulate(latencies_us.begin(), latencies_us.end(), 0.0);
  const double average = total_us / static_cast<double>(latencies_us.size());
  const double mean_recall =
      std::accumulate(recalls.begin(), recalls.end(), 0.0) / static_cast<double>(recalls.size());
  const double minimum_recall = *std::min_element(recalls.begin(), recalls.end());
  std::sort(latencies_us.begin(), latencies_us.end());
  return QueryMetrics{average,
                      percentile(latencies_us, 0.50),
                      percentile(latencies_us, 0.95),
                      percentile(latencies_us, 0.99),
                      static_cast<double>(queries.size()) * 1'000'000.0 / total_us,
                      mean_recall,
                      minimum_recall,
                      short_queries};
}

void print_usage() {
  std::cerr << "Usage: fast_vector_hnsw_benchmark [--vectors N] [--dimension D] "
               "[--queries Q] [--k K] [--warmup W] [--m M] "
               "[--ef-construction N] [--ef-search E1,E2,...] [--seed S] "
               "[--hnsw-seed S] [--kernel scalar|auto|avx2] "
               "[--neighbor-selection simple|heuristic] [--delete-percent 0..99] "
               "[--rebuild true|false]\n";
}

class TemporarySnapshot {
 public:
  explicit TemporarySnapshot(const std::string& tag)
      : path_(std::filesystem::temp_directory_path() /
              ("fast-vector-rebuild-" + tag + "-" +
               std::to_string(Clock::now().time_since_epoch().count()) + ".fvh")) {}
  ~TemporarySnapshot() {
    std::error_code error;
    std::filesystem::remove(path_, error);
  }
  const std::filesystem::path& path() const noexcept { return path_; }

 private:
  std::filesystem::path path_;
};

}  // namespace

int main(const int argc, char* argv[]) {
  try {
    const BenchmarkConfig config = parse_arguments(argc, argv);
    if (config.kernel == fast_vector::DotProductKernel::Avx2 &&
        !fast_vector::avx2_dot_product_available()) {
      throw std::invalid_argument("--kernel avx2 is unavailable on this build or CPU");
    }

    std::mt19937 generator(config.data_seed);
    std::vector<std::vector<float>> database;
    std::vector<std::vector<float>> queries;
    database.reserve(config.vector_count);
    queries.reserve(config.query_count);
    for (std::size_t i = 0; i < config.vector_count; ++i) {
      database.push_back(random_vector(generator, config.dimension));
    }
    for (std::size_t i = 0; i < config.query_count; ++i) {
      queries.push_back(random_vector(generator, config.dimension));
    }

    const auto rss_before_indexes = resident_set_size_bytes();
    fast_vector::FlatIndex flat(config.dimension, config.kernel);
    const auto flat_build_start = Clock::now();
    for (std::size_t i = 0; i < database.size(); ++i) {
      flat.add(static_cast<fast_vector::VectorId>(i), database[i]);
    }
    const auto flat_build_end = Clock::now();
    const auto rss_after_flat = resident_set_size_bytes();

    fast_vector::HnswIndex hnsw(fast_vector::HnswConfig{
        .dimension = config.dimension,
        .max_connections = config.max_connections,
        .ef_construction = config.ef_construction,
        .ef_search = config.ef_search_values.front(),
        .random_seed = config.hnsw_seed,
        .kernel = config.kernel,
        .neighbor_selection = config.neighbor_selection,
    });
    const auto hnsw_build_start = Clock::now();
    for (std::size_t i = 0; i < database.size(); ++i) {
      hnsw.add(static_cast<fast_vector::VectorId>(i), database[i]);
    }
    const auto hnsw_build_end = Clock::now();
    const auto rss_after_hnsw = resident_set_size_bytes();

    // A separate RNG keeps deletion sets nested across ratios without changing data or queries.
    std::vector<fast_vector::VectorId> deletion_order(config.vector_count);
    std::iota(deletion_order.begin(), deletion_order.end(), fast_vector::VectorId{0});
    std::mt19937 deletion_generator(config.data_seed);
    std::shuffle(deletion_order.begin(), deletion_order.end(), deletion_generator);
    const auto delete_count = config.vector_count / 100 * config.delete_percent +
                              config.vector_count % 100 * config.delete_percent / 100;
    std::unordered_set<fast_vector::VectorId> deleted_ids;
    for (std::size_t i = 0; i < delete_count; ++i) {
      const auto id = deletion_order[i];
      if (!flat.erase(id)) {
        throw std::runtime_error("Flat deletion failed");
      }
      deleted_ids.insert(id);
    }
    const auto delete_start = Clock::now();
    for (std::size_t i = 0; i < delete_count; ++i) {
      if (!hnsw.erase(deletion_order[i])) {
        throw std::runtime_error("HNSW deletion failed");
      }
    }
    const auto delete_end = Clock::now();
    const auto rss_after_deletion = resident_set_size_bytes();

    double checksum = 0.0;
    std::vector<std::vector<fast_vector::SearchResult>> truth;
    truth.reserve(queries.size());
    for (std::size_t i = 0; i < config.warmup_count; ++i) {
      checksum += flat.search(queries[i % queries.size()], config.top_k).front().score;
    }
    std::vector<double> flat_latencies;
    flat_latencies.reserve(queries.size());
    for (const auto& query : queries) {
      const auto start = Clock::now();
      truth.push_back(flat.search(query, config.top_k));
      const auto end = Clock::now();
      checksum += truth.back().front().score;
      flat_latencies.push_back(std::chrono::duration<double, std::micro>(end - start).count());
    }
    const double flat_total_us = std::accumulate(flat_latencies.begin(), flat_latencies.end(), 0.0);
    const double flat_average = flat_total_us / static_cast<double>(flat_latencies.size());
    std::sort(flat_latencies.begin(), flat_latencies.end());

    const fast_vector::HnswStats stats = hnsw.stats();
    const std::uint64_t vector_payload_bytes =
        static_cast<std::uint64_t>(config.vector_count) *
        (static_cast<std::uint64_t>(config.dimension) * sizeof(float) +
         sizeof(fast_vector::VectorId));
    const std::uint64_t graph_edge_payload_bytes =
        static_cast<std::uint64_t>(stats.directed_edge_count) * sizeof(std::uint32_t);

#if defined(NDEBUG)
    constexpr std::string_view kBuildMode = "Release-like (NDEBUG defined)";
#else
    constexpr std::string_view kBuildMode = "Debug-like (NDEBUG not defined)";
#endif

    std::cout
        << std::fixed << std::setprecision(3) << "Build mode: " << kBuildMode << '\n'
        << "Hardware concurrency: " << std::thread::hardware_concurrency() << '\n'
        << "Vector count: " << config.vector_count << '\n'
        << "Dimension: " << config.dimension << '\n'
        << "Query count: " << config.query_count << '\n'
        << "Top-K: " << config.top_k << '\n'
        << "Warmup queries: " << config.warmup_count << '\n'
        << "M: " << config.max_connections << '\n'
        << "efConstruction: " << config.ef_construction << '\n'
        << "Neighbor selection: "
        << (config.neighbor_selection == fast_vector::HnswNeighborSelection::Simple ? "simple"
                                                                                    : "heuristic")
        << '\n'
        << "Dot-product kernel: " << fast_vector::dot_product_kernel_name(config.kernel) << '\n'
        << "Data seed: " << config.data_seed << '\n'
        << "HNSW seed: " << config.hnsw_seed << '\n'
        << "Deletion percent: " << config.delete_percent << '\n'
        << "Deletion seed: " << config.data_seed << '\n'
        << "Live vectors: " << stats.live_count << '\n'
        << "Deleted vectors: " << stats.deleted_count << '\n'
        << "HNSW deletion time (ms): "
        << std::chrono::duration<double, std::milli>(delete_end - delete_start).count() << '\n'
        << "RSS after deletion (bytes): "
        << (rss_after_deletion ? std::to_string(*rss_after_deletion) : "unavailable") << '\n'
        << "Flat build time (ms): "
        << std::chrono::duration<double, std::milli>(flat_build_end - flat_build_start).count()
        << '\n'
        << "HNSW build time (ms): "
        << std::chrono::duration<double, std::milli>(hnsw_build_end - hnsw_build_start).count()
        << '\n'
        << "Approximate Flat RSS delta (bytes): "
        << non_negative_difference(rss_after_flat, rss_before_indexes) << '\n'
        << "Approximate HNSW RSS delta (bytes): "
        << non_negative_difference(rss_after_hnsw, rss_after_flat) << '\n'
        << "Normalized vector and ID payload (bytes): " << vector_payload_bytes << '\n'
        << "Graph edge payload (bytes): " << graph_edge_payload_bytes << '\n'
        << "Graph directed edges: " << stats.directed_edge_count << '\n'
        << "Graph maximum level: " << stats.max_level << '\n'
        << "Graph maximum node degree: " << stats.maximum_node_degree << '\n'
        << "Flat average latency (us): " << flat_average << '\n'
        << "Flat P50 latency (us): " << percentile(flat_latencies, 0.50) << '\n'
        << "Flat P95 latency (us): " << percentile(flat_latencies, 0.95) << '\n'
        << "Flat P99 latency (us): " << percentile(flat_latencies, 0.99) << '\n'
        << "Flat QPS: " << static_cast<double>(queries.size()) * 1'000'000.0 / flat_total_us << '\n'
        << "\nHNSW query sweep:\n"
        << "efSearch,mean_recall,min_recall,average_us,p50_us,p95_us,p99_us,qps,short_queries\n";

    for (const std::size_t ef_search : config.ef_search_values) {
      const QueryMetrics metrics = measure_queries(
          queries, truth, deleted_ids, config.warmup_count,
          [&hnsw, &config, ef_search](const std::vector<float>& query) {
            return hnsw.search(query, config.top_k, ef_search);
          },
          checksum);
      std::cout << ef_search << ',' << metrics.mean_recall << ',' << metrics.minimum_recall << ','
                << metrics.average_us << ',' << metrics.p50_us << ',' << metrics.p95_us << ','
                << metrics.p99_us << ',' << metrics.qps << ',' << metrics.short_queries << '\n';
    }
    if (config.rebuild) {
      const auto rss_before_rebuild = resident_set_size_bytes();
      const auto rebuild_start = Clock::now();
      const auto rebuilt = hnsw.rebuild(config.hnsw_seed);
      const auto rebuild_end = Clock::now();
      const auto rss_after_rebuild = resident_set_size_bytes();
      const auto rebuilt_stats = rebuilt.stats();
      if (rebuilt.live_ids() != hnsw.live_ids() || rebuilt.size() != flat.size() ||
          rebuilt_stats.node_count != flat.size() || rebuilt_stats.deleted_count != 0) {
        throw std::runtime_error("rebuild changed the live ID set or counts");
      }
      TemporarySnapshot source_file("source");
      TemporarySnapshot rebuilt_file("compact");
      fast_vector::save_hnsw_index(hnsw, source_file.path());
      fast_vector::save_hnsw_index(rebuilt, rebuilt_file.path());
      std::cout
          << "\nRebuild seed: " << config.hnsw_seed << '\n'
          << "Rebuild time (ms): "
          << std::chrono::duration<double, std::milli>(rebuild_end - rebuild_start).count() << '\n'
          << "Live-ID verification: match\n"
          << "Rebuilt physical nodes: " << rebuilt_stats.node_count << '\n'
          << "Rebuilt tombstones: " << rebuilt_stats.deleted_count << '\n'
          << "Source snapshot bytes: " << std::filesystem::file_size(source_file.path()) << '\n'
          << "Rebuilt snapshot bytes: " << std::filesystem::file_size(rebuilt_file.path()) << '\n'
          << "Source vector/ID payload bytes: " << vector_payload_bytes << '\n'
          << "Rebuilt vector/ID payload bytes: "
          << static_cast<std::uint64_t>(rebuilt_stats.node_count) *
                 (static_cast<std::uint64_t>(config.dimension) * sizeof(float) +
                  sizeof(fast_vector::VectorId))
          << '\n'
          << "RSS before rebuild (bytes): "
          << (rss_before_rebuild ? std::to_string(*rss_before_rebuild) : "unavailable") << '\n'
          << "RSS with both HNSW indexes (bytes): "
          << (rss_after_rebuild ? std::to_string(*rss_after_rebuild) : "unavailable") << '\n'
          << "Rebuilt HNSW query sweep:\n"
          << "efSearch,mean_recall,min_recall,average_us,p50_us,p95_us,p99_us,qps,short_queries\n";
      for (const auto ef_search : config.ef_search_values) {
        const auto metrics = measure_queries(
            queries, truth, deleted_ids, config.warmup_count,
            [&rebuilt, &config, ef_search](const auto& query) {
              return rebuilt.search(query, config.top_k, ef_search);
            },
            checksum);
        std::cout << ef_search << ',' << metrics.mean_recall << ',' << metrics.minimum_recall << ','
                  << metrics.average_us << ',' << metrics.p50_us << ',' << metrics.p95_us << ','
                  << metrics.p99_us << ',' << metrics.qps << ',' << metrics.short_queries << '\n';
      }
    }
    std::cout << "Checksum: " << checksum << '\n';
  } catch (const std::exception& error) {
    std::cerr << "Benchmark failed: " << error.what() << '\n';
    print_usage();
    return 1;
  }
  return 0;
}
