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
#include <vector>

#if defined(__linux__)
#include <unistd.h>
#endif

#include "fast_vector/flat_index.h"
#include "fast_vector/flat_index_io.h"

namespace {

using Clock = std::chrono::steady_clock;

struct BenchmarkConfig {
    std::size_t vector_count = 10'000;
    std::size_t dimension = 128;
    std::size_t query_count = 1'000;
    std::size_t top_k = 10;
    std::size_t warmup_count = 20;
    std::uint32_t seed = 20250908U;
};

class TemporaryIndexFile {
public:
    TemporaryIndexFile() {
        const auto timestamp = Clock::now().time_since_epoch().count();
        path_ = std::filesystem::temp_directory_path() /
                ("fast-vector-benchmark-" + std::to_string(timestamp) + ".fv");
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
        } else if (option == "--seed") {
            const std::uint64_t seed = parse_unsigned(value, option);
            if (seed > std::numeric_limits<std::uint32_t>::max()) {
                throw std::invalid_argument("--seed must fit in uint32_t");
            }
            config.seed = static_cast<std::uint32_t>(seed);
        } else {
            throw std::invalid_argument("unknown benchmark option: " + option);
        }
    }
    return config;
}

double percentile(const std::vector<double>& sorted_values, const double fraction) {
    const auto rank = static_cast<std::size_t>(
        std::ceil(fraction * static_cast<double>(sorted_values.size())));
    return sorted_values[std::max<std::size_t>(1, rank) - 1];
}

std::vector<float> random_vector(std::mt19937& generator, const std::size_t dimension) {
    std::uniform_real_distribution<float> distribution(-1.0F, 1.0F);
    std::vector<float> values(dimension);
    for (float& value : values) {
        value = distribution(generator);
    }
    return values;
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

std::uint64_t non_negative_difference(
    const std::optional<std::uint64_t> after, const std::optional<std::uint64_t> before) {
    if (!after || !before || *after < *before) {
        return 0;
    }
    return *after - *before;
}

void print_usage() {
    std::cerr << "Usage: fast_vector_benchmark [--vectors N] [--dimension D] "
                 "[--queries Q] [--k K] [--warmup W] [--seed S]\n";
}

}  // namespace

int main(const int argc, char* argv[]) {
    try {
        const BenchmarkConfig config = parse_arguments(argc, argv);
        std::mt19937 generator(config.seed);

        std::vector<std::vector<float>> queries;
        queries.reserve(config.query_count);
        for (std::size_t i = 0; i < config.query_count; ++i) {
            queries.push_back(random_vector(generator, config.dimension));
        }

        const auto rss_before_build = resident_set_size_bytes();
        fast_vector::FlatIndex index(config.dimension);
        const auto build_start = Clock::now();
        for (std::size_t i = 0; i < config.vector_count; ++i) {
            index.add(static_cast<fast_vector::VectorId>(i),
                      random_vector(generator, config.dimension));
        }
        const auto build_end = Clock::now();
        const auto rss_after_build = resident_set_size_bytes();

        TemporaryIndexFile index_file;
        const auto save_start = Clock::now();
        fast_vector::save_flat_index(index, index_file.path());
        const auto save_end = Clock::now();
        const std::uint64_t index_file_size = std::filesystem::file_size(index_file.path());

        const auto load_start = Clock::now();
        const fast_vector::FlatIndex loaded_index =
            fast_vector::load_flat_index(index_file.path());
        const auto load_end = Clock::now();
        const auto rss_after_load = resident_set_size_bytes();

        double checksum = 0.0;
        for (std::size_t i = 0; i < config.warmup_count; ++i) {
            checksum +=
                loaded_index.search(queries[i % queries.size()], config.top_k).front().score;
        }

        std::vector<double> latencies_us;
        latencies_us.reserve(config.query_count);
        const auto search_start = Clock::now();
        for (const auto& query : queries) {
            const auto query_start = Clock::now();
            const auto results = loaded_index.search(query, config.top_k);
            const auto query_end = Clock::now();
            checksum += results.front().score;
            latencies_us.push_back(
                std::chrono::duration<double, std::micro>(query_end - query_start).count());
        }
        const auto search_end = Clock::now();

        const double build_ms =
            std::chrono::duration<double, std::milli>(build_end - build_start).count();
        const double save_ms =
            std::chrono::duration<double, std::milli>(save_end - save_start).count();
        const double load_ms =
            std::chrono::duration<double, std::milli>(load_end - load_start).count();
        const double total_search_seconds =
            std::chrono::duration<double>(search_end - search_start).count();
        const double average_us =
            std::accumulate(latencies_us.begin(), latencies_us.end(), 0.0) /
            static_cast<double>(latencies_us.size());
        std::sort(latencies_us.begin(), latencies_us.end());

#if defined(NDEBUG)
        constexpr std::string_view kBuildMode = "Release-like (NDEBUG defined)";
#else
        constexpr std::string_view kBuildMode = "Debug-like (NDEBUG not defined)";
#endif

        std::cout << std::fixed << std::setprecision(3)
                  << "Build mode: " << kBuildMode << '\n'
                  << "Hardware concurrency: " << std::thread::hardware_concurrency() << '\n'
                  << "Vector count: " << config.vector_count << '\n'
                  << "Dimension: " << config.dimension << '\n'
                  << "Query count: " << config.query_count << '\n'
                  << "Top-K: " << config.top_k << '\n'
                  << "Warmup queries: " << config.warmup_count << '\n'
                  << "Random seed: " << config.seed << '\n'
                  << "Index build time (ms): " << build_ms << '\n'
                  << "Index save time (ms): " << save_ms << '\n'
                  << "Index load time (ms): " << load_ms << '\n'
                  << "Index file size (bytes): " << index_file_size << '\n'
                  << "Approximate build RSS delta (bytes): "
                  << non_negative_difference(rss_after_build, rss_before_build) << '\n'
                  << "Approximate loaded-index RSS delta (bytes): "
                  << non_negative_difference(rss_after_load, rss_after_build) << '\n'
                  << "Average query latency (us): " << average_us << '\n'
                  << "P50 query latency (us): " << percentile(latencies_us, 0.50) << '\n'
                  << "P95 query latency (us): " << percentile(latencies_us, 0.95) << '\n'
                  << "P99 query latency (us): " << percentile(latencies_us, 0.99) << '\n'
                  << "QPS: " << static_cast<double>(config.query_count) / total_search_seconds
                  << '\n'
                  << "Checksum: " << checksum << '\n';
    } catch (const std::exception& error) {
        std::cerr << "Benchmark failed: " << error.what() << '\n';
        print_usage();
        return 1;
    }
    return 0;
}
