#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <iomanip>
#include <iostream>
#include <numeric>
#include <random>
#include <vector>

#include "fast_vector/flat_index.h"

namespace {

using Clock = std::chrono::steady_clock;

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

}  // namespace

int main() {
    constexpr std::size_t kVectorCount = 10'000;
    constexpr std::size_t kDimension = 128;
    constexpr std::size_t kQueryCount = 1'000;
    constexpr std::size_t kTopK = 10;
    constexpr std::size_t kWarmupCount = 20;
    constexpr std::uint32_t kSeed = 20250908U;

    std::mt19937 generator(kSeed);
    fast_vector::FlatIndex index(kDimension);

    const auto build_start = Clock::now();
    for (std::size_t i = 0; i < kVectorCount; ++i) {
        index.add(static_cast<fast_vector::VectorId>(i), random_vector(generator, kDimension));
    }
    const auto build_end = Clock::now();

    std::vector<std::vector<float>> queries;
    queries.reserve(kQueryCount);
    for (std::size_t i = 0; i < kQueryCount; ++i) {
        queries.push_back(random_vector(generator, kDimension));
    }

    double checksum = 0.0;
    for (std::size_t i = 0; i < kWarmupCount; ++i) {
        checksum += index.search(queries[i % queries.size()], kTopK).front().score;
    }

    std::vector<double> latencies_us;
    latencies_us.reserve(kQueryCount);
    const auto search_start = Clock::now();
    for (const auto& query : queries) {
        const auto query_start = Clock::now();
        const auto results = index.search(query, kTopK);
        const auto query_end = Clock::now();
        checksum += results.front().score;
        latencies_us.push_back(
            std::chrono::duration<double, std::micro>(query_end - query_start).count());
    }
    const auto search_end = Clock::now();

    const double build_ms =
        std::chrono::duration<double, std::milli>(build_end - build_start).count();
    const double total_search_seconds =
        std::chrono::duration<double>(search_end - search_start).count();
    const double average_us =
        std::accumulate(latencies_us.begin(), latencies_us.end(), 0.0) /
        static_cast<double>(latencies_us.size());
    std::sort(latencies_us.begin(), latencies_us.end());

    std::cout << std::fixed << std::setprecision(3)
              << "Vector count: " << kVectorCount << '\n'
              << "Dimension: " << kDimension << '\n'
              << "Query count: " << kQueryCount << '\n'
              << "Top-K: " << kTopK << '\n'
              << "Random seed: " << kSeed << '\n'
              << "Index build time (ms): " << build_ms << '\n'
              << "Average query latency (us): " << average_us << '\n'
              << "P50 query latency (us): " << percentile(latencies_us, 0.50) << '\n'
              << "P95 query latency (us): " << percentile(latencies_us, 0.95) << '\n'
              << "P99 query latency (us): " << percentile(latencies_us, 0.99) << '\n'
              << "QPS: " << static_cast<double>(kQueryCount) / total_search_seconds << '\n'
              << "Checksum: " << checksum << '\n';
    return 0;
}
