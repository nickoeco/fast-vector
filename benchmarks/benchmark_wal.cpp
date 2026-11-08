#include <algorithm>
#include <charconv>
#include <chrono>
#include <filesystem>
#include <iomanip>
#include <iostream>
#include <numeric>
#include <random>
#include <stdexcept>
#include <string>
#include <vector>

#include "fast_vector/durable_flat_index.h"

namespace {
using Clock = std::chrono::steady_clock;
double us(Clock::time_point start) {
  return std::chrono::duration<double, std::micro>(Clock::now() - start).count();
}
double percentile(std::vector<double> values, double p) {
  std::sort(values.begin(), values.end());
  const auto position = p * (values.size() - 1);
  const auto lower = static_cast<std::size_t>(position);
  return values[lower] +
         (position - lower) * (values[std::min(lower + 1, values.size() - 1)] - values[lower]);
}
std::size_t number(const char* value) {
  std::string text(value);
  std::size_t result = 0;
  const auto [end, error] = std::from_chars(text.data(), text.data() + text.size(), result);
  if (error != std::errc{} || end != text.data() + text.size() || result == 0)
    throw std::invalid_argument("expected positive integer");
  return result;
}
}  // namespace

int main(int argc, char* argv[]) {
  try {
    if (argc != 4)
      throw std::invalid_argument("usage: fast_vector_wal_benchmark NEW_DIRECTORY COUNT DIMENSION");
    const std::filesystem::path directory = argv[1];
    const auto count = number(argv[2]), dimension = number(argv[3]);
    if (count < 2 || count > 1'000'000 || dimension > 1'000'000)
      throw std::invalid_argument("benchmark size outside supported bounds");
    if (!std::filesystem::create_directory(directory))
      throw std::invalid_argument("benchmark requires a new directory");
    fast_vector::WalOptions options{
        .directory = directory, .dimension = dimension, .fault_hook = {}};
    std::mt19937 random(42);
    std::uniform_real_distribution<float> distribution(-1, 1);
    std::vector<double> latency, sync;
    latency.reserve(count);
    sync.reserve(count);
    std::vector<float> vector(dimension);
    double checkpoint_us = 0;
    {
      fast_vector::DurableFlatIndex index(options);
      for (std::size_t i = 0; i < count; ++i) {
        for (auto& value : vector) value = distribution(random);
        const auto operation = "benchmark-" + std::to_string(i);
        const auto start = Clock::now();
        (void)index.commit(operation, i, vector);
        latency.push_back(us(start));
        sync.push_back(index.durability_stats().last_sync_us);
        if (i + 1 == count / 2) {
          const auto checkpoint_start = Clock::now();
          (void)index.checkpoint();
          checkpoint_us = us(checkpoint_start);
        }
      }
    }
    double recovery_us = 0, wal_only_recovery_us = 0;
    fast_vector::DurabilityStats stats;
    {
      const auto start = Clock::now();
      fast_vector::DurableFlatIndex recovered(options);
      recovery_us = us(start);
      stats = recovered.durability_stats();
      if (recovered.size() != count || stats.recovered_records != count - count / 2)
        throw std::runtime_error("recovery count mismatch");
    }
    // This is a newly created benchmark fixture, not an operational recovery directory.
    std::filesystem::rename(directory / "recovery.manifest",
                            directory / "benchmark-saved.manifest");
    {
      const auto start = Clock::now();
      fast_vector::DurableFlatIndex recovered(options);
      wal_only_recovery_us = us(start);
      if (recovered.size() != count || recovered.durability_stats().recovered_records != count)
        throw std::runtime_error("WAL-only recovery count mismatch");
    }
    std::filesystem::rename(directory / "benchmark-saved.manifest",
                            directory / "recovery.manifest");
    const auto wal_bytes = std::filesystem::file_size(directory / "operations.wal");
    std::cout << std::fixed << std::setprecision(3) << "{\n  \"vectors\": " << count
              << ", \"dimension\": " << dimension << ", \"seed\": 42,\n"
              << "  \"build_mode\": \""
#ifdef NDEBUG
              << "Release"
#else
              << "Debug"
#endif
              << "\", \"sync_policy\": \"fdatasync_per_operation\",\n"
              << "  \"write_mean_us\": "
              << std::accumulate(latency.begin(), latency.end(), 0.0) / count << ",\n"
              << "  \"write_p50_us\": " << percentile(latency, .5)
              << ", \"write_p95_us\": " << percentile(latency, .95)
              << ", \"write_p99_us\": " << percentile(latency, .99) << ",\n"
              << "  \"sync_mean_us\": " << std::accumulate(sync.begin(), sync.end(), 0.0) / count
              << ",\n"
              << "  \"checkpoint_lsn\": " << stats.checkpoint_lsn
              << ", \"checkpoint_us\": " << checkpoint_us << ",\n"
              << "  \"recovery_us\": " << recovery_us
              << ", \"replayed_records\": " << stats.recovered_records << ",\n"
              << "  \"replay_records_per_second\": " << stats.recovered_records * 1e6 / recovery_us
              << ",\n"
              << "  \"validated_records_per_second\": " << count * 1e6 / recovery_us << ",\n"
              << "  \"wal_only_recovery_us\": " << wal_only_recovery_us << ",\n"
              << "  \"wal_only_records_per_second\": " << count * 1e6 / wal_only_recovery_us
              << ",\n"
              << "  \"wal_bytes\": " << wal_bytes << "\n}\n";
    return 0;
  } catch (const std::exception& error) {
    std::cerr << error.what() << '\n';
    return 1;
  }
}
