#include <algorithm>
#include <charconv>
#include <chrono>
#include <filesystem>
#include <fstream>
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
std::uint64_t rss_kib() {
  std::ifstream input("/proc/self/status");
  std::string line;
  while (std::getline(input, line))
    if (line.starts_with("VmRSS:")) return std::stoull(line.substr(6));
  return 0;
}
std::uint64_t directory_bytes(const std::filesystem::path& directory) {
  std::uint64_t bytes = 0;
  for (const auto& entry : std::filesystem::directory_iterator(directory))
    if (entry.is_regular_file()) bytes += entry.file_size();
  return bytes;
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
    if (argc != 4 && argc != 5 && argc != 6)
      throw std::invalid_argument(
          "usage: fast_vector_wal_benchmark NEW_DIRECTORY COUNT DIMENSION [SEGMENT_BYTES "
          "[EPOCH_OPERATIONS]]");
    const std::filesystem::path directory = argv[1];
    const auto count = number(argv[2]), dimension = number(argv[3]);
    if (count < 2 || count > 1'000'000 || dimension > 1'000'000)
      throw std::invalid_argument("benchmark size outside supported bounds");
    if (!std::filesystem::create_directory(directory))
      throw std::invalid_argument("benchmark requires a new directory");
    fast_vector::WalOptions options{
        .directory = directory, .dimension = dimension, .fault_hook = {}};
    if (argc >= 5) {
      options.rotate_on_checkpoint = true;
      options.maximum_wal_bytes = number(argv[4]);
    }
    if (argc == 6) {
      options.dedup_retention = true;
      options.maximum_epoch_operations = number(argv[5]);
    }
    std::mt19937 random(42);
    std::uniform_real_distribution<float> distribution(-1, 1);
    std::vector<double> latency, sync;
    latency.reserve(count);
    sync.reserve(count);
    std::vector<float> vector(dimension);
    std::vector<std::uint64_t> epochs(count, 0);
    double checkpoint_us = 0;
    double quota_checkpoint_us = 0, build_us = 0;
    std::uint64_t quota_checkpoints = 0;
    std::uint64_t before_bytes = 0, after_bytes = 0, wal_before = 0, wal_after = 0, cleaned = 0;
    {
      fast_vector::DurableFlatIndex index(options);
      const auto build_start = Clock::now();
      for (std::size_t i = 0; i < count; ++i) {
        auto state = index.durability_stats();
        if (options.dedup_retention &&
            state.current_epoch_operations >= state.maximum_epoch_operations) {
          const auto quota_start = Clock::now();
          (void)index.checkpoint();
          quota_checkpoint_us += us(quota_start);
          ++quota_checkpoints;
          state = index.durability_stats();
        }
        epochs[i] = state.current_epoch;
        for (auto& value : vector) value = distribution(random);
        const auto operation = "benchmark-" + std::to_string(i);
        const auto start = Clock::now();
        (void)index.commit(operation, i, vector, nullptr, epochs[i]);
        latency.push_back(us(start));
        sync.push_back(index.durability_stats().last_sync_us);
        if (i + 1 == count / 2) {
          before_bytes = directory_bytes(directory);
          wal_before = index.durability_stats().active_wal_bytes;
          const auto checkpoint_start = Clock::now();
          (void)index.checkpoint();
          checkpoint_us = us(checkpoint_start);
          after_bytes = directory_bytes(directory);
          wal_after = index.durability_stats().active_wal_bytes;
        }
      }
      cleaned = index.durability_stats().cleaned_files;
      build_us = us(build_start);
    }
    double recovery_us = 0, wal_only_recovery_us = 0;
    fast_vector::DurabilityStats stats;
    double retry_us = 0;
    std::uint64_t recovery_rss = 0;
    std::uint64_t retried = 0, expired = 0;
    {
      const auto start = Clock::now();
      fast_vector::DurableFlatIndex recovered(options);
      recovery_us = us(start);
      stats = recovered.durability_stats();
      if (recovered.size() != count || stats.recovered_records != count - stats.checkpoint_lsn)
        throw std::runtime_error("recovery count mismatch");
      recovery_rss = rss_kib();
      std::mt19937 repeated(42);
      for (std::size_t i = 0; i < count; ++i) {
        for (auto& value : vector) value = distribution(repeated);
        const auto key = "benchmark-" + std::to_string(i);
        if (epochs[i] < stats.oldest_epoch) {
          try {
            (void)recovered.commit(key, i, vector, nullptr, epochs[i]);
            throw std::runtime_error("expired retry was accepted");
          } catch (const fast_vector::DedupEpochError&) {
            ++expired;
          }
          continue;
        }
        const auto retry_start = Clock::now();
        const auto retry = recovered.commit(key, i, vector, nullptr, epochs[i]);
        retry_us += us(retry_start);
        ++retried;
        if (!retry.replayed) throw std::runtime_error("retry unexpectedly inserted");
      }
      retry_us /= retried;
      if (recovered.size() != count || recovered.durability_stats().applied_lsn != count)
        throw std::runtime_error("retry altered index");
    }
    // This is a newly created benchmark fixture, not an operational recovery directory.
    if (!options.rotate_on_checkpoint) {
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
    }
    const auto wal_bytes = stats.active_wal_bytes;
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
              << "  \"wal_only_recovery_us\": "
              << (options.rotate_on_checkpoint ? "null" : std::to_string(wal_only_recovery_us))
              << ",\n"
              << "  \"wal_only_records_per_second\": "
              << (options.rotate_on_checkpoint ? "null"
                                               : std::to_string(count * 1e6 / wal_only_recovery_us))
              << ",\n"
              << "  \"dedup_metadata_bytes\": " << stats.dedup_metadata_bytes << ",\n"
              << "  \"old_original_vector_payload_bytes\": " << count * dimension * 4 << ",\n"
              << "  \"dedup_file_bytes\": " << stats.dedup_file_bytes << ",\n"
              << "  \"wal_scan_bytes\": " << stats.wal_scan_bytes << ",\n"
              << "  \"retry_mean_us\": " << retry_us << ",\n"
              << "  \"recovery_rss_kib\": " << recovery_rss << ",\n"
              << "  \"rotation_enabled\": " << (options.rotate_on_checkpoint ? "true" : "false")
              << ",\n"
              << "  \"segment_byte_limit\": " << options.maximum_wal_bytes << ",\n"
              << "  \"wal_generation\": " << stats.wal_generation << ",\n"
              << "  \"cleaned_files_during_build\": " << cleaned << ",\n"
              << "  \"directory_bytes\": " << directory_bytes(directory) << ",\n"
              << "  \"directory_bytes_before_checkpoint\": " << before_bytes << ",\n"
              << "  \"directory_bytes_after_checkpoint\": " << after_bytes << ",\n"
              << "  \"wal_before_checkpoint_bytes\": " << wal_before << ",\n"
              << "  \"wal_after_checkpoint_bytes\": " << wal_after << ",\n"
              << "  \"current_epoch\": " << stats.current_epoch << ",\n"
              << "  \"oldest_epoch\": " << stats.oldest_epoch << ",\n"
              << "  \"retained_epochs\": " << stats.retained_epochs << ",\n"
              << "  \"epoch_operation_limit\": " << stats.maximum_epoch_operations << ",\n"
              << "  \"dedup_entries\": " << stats.dedup_entries << ",\n"
              << "  \"retained_retries\": " << retried << ",\n"
              << "  \"expired_rejections\": " << expired << ",\n"
              << "  \"quota_checkpoints\": " << quota_checkpoints << ",\n"
              << "  \"quota_checkpoint_total_us\": " << quota_checkpoint_us << ",\n"
              << "  \"build_pipeline_us\": " << build_us << ",\n"
              << "  \"wal_bytes\": " << wal_bytes << "\n}\n";
    return 0;
  } catch (const std::exception& error) {
    std::cerr << error.what() << '\n';
    return 1;
  }
}
