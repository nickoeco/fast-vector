#pragma once

#include <array>
#include <atomic>
#include <cstdint>
#include <filesystem>
#include <fstream>
#include <memory>
#include <stdexcept>

namespace fast_vector {
class ResourceExhausted : public std::runtime_error {
 public:
  ResourceExhausted() : std::runtime_error("configured resource budget exceeded") {}
};
struct ResourceLimits {
  std::uint64_t maximum_dimension = 65'536;
  std::uint64_t maximum_vectors = 1'000'000;
  std::uint64_t maximum_index_bytes = 1024ULL * 1024 * 1024;
  std::uint64_t maximum_top_k = 10'000;
};
inline std::uint64_t resource_add(std::uint64_t a, std::uint64_t b) {
  if (b > UINT64_MAX - a) throw ResourceExhausted();
  return a + b;
}
inline std::uint64_t resource_multiply(std::uint64_t a, std::uint64_t b) {
  if (a && b > UINT64_MAX / a) throw ResourceExhausted();
  return a * b;
}
inline void check_shape(const ResourceLimits& limits, std::uint64_t dimension,
                        std::uint64_t count) {
  if (dimension > limits.maximum_dimension || count > limits.maximum_vectors ||
      resource_multiply(count, resource_add(resource_multiply(dimension, 4), 512)) >
          limits.maximum_index_bytes)
    throw ResourceExhausted();
}
inline void check_wal_shape(const ResourceLimits& limits, std::uint64_t dimension,
                            std::uint64_t count) {
  check_shape(limits, dimension, count);
  if (resource_multiply(count, resource_add(resource_multiply(dimension, 4), 1024)) >
      limits.maximum_index_bytes)
    throw ResourceExhausted();
}
/** Fixed header preflight; full format/checksum validation remains the loader's job. */
inline std::uint64_t snapshot_loading_units(const std::filesystem::path& path,
                                            const ResourceLimits& limits) {
  std::ifstream input(path, std::ios::binary);
  std::array<unsigned char, 40> bytes{};
  input.read(reinterpret_cast<char*>(bytes.data()), bytes.size());
  if (!input) throw std::runtime_error("snapshot header truncated");
  auto get = [&](std::size_t offset) {
    std::uint64_t value = 0;
    for (std::size_t i = 0; i < 8; ++i) value |= std::uint64_t{bytes[offset + i]} << (i * 8);
    return value;
  };
  const auto dimension = get(24), count = get(32);
  check_shape(limits, dimension, count);
  return resource_add(resource_multiply(std::filesystem::file_size(path), 16),
                      resource_multiply(count, 512));
}
/** Accounted admission units, not allocator/RSS measurements. Leases retain their budget. */
class MemoryBudget : public std::enable_shared_from_this<MemoryBudget> {
 public:
  explicit MemoryBudget(std::uint64_t limit) : limit_(limit) {}
  [[nodiscard]] std::shared_ptr<void> reserve(std::uint64_t bytes) {
    auto owner = shared_from_this();
    auto current = used_.load();
    do {
      if (current > limit_ || bytes > limit_ - current) throw ResourceExhausted();
    } while (!used_.compare_exchange_weak(current, current + bytes));
    // shared_ptr invokes its deleter even if control-block allocation fails.
    return std::shared_ptr<void>(
        this, [owner = std::move(owner), bytes](void*) { owner->used_.fetch_sub(bytes); });
  }
  std::uint64_t used() const noexcept { return used_.load(); }
  std::uint64_t limit() const noexcept { return limit_; }

 private:
  std::uint64_t limit_;
  std::atomic<std::uint64_t> used_{0};
};
}  // namespace fast_vector
