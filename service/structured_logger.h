#pragma once

#include <atomic>
#include <cstdint>
#include <initializer_list>
#include <mutex>
#include <optional>
#include <ostream>
#include <span>
#include <string_view>
#include <variant>

namespace fast_vector::service {

struct LogRecord {
  std::string_view severity = "INFO";
  std::string_view event{};
  std::uint64_t request_id = 0;
  std::string_view method{};
  std::string_view status{};
  std::optional<std::uint64_t> latency_us = std::nullopt;
  std::string_view index_type{};
  std::optional<std::uint64_t> vector_count = std::nullopt;
};

struct LogField {
  std::string_view name;
  std::variant<std::string_view, std::uint64_t, bool> value;
};

/** One shared writer per stream. The stream must outlive the logger.
 * Call sites supply allowlisted metadata, never request payloads or exception messages.
 */
class StructuredLogger {
 public:
  explicit StructuredLogger(std::ostream& output) : output_(output) {}
  /** Best-effort logging: allocation or stream failure never escapes into request handling. */
  void write(const LogRecord& record, std::span<const LogField> fields = {}) noexcept;
  void write(const LogRecord& record, std::initializer_list<LogField> fields) noexcept {
    write(record, std::span<const LogField>(fields.begin(), fields.size()));
  }
  [[nodiscard]] std::uint64_t dropped_records() const noexcept {
    return dropped_.load(std::memory_order_relaxed);
  }

 private:
  std::ostream& output_;
  std::mutex mutex_;
  std::atomic<std::uint64_t> dropped_{0};
};

}  // namespace fast_vector::service
