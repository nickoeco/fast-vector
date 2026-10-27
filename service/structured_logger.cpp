#include "service/structured_logger.h"

#include <chrono>
#include <ctime>
#include <iomanip>
#include <locale>
#include <sstream>
#include <stdexcept>
#include <type_traits>

namespace fast_vector::service {
namespace {

void json_string(std::ostream& output, const std::string_view value) {
  constexpr char hex[] = "0123456789abcdef";
  output << '"';
  for (const unsigned char byte : value) {
    if (byte == '"' || byte == '\\') {
      output << '\\' << static_cast<char>(byte);
    } else if (byte < 0x20) {
      output << "\\u00" << hex[byte >> 4U] << hex[byte & 0x0FU];
    } else {
      output << static_cast<char>(byte);
    }
  }
  output << '"';
}

void nullable_string(std::ostream& output, const std::string_view value) {
  if (value.empty()) {
    output << "null";
  } else {
    json_string(output, value);
  }
}

void optional_number(std::ostream& output, const std::optional<std::uint64_t> value) {
  if (value) {
    output << *value;
  } else {
    output << "null";
  }
}

void timestamp(std::ostream& output) {
  const auto now = std::chrono::system_clock::now();
  const auto seconds = std::chrono::floor<std::chrono::seconds>(now);
  const auto milliseconds = std::chrono::duration_cast<std::chrono::milliseconds>(now - seconds);
  const auto time = std::chrono::system_clock::to_time_t(seconds);
  std::tm utc{};
#if defined(_WIN32)
  if (gmtime_s(&utc, &time) != 0) {
#else
  if (gmtime_r(&time, &utc) == nullptr) {
#endif
    throw std::runtime_error("UTC conversion failed");
  }
  output << '"' << std::put_time(&utc, "%Y-%m-%dT%H:%M:%S") << '.' << std::setw(3)
         << std::setfill('0') << milliseconds.count() << "Z\"";
}

}  // namespace

void StructuredLogger::write(const LogRecord& record,
                             const std::span<const LogField> fields) noexcept {
  try {
    std::ostringstream line;
    line.exceptions(std::ios::badbit | std::ios::failbit);
    line.imbue(std::locale::classic());
    line << "{\"timestamp\":";
    timestamp(line);
    line << ",\"severity\":";
    json_string(line, record.severity);
    line << ",\"event\":";
    json_string(line, record.event);
    line << ",\"request_id\":";
    if (record.request_id == 0) {
      line << "null";
    } else {
      line << '"' << record.request_id << '"';
    }
    line << ",\"method\":";
    nullable_string(line, record.method);
    line << ",\"status\":";
    nullable_string(line, record.status);
    line << ",\"latency_us\":";
    optional_number(line, record.latency_us);
    line << ",\"index_type\":";
    nullable_string(line, record.index_type);
    line << ",\"vector_count\":";
    optional_number(line, record.vector_count);
    for (const auto& field : fields) {
      line << ',';
      json_string(line, field.name);
      line << ':';
      std::visit(
          [&line](const auto& value) {
            using Value = std::decay_t<decltype(value)>;
            if constexpr (std::is_same_v<Value, std::string_view>) {
              json_string(line, value);
            } else if constexpr (std::is_same_v<Value, bool>) {
              line << (value ? "true" : "false");
            } else {
              line << value;
            }
          },
          field.value);
    }
    line << "}\n";
    const auto message = line.str();
    std::lock_guard lock(mutex_);
    output_.write(message.data(), static_cast<std::streamsize>(message.size()));
    output_.flush();
    if (!output_) {
      dropped_.fetch_add(1, std::memory_order_relaxed);
    }
  } catch (...) {
    dropped_.fetch_add(1, std::memory_order_relaxed);
  }
}

}  // namespace fast_vector::service
