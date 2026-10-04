#include "tools/vector_jsonl.h"

#include <charconv>
#include <cmath>
#include <cstddef>
#include <limits>
#include <stdexcept>
#include <string>
#include <utility>

namespace fast_vector::tools {
namespace {

bool is_space(const char value) noexcept {
  return value == ' ' || value == '\t' || value == '\r' || value == '\n';
}

class JsonCursor {
 public:
  explicit JsonCursor(const std::string_view input) : input_(input) {}

  void skip_whitespace() noexcept {
    while (position_ < input_.size() && is_space(input_[position_])) {
      ++position_;
    }
  }

  void expect(const char expected) {
    skip_whitespace();
    if (position_ >= input_.size() || input_[position_] != expected) {
      fail(std::string("expected '") + expected + "'");
    }
    ++position_;
  }

  [[nodiscard]] bool consume(const char expected) noexcept {
    skip_whitespace();
    if (position_ < input_.size() && input_[position_] == expected) {
      ++position_;
      return true;
    }
    return false;
  }

  [[nodiscard]] std::string parse_key() {
    skip_whitespace();
    if (position_ >= input_.size() || input_[position_] != '"') {
      fail("expected an object key");
    }
    ++position_;
    const std::size_t begin = position_;
    while (position_ < input_.size() && input_[position_] != '"') {
      const unsigned char value = static_cast<unsigned char>(input_[position_]);
      if (value < 0x20U || input_[position_] == '\\') {
        fail("object keys must be unescaped printable ASCII");
      }
      ++position_;
    }
    if (position_ >= input_.size()) {
      fail("unterminated object key");
    }
    std::string key(input_.substr(begin, position_ - begin));
    ++position_;
    return key;
  }

  [[nodiscard]] VectorId parse_id() {
    skip_whitespace();
    const std::size_t begin = position_;
    while (position_ < input_.size() && input_[position_] >= '0' && input_[position_] <= '9') {
      ++position_;
    }
    if (begin == position_) {
      fail("id must be an unsigned integer");
    }
    if (position_ - begin > 1 && input_[begin] == '0') {
      fail("id must not contain leading zeros");
    }
    VectorId id = 0;
    const auto [end, error] = std::from_chars(input_.data() + begin, input_.data() + position_, id);
    if (error != std::errc{} || end != input_.data() + position_) {
      fail("id is outside the uint64 range");
    }
    return id;
  }

  [[nodiscard]] float parse_float() {
    skip_whitespace();
    const std::size_t begin = position_;
    while (position_ < input_.size() && !is_space(input_[position_]) && input_[position_] != ',' &&
           input_[position_] != ']') {
      ++position_;
    }
    if (begin == position_) {
      fail("expected a floating-point value");
    }
    float value = 0.0F;
    const auto [end, error] = std::from_chars(input_.data() + begin, input_.data() + position_,
                                              value, std::chars_format::general);
    if (error != std::errc{} || end != input_.data() + position_ || !std::isfinite(value)) {
      fail("values must contain finite JSON numbers representable as float32");
    }
    return value;
  }

  [[nodiscard]] std::vector<float> parse_values() {
    expect('[');
    std::vector<float> values;
    if (consume(']')) {
      return values;
    }
    while (true) {
      values.push_back(parse_float());
      if (consume(']')) {
        return values;
      }
      expect(',');
    }
  }

  void expect_end() {
    skip_whitespace();
    if (position_ != input_.size()) {
      fail("unexpected trailing data");
    }
  }

 private:
  [[noreturn]] void fail(const std::string& message) const {
    throw std::invalid_argument(message + " at byte " + std::to_string(position_ + 1));
  }

  std::string_view input_;
  std::size_t position_ = 0;
};

}  // namespace

ParsedVectorRecord parse_vector_json_line(const std::string_view line) {
  JsonCursor cursor(line);
  cursor.expect('{');
  bool has_id = false;
  bool has_values = false;
  VectorId id = 0;
  std::vector<float> values;

  if (cursor.consume('}')) {
    throw std::invalid_argument("vector object must contain id and values");
  }
  while (true) {
    const std::string key = cursor.parse_key();
    cursor.expect(':');
    if (key == "id") {
      if (has_id) {
        throw std::invalid_argument("duplicate id member");
      }
      id = cursor.parse_id();
      has_id = true;
    } else if (key == "values") {
      if (has_values) {
        throw std::invalid_argument("duplicate values member");
      }
      values = cursor.parse_values();
      has_values = true;
    } else {
      throw std::invalid_argument("unknown vector member: " + key);
    }

    if (cursor.consume('}')) {
      break;
    }
    cursor.expect(',');
  }
  cursor.expect_end();
  if (!has_id || !has_values) {
    throw std::invalid_argument("vector object must contain id and values");
  }
  return ParsedVectorRecord{id, std::move(values)};
}

bool is_json_whitespace_only(const std::string_view line) noexcept {
  for (const char value : line) {
    if (!is_space(value)) {
      return false;
    }
  }
  return true;
}

}  // namespace fast_vector::tools
