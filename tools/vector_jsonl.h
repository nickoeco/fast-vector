#pragma once

#include <cstdint>
#include <string_view>
#include <vector>

#include "fast_vector/vector_index.h"

namespace fast_vector::tools {

struct ParsedVectorRecord {
  VectorId id;
  std::vector<float> values;
};

/** Parse one strict vector JSON object containing only `id` and `values`. */
[[nodiscard]] ParsedVectorRecord parse_vector_json_line(std::string_view line);

/** Return true when a line contains only JSON whitespace. */
[[nodiscard]] bool is_json_whitespace_only(std::string_view line) noexcept;

}  // namespace fast_vector::tools
