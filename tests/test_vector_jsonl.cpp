#include <gtest/gtest.h>

#include <limits>
#include <stdexcept>
#include <string_view>

#include "tools/vector_jsonl.h"

namespace {

TEST(VectorJsonlTest, ParsesMembersInEitherOrder) {
  const auto first =
      fast_vector::tools::parse_vector_json_line(R"({"id": 42, "values": [1.0, -2.5e-1, 3]})");
  EXPECT_EQ(first.id, 42U);
  EXPECT_EQ(first.values, (std::vector<float>{1.0F, -0.25F, 3.0F}));

  const auto second = fast_vector::tools::parse_vector_json_line(
      R"( { "values" : [0.5], "id" : 18446744073709551615 } )");
  EXPECT_EQ(second.id, std::numeric_limits<fast_vector::VectorId>::max());
  EXPECT_EQ(second.values, (std::vector<float>{0.5F}));
}

TEST(VectorJsonlTest, RecognizesBlankLines) {
  EXPECT_TRUE(fast_vector::tools::is_json_whitespace_only(" \t\r"));
  EXPECT_FALSE(fast_vector::tools::is_json_whitespace_only(" {} "));
}

TEST(VectorJsonlTest, RejectsInvalidObjects) {
  constexpr std::string_view invalid_lines[] = {
      R"({})",
      R"({"id": 1})",
      R"({"values": [1.0]})",
      R"({"id": -1, "values": [1.0]})",
      R"({"id": 01, "values": [1.0]})",
      R"({"id": 1, "id": 2, "values": [1.0]})",
      R"({"id": 1, "values": [1.0], "extra": 2})",
      R"({"id": 1, "values": [1.0,]})",
      R"({"id": 1, "values": [1.0]} trailing)",
      R"({"id": 18446744073709551616, "values": [1.0]})",
      R"({"id": 1, "values": [1e1000]})",
  };
  for (const std::string_view line : invalid_lines) {
    EXPECT_THROW(static_cast<void>(fast_vector::tools::parse_vector_json_line(line)),
                 std::invalid_argument)
        << line;
  }
}

}  // namespace
