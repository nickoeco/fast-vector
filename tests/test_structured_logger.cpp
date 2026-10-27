#include <gtest/gtest.h>

#include <sstream>
#include <thread>
#include <vector>

#include "service/structured_logger.h"

namespace {

TEST(StructuredLoggerTest, EscapesStringsAndUsesJsonScalarTypes) {
  std::ostringstream output;
  fast_vector::service::StructuredLogger logger(output);
  logger.write({.event = "escape_test",
                .request_id = 7,
                .method = "Search",
                .status = "OK",
                .latency_us = 42,
                .index_type = "flat",
                .vector_count = 3},
               {{"sample", std::string_view("quote\"\\\n\r\t\x01")},
                {"enabled", true},
                {"count", std::uint64_t{9}}});
  const auto text = output.str();
  EXPECT_NE(text.find("\"request_id\":\"7\""), std::string::npos);
  EXPECT_NE(text.find("quote\\\"\\\\\\u000a\\u000d\\u0009\\u0001"), std::string::npos);
  EXPECT_NE(text.find("\"latency_us\":42"), std::string::npos);
  EXPECT_NE(text.find("\"enabled\":true"), std::string::npos);
  EXPECT_NE(text.find("\"count\":9"), std::string::npos);
  EXPECT_EQ(text.back(), '\n');
  EXPECT_EQ(logger.dropped_records(), 0U);
}

TEST(StructuredLoggerTest, ConcurrentWritesKeepCompleteRecords) {
  std::ostringstream output;
  fast_vector::service::StructuredLogger logger(output);
  std::vector<std::jthread> threads;
  for (std::uint64_t worker = 0; worker < 4; ++worker) {
    threads.emplace_back([&logger, worker] {
      for (std::uint64_t i = 0; i < 50; ++i) {
        logger.write({.event = "concurrent", .request_id = worker * 50 + i + 1});
      }
    });
  }
  threads.clear();
  std::istringstream input(output.str());
  std::string line;
  std::size_t count = 0;
  while (std::getline(input, line)) {
    ASSERT_FALSE(line.empty());
    EXPECT_EQ(line.front(), '{');
    EXPECT_EQ(line.back(), '}');
    EXPECT_EQ(line.find('{'), line.rfind('{'));
    ++count;
  }
  EXPECT_EQ(count, 200U);
  EXPECT_EQ(logger.dropped_records(), 0U);
}

TEST(StructuredLoggerTest, StreamFailureIsContainedAndCounted) {
  std::ostringstream output;
  fast_vector::service::StructuredLogger logger(output);
  output.setstate(std::ios::badbit);
  logger.write({.event = "lost"});
  EXPECT_EQ(logger.dropped_records(), 1U);
  EXPECT_THROW(output.exceptions(std::ios::badbit), std::ios_base::failure);
  logger.write({.event = "also_lost"});
  EXPECT_EQ(logger.dropped_records(), 2U);
}

}  // namespace
