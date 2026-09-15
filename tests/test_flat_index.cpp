#include <limits>
#include <stdexcept>
#include <vector>

#include <gtest/gtest.h>

#include "fast_vector/flat_index.h"

namespace {

constexpr float kTolerance = 1.0e-5F;

TEST(FlatIndexTest, RejectsZeroDimension) {
    EXPECT_THROW(fast_vector::FlatIndex(0), std::invalid_argument);
}

TEST(FlatIndexTest, AddsAndFindsOneVector) {
    fast_vector::FlatIndex index(2);
    index.add(42, std::vector<float>{3.0F, 4.0F});

    const auto results = index.search(std::vector<float>{3.0F, 4.0F}, 1);
    ASSERT_EQ(results.size(), 1U);
    EXPECT_EQ(results[0].id, 42U);
    EXPECT_NEAR(results[0].score, 1.0F, kTolerance);
    EXPECT_EQ(index.size(), 1U);
    EXPECT_EQ(index.dimension(), 2U);
}

TEST(FlatIndexTest, ReturnsCorrectDeterministicTopK) {
    fast_vector::FlatIndex index(2);
    index.add(30, std::vector<float>{0.0F, 1.0F});
    index.add(20, std::vector<float>{1.0F, -1.0F});
    index.add(10, std::vector<float>{1.0F, 1.0F});
    index.add(40, std::vector<float>{-1.0F, 0.0F});

    const auto results = index.search(std::vector<float>{1.0F, 0.0F}, 3);
    ASSERT_EQ(results.size(), 3U);
    EXPECT_EQ(results[0].id, 10U);
    EXPECT_EQ(results[1].id, 20U);
    EXPECT_EQ(results[2].id, 30U);
    EXPECT_GE(results[0].score, results[1].score);
    EXPECT_GE(results[1].score, results[2].score);
}

TEST(FlatIndexTest, RejectsInvalidInputWithoutChangingIndex) {
    fast_vector::FlatIndex index(2);
    index.add(1, std::vector<float>{1.0F, 0.0F});

    EXPECT_THROW(index.add(1, std::vector<float>{0.0F, 1.0F}), std::invalid_argument);
    EXPECT_THROW(index.add(2, std::vector<float>{1.0F}), std::invalid_argument);
    EXPECT_THROW(index.add(2, std::vector<float>{0.0F, 0.0F}), std::invalid_argument);
    EXPECT_THROW(
        index.add(2, std::vector<float>{std::numeric_limits<float>::infinity(), 0.0F}),
        std::invalid_argument);
    EXPECT_EQ(index.size(), 1U);

    const auto results = index.search(std::vector<float>{1.0F, 0.0F}, 1);
    ASSERT_EQ(results.size(), 1U);
    EXPECT_EQ(results[0].id, 1U);
}

TEST(FlatIndexTest, HandlesSearchBoundaries) {
    fast_vector::FlatIndex empty(2);
    EXPECT_TRUE(empty.search(std::vector<float>{1.0F, 0.0F}, 5).empty());
    EXPECT_TRUE(empty.search(std::vector<float>{}, 0).empty());
    EXPECT_THROW(empty.search(std::vector<float>{1.0F}, 1), std::invalid_argument);
    EXPECT_THROW(empty.search(std::vector<float>{0.0F, 0.0F}, 1), std::invalid_argument);

    fast_vector::FlatIndex index(2);
    index.add(1, std::vector<float>{1.0F, 0.0F});
    index.add(2, std::vector<float>{0.0F, 1.0F});
    EXPECT_EQ(index.search(std::vector<float>{1.0F, 0.0F}, 10).size(), 2U);
    EXPECT_TRUE(index.search(std::vector<float>{}, 0).empty());
    EXPECT_THROW(index.search(std::vector<float>{1.0F}, 1), std::invalid_argument);
    EXPECT_THROW(index.search(std::vector<float>{0.0F, 0.0F}, 1), std::invalid_argument);
}

TEST(FlatIndexTest, RepeatedSearchDoesNotChangeState) {
    fast_vector::FlatIndex index(2);
    index.add(7, std::vector<float>{1.0F, 0.0F});
    index.add(8, std::vector<float>{0.0F, 1.0F});
    const auto first = index.search(std::vector<float>{0.5F, 0.5F}, 2);
    const auto second = index.search(std::vector<float>{0.5F, 0.5F}, 2);

    EXPECT_EQ(first, second);
    EXPECT_EQ(index.size(), 2U);
}

}  // namespace
