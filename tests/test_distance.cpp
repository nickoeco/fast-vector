#include <cmath>
#include <limits>
#include <stdexcept>
#include <vector>

#include <gtest/gtest.h>

#include "fast_vector/distance.h"

namespace {

constexpr float kTolerance = 1.0e-5F;

TEST(DistanceTest, NormalizedVectorHasUnitLength) {
    const auto normalized = fast_vector::normalize_l2(std::vector<float>{3.0F, 4.0F});
    EXPECT_NEAR(fast_vector::dot_product(normalized, normalized), 1.0F, kTolerance);
    EXPECT_NEAR(normalized[0], 0.6F, kTolerance);
    EXPECT_NEAR(normalized[1], 0.8F, kTolerance);
}

TEST(DistanceTest, ComputesExpectedDirections) {
    const auto x = fast_vector::normalize_l2(std::vector<float>{1.0F, 0.0F});
    const auto same = fast_vector::normalize_l2(std::vector<float>{2.0F, 0.0F});
    const auto orthogonal = fast_vector::normalize_l2(std::vector<float>{0.0F, 3.0F});
    const auto opposite = fast_vector::normalize_l2(std::vector<float>{-4.0F, 0.0F});

    EXPECT_NEAR(fast_vector::dot_product(x, same), 1.0F, kTolerance);
    EXPECT_NEAR(fast_vector::dot_product(x, orthogonal), 0.0F, kTolerance);
    EXPECT_NEAR(fast_vector::dot_product(x, opposite), -1.0F, kTolerance);
}

TEST(DistanceTest, RejectsZeroAndNonFiniteVectors) {
    EXPECT_THROW(fast_vector::normalize_l2(std::vector<float>{0.0F, 0.0F}), std::invalid_argument);
    EXPECT_THROW(
        fast_vector::normalize_l2(std::vector<float>{std::numeric_limits<float>::quiet_NaN(), 1.0F}),
        std::invalid_argument);
    EXPECT_THROW(
        fast_vector::normalize_l2(std::vector<float>{std::numeric_limits<float>::infinity(), 1.0F}),
        std::invalid_argument);
}

TEST(DistanceTest, RejectsMismatchedDotProductDimensions) {
    EXPECT_THROW(
        static_cast<void>(fast_vector::dot_product(
            std::vector<float>{1.0F}, std::vector<float>{1.0F, 2.0F})),
        std::invalid_argument);
}

}  // namespace
