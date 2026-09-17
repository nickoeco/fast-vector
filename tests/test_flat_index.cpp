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

TEST(FlatIndexTest, OptimizedKernelsPreserveTopKIds) {
    fast_vector::FlatIndex scalar(3, fast_vector::DotProductKernel::Scalar);
    fast_vector::FlatIndex automatic(3, fast_vector::DotProductKernel::AutoVectorized);
    for (fast_vector::VectorId id = 1; id <= 20; ++id) {
        const std::vector<float> vector{
            static_cast<float>(id), static_cast<float>(id % 7 + 1),
            static_cast<float>(id % 5 + 2)};
        scalar.add(id, vector);
        automatic.add(id, vector);
    }
    const std::vector<float> query{0.5F, 1.5F, 2.5F};
    const auto scalar_results = scalar.search(query, 10);
    const auto automatic_results = automatic.search(query, 10);
    ASSERT_EQ(automatic_results.size(), scalar_results.size());
    for (std::size_t i = 0; i < scalar_results.size(); ++i) {
        EXPECT_EQ(automatic_results[i].id, scalar_results[i].id);
        EXPECT_NEAR(automatic_results[i].score, scalar_results[i].score, 1.0e-5F);
    }

    if (fast_vector::avx2_dot_product_available()) {
        fast_vector::FlatIndex avx2(3, fast_vector::DotProductKernel::Avx2);
        for (fast_vector::VectorId id = 1; id <= 20; ++id) {
            avx2.add(id, std::vector<float>{
                             static_cast<float>(id), static_cast<float>(id % 7 + 1),
                             static_cast<float>(id % 5 + 2)});
        }
        const auto avx2_results = avx2.search(query, 10);
        ASSERT_EQ(avx2_results.size(), scalar_results.size());
        for (std::size_t i = 0; i < scalar_results.size(); ++i) {
            EXPECT_EQ(avx2_results[i].id, scalar_results[i].id);
            EXPECT_NEAR(avx2_results[i].score, scalar_results[i].score, 1.0e-5F);
        }
    }
}

}  // namespace
