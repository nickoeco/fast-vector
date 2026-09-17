#include "distance_kernels.h"

#include <immintrin.h>

namespace fast_vector::detail {

float dot_product_avx2(const float* lhs, const float* rhs, const std::size_t size) {
    __m256 sum = _mm256_setzero_ps();
    std::size_t i = 0;
    for (; i + 8 <= size; i += 8) {
        const __m256 left = _mm256_loadu_ps(lhs + i);
        const __m256 right = _mm256_loadu_ps(rhs + i);
        sum = _mm256_add_ps(sum, _mm256_mul_ps(left, right));
    }

    alignas(32) float lanes[8];
    _mm256_store_ps(lanes, sum);
    float result = 0.0F;
    for (const float lane : lanes) {
        result += lane;
    }
    for (; i < size; ++i) {
        result += lhs[i] * rhs[i];
    }
    return result;
}

}  // namespace fast_vector::detail
