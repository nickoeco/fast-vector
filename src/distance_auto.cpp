#include "distance_kernels.h"

namespace fast_vector::detail {

float dot_product_auto(const float* lhs, const float* rhs, const std::size_t size) {
    float result = 0.0F;
    for (std::size_t i = 0; i < size; ++i) {
        result += lhs[i] * rhs[i];
    }
    return result;
}

}  // namespace fast_vector::detail
