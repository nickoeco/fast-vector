#pragma once

#include <cstddef>

#include "fast_vector/distance.h"

namespace fast_vector::detail {

using RawDotProduct = float (*)(const float*, const float*, std::size_t);

[[nodiscard]] float dot_product_scalar(const float* lhs, const float* rhs, std::size_t size);
[[nodiscard]] float dot_product_auto(const float* lhs, const float* rhs, std::size_t size);

[[nodiscard]] RawDotProduct resolve_dot_product_kernel(DotProductKernel kernel);

#if defined(FAST_VECTOR_HAS_AVX2_IMPL)
[[nodiscard]] float dot_product_avx2(const float* lhs, const float* rhs, std::size_t size);
#endif

}  // namespace fast_vector::detail
