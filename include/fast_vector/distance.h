#pragma once

#include <span>
#include <vector>

namespace fast_vector {

/** Select the dot-product implementation used by an index. */
enum class DotProductKernel {
    Scalar,
    AutoVectorized,
    Avx2,
};

/** Return an L2-normalized copy. Reject empty, zero, and non-finite vectors. */
[[nodiscard]] std::vector<float> normalize_l2(std::span<const float> vector);

/** Compute the dot product of two equal-length vectors. */
[[nodiscard]] float dot_product(
    std::span<const float> lhs, std::span<const float> rhs,
    DotProductKernel kernel = DotProductKernel::Scalar);

/** Return whether an AVX2 implementation was built and is supported by this CPU. */
[[nodiscard]] bool avx2_dot_product_available() noexcept;

[[nodiscard]] const char* dot_product_kernel_name(DotProductKernel kernel) noexcept;

}  // namespace fast_vector
