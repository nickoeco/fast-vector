#pragma once

#include <span>
#include <vector>

namespace fast_vector {

/** Return an L2-normalized copy. Reject empty, zero, and non-finite vectors. */
[[nodiscard]] std::vector<float> normalize_l2(std::span<const float> vector);

/** Compute the dot product of two equal-length vectors. */
[[nodiscard]] float dot_product(std::span<const float> lhs, std::span<const float> rhs);

}  // namespace fast_vector
