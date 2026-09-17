#include "fast_vector/distance.h"

#include <cmath>
#include <stdexcept>

#if defined(_MSC_VER)
#include <intrin.h>
#endif

#include "distance_kernels.h"

namespace fast_vector {
namespace detail {

float dot_product_scalar(const float* lhs, const float* rhs, const std::size_t size) {
    double result = 0.0;
    for (std::size_t i = 0; i < size; ++i) {
        result += static_cast<double>(lhs[i]) * static_cast<double>(rhs[i]);
    }
    return static_cast<float>(result);
}

RawDotProduct resolve_dot_product_kernel(const DotProductKernel kernel) {
    if (kernel == DotProductKernel::AutoVectorized) {
        return dot_product_auto;
    }
    if (kernel == DotProductKernel::Avx2) {
        if (!avx2_dot_product_available()) {
            throw std::invalid_argument("AVX2 dot product is unavailable on this build or CPU");
        }
#if defined(FAST_VECTOR_HAS_AVX2_IMPL)
        return dot_product_avx2;
#endif
    }
    return dot_product_scalar;
}

}  // namespace detail

std::vector<float> normalize_l2(const std::span<const float> vector) {
    if (vector.empty()) {
        throw std::invalid_argument("vector must not be empty");
    }

    double squared_norm = 0.0;
    for (const float value : vector) {
        if (!std::isfinite(value)) {
            throw std::invalid_argument("vector values must be finite");
        }
        const double wide_value = static_cast<double>(value);
        squared_norm += wide_value * wide_value;
    }

    if (squared_norm == 0.0) {
        throw std::invalid_argument("zero vector cannot be normalized");
    }

    const double inverse_norm = 1.0 / std::sqrt(squared_norm);
    std::vector<float> normalized;
    normalized.reserve(vector.size());
    for (const float value : vector) {
        normalized.push_back(static_cast<float>(static_cast<double>(value) * inverse_norm));
    }
    return normalized;
}

float dot_product(const std::span<const float> lhs, const std::span<const float> rhs,
                  const DotProductKernel kernel) {
    if (lhs.size() != rhs.size()) {
        throw std::invalid_argument("dot product requires vectors of equal length");
    }

    return detail::resolve_dot_product_kernel(kernel)(lhs.data(), rhs.data(), lhs.size());
}

bool avx2_dot_product_available() noexcept {
#if defined(FAST_VECTOR_HAS_AVX2_IMPL) && (defined(__GNUC__) || defined(__clang__))
    return __builtin_cpu_supports("avx2");
#elif defined(FAST_VECTOR_HAS_AVX2_IMPL) && defined(_MSC_VER)
    int registers[4]{};
    __cpuidex(registers, 7, 0);
    return (registers[1] & (1 << 5)) != 0;
#else
    return false;
#endif
}

const char* dot_product_kernel_name(const DotProductKernel kernel) noexcept {
    switch (kernel) {
        case DotProductKernel::Scalar:
            return "scalar";
        case DotProductKernel::AutoVectorized:
            return "auto";
        case DotProductKernel::Avx2:
            return "avx2";
    }
    return "unknown";
}

}  // namespace fast_vector
