#include "fast_vector/distance.h"

#include <cmath>
#include <stdexcept>

namespace fast_vector {

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

float dot_product(const std::span<const float> lhs, const std::span<const float> rhs) {
    if (lhs.size() != rhs.size()) {
        throw std::invalid_argument("dot product requires vectors of equal length");
    }

    double result = 0.0;
    for (std::size_t i = 0; i < lhs.size(); ++i) {
        result += static_cast<double>(lhs[i]) * static_cast<double>(rhs[i]);
    }
    return static_cast<float>(result);
}

}  // namespace fast_vector
