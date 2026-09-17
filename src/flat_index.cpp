#include "fast_vector/flat_index.h"

#include <algorithm>
#include <functional>
#include <queue>
#include <stdexcept>

#include "fast_vector/distance.h"
#include "distance_kernels.h"

namespace fast_vector {
namespace {

bool is_better(const SearchResult& lhs, const SearchResult& rhs) {
    if (lhs.score != rhs.score) {
        return lhs.score > rhs.score;
    }
    return lhs.id < rhs.id;
}

struct BetterResult {
    bool operator()(const SearchResult& lhs, const SearchResult& rhs) const {
        return is_better(lhs, rhs);
    }
};

}  // namespace

FlatIndex::FlatIndex(const std::size_t dimension, const DotProductKernel kernel)
    : dimension_(dimension), kernel_(kernel) {
    if (dimension_ == 0) {
        throw std::invalid_argument("index dimension must be greater than zero");
    }
}

void FlatIndex::add(const VectorId id, const std::span<const float> vector) {
    if (vector.size() != dimension_) {
        throw std::invalid_argument("vector dimension does not match index dimension");
    }
    if (id_set_.contains(id)) {
        throw std::invalid_argument("vector ID already exists");
    }

    std::vector<float> normalized = normalize_l2(vector);

    const auto [position, inserted] = id_set_.insert(id);
    if (!inserted) {
        throw std::invalid_argument("vector ID already exists");
    }

    try {
        ids_.push_back(id);
        vectors_.insert(vectors_.end(), normalized.begin(), normalized.end());
    } catch (...) {
        // Appending trivial element types has a strong guarantee; only the ID may need rollback.
        if (!ids_.empty() && ids_.back() == id) {
            ids_.pop_back();
        }
        id_set_.erase(position);
        throw;
    }
}

std::vector<SearchResult> FlatIndex::search(
    const std::span<const float> query, const std::size_t k) const {
    if (k == 0) {
        return {};
    }
    if (query.size() != dimension_) {
        throw std::invalid_argument("query dimension does not match index dimension");
    }

    const std::vector<float> normalized_query = normalize_l2(query);
    const detail::RawDotProduct dot_product_kernel =
        detail::resolve_dot_product_kernel(kernel_);
    const std::size_t result_count = std::min(k, size());
    if (result_count == 0) {
        return {};
    }

    std::priority_queue<SearchResult, std::vector<SearchResult>, BetterResult> candidates;
    for (std::size_t i = 0; i < size(); ++i) {
        const std::span<const float> stored_vector(
            vectors_.data() + i * dimension_, dimension_);
        const SearchResult candidate{
            ids_[i],
            dot_product_kernel(normalized_query.data(), stored_vector.data(), dimension_)};

        if (candidates.size() < result_count) {
            candidates.push(candidate);
        } else if (is_better(candidate, candidates.top())) {
            candidates.pop();
            candidates.push(candidate);
        }
    }

    std::vector<SearchResult> results;
    results.reserve(result_count);
    while (!candidates.empty()) {
        results.push_back(candidates.top());
        candidates.pop();
    }
    std::sort(results.begin(), results.end(), is_better);
    return results;
}

std::size_t FlatIndex::size() const noexcept {
    return ids_.size();
}

std::size_t FlatIndex::dimension() const noexcept {
    return dimension_;
}

DotProductKernel FlatIndex::kernel() const noexcept {
    return kernel_;
}

}  // namespace fast_vector
