#pragma once

#include <cstddef>
#include <span>
#include <vector>

#include "fast_vector/search_result.h"

namespace fast_vector {

/** Common interface for fixed-dimension vector indexes. */
class VectorIndex {
public:
    virtual ~VectorIndex() = default;

    /** Add one vector. The index copies the input and owns the stored data. */
    virtual void add(VectorId id, std::span<const float> vector) = 0;

    /** Return at most k neighbors ordered by descending cosine similarity. */
    [[nodiscard]] virtual std::vector<SearchResult> search(
        std::span<const float> query, std::size_t k) const = 0;

    [[nodiscard]] virtual std::size_t size() const noexcept = 0;
    [[nodiscard]] virtual std::size_t dimension() const noexcept = 0;
};

}  // namespace fast_vector
