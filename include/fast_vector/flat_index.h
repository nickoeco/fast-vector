#pragma once

#include <cstddef>
#include <span>
#include <unordered_set>
#include <vector>

#include "fast_vector/vector_index.h"

namespace fast_vector {

class FlatIndexSerializer;

/** Exact cosine-similarity index backed by row-major contiguous storage. */
class FlatIndex final : public VectorIndex {
public:
    explicit FlatIndex(std::size_t dimension);

    void add(VectorId id, std::span<const float> vector) override;

    [[nodiscard]] std::vector<SearchResult> search(
        std::span<const float> query, std::size_t k) const override;

    [[nodiscard]] std::size_t size() const noexcept override;
    [[nodiscard]] std::size_t dimension() const noexcept override;

private:
    friend class FlatIndexSerializer;

    std::size_t dimension_;
    std::vector<float> vectors_;
    std::vector<VectorId> ids_;
    std::unordered_set<VectorId> id_set_;
};

}  // namespace fast_vector
