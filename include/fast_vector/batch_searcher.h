#pragma once

#include <cstddef>
#include <memory>
#include <span>
#include <vector>

#include "fast_vector/vector_index.h"

namespace fast_vector {

/**
 * Reusable fixed-size worker pool for ordered batch queries.
 *
 * The referenced index must outlive this object and must not be mutated while searches
 * are running. Queries are a row-major query_count-by-dimension matrix.
 */
class BatchSearcher {
public:
    BatchSearcher(const VectorIndex& index, std::size_t thread_count);
    ~BatchSearcher();

    BatchSearcher(const BatchSearcher&) = delete;
    BatchSearcher& operator=(const BatchSearcher&) = delete;
    BatchSearcher(BatchSearcher&&) = delete;
    BatchSearcher& operator=(BatchSearcher&&) = delete;

    /** Search a batch while preserving query order in the returned outer vector. */
    [[nodiscard]] std::vector<std::vector<SearchResult>> search(
        std::span<const float> queries, std::size_t query_count, std::size_t k) const;

    [[nodiscard]] std::size_t thread_count() const noexcept;

private:
    class Impl;
    std::unique_ptr<Impl> impl_;
};

}  // namespace fast_vector
