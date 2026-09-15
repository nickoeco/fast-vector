#pragma once

#include <cstdint>

namespace fast_vector {

using VectorId = std::uint64_t;

/** A neighbor returned by a similarity search. */
struct SearchResult {
    VectorId id;
    float score;

    bool operator==(const SearchResult&) const = default;
};

}  // namespace fast_vector
