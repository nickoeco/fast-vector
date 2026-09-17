#pragma once

#include <filesystem>

#include "fast_vector/flat_index.h"

namespace fast_vector {

/** Save a FlatIndex using the versioned fast-vector binary format. */
void save_flat_index(const FlatIndex& index, const std::filesystem::path& path);

/** Load and validate a FlatIndex saved by save_flat_index(). */
[[nodiscard]] FlatIndex load_flat_index(
    const std::filesystem::path& path,
    DotProductKernel kernel = DotProductKernel::Scalar);

}  // namespace fast_vector
