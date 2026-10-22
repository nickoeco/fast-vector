#pragma once

#include <filesystem>

#include "fast_vector/hnsw_index.h"

namespace fast_vector {

/** Save an HNSW graph, vectors, IDs, and configuration in the versioned binary format. */
void save_hnsw_index(const HnswIndex& index, const std::filesystem::path& path);

/** Durably publish a validated snapshot on Linux; replacement requires overwrite=true. */
void publish_hnsw_index(const HnswIndex& index, const std::filesystem::path& path,
                        bool overwrite = false);

/** Load and validate an HNSW snapshot; the selected kernel is a runtime choice. */
[[nodiscard]] HnswIndex load_hnsw_index(const std::filesystem::path& path,
                                        DotProductKernel kernel = DotProductKernel::Scalar);

}  // namespace fast_vector
