#include <exception>
#include <iostream>
#include <stdexcept>
#include <string_view>
#include <vector>

#include "fast_vector/hnsw_index.h"
#include "fast_vector/hnsw_index_io.h"

int main(const int argc, char* argv[]) {
  const std::string_view mode = argc == 3 ? argv[2] : "";
  if (argc != 2 && !(argc == 3 && (mode == "--all-deleted" || mode == "--replacement"))) {
    std::cerr << "usage: make_hnsw_snapshot PATH [--all-deleted|--replacement]\n";
    return 1;
  }
  try {
    fast_vector::HnswIndex index({
        .dimension = 3,
        .max_connections = mode == "--replacement" ? 2U : 4U,
        .ef_construction = 16,
        .ef_search = 16,
        .random_seed = 17,
    });
    if (mode == "--replacement") {
      index.add(999, std::vector<float>{0.0F, 0.0F, 1.0F});
      index.add(1000, std::vector<float>{1.0F, 0.0F, 0.0F});
      fast_vector::save_hnsw_index(index, argv[1]);
      return 0;
    }
    index.add(101, std::vector<float>{1.0F, 0.0F, 0.0F});
    index.add(102, std::vector<float>{0.0F, 1.0F, 0.0F});
    index.add(103, std::vector<float>{-1.0F, 0.0F, 0.0F});
    index.add(999, std::vector<float>{0.0F, 0.0F, 1.0F});
    if (!index.erase(999)) {
      throw std::runtime_error("failed to mark test vector deleted");
    }
    if (mode == "--all-deleted") {
      for (const auto id : {101U, 102U, 103U}) {
        if (!index.erase(id)) {
          throw std::runtime_error("failed to prepare all-deleted snapshot");
        }
      }
    }
    fast_vector::save_hnsw_index(index, argv[1]);
  } catch (const std::exception& error) {
    std::cerr << "failed to create HNSW test snapshot: " << error.what() << '\n';
    return 1;
  }
  return 0;
}
