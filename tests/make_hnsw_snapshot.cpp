#include <exception>
#include <iostream>
#include <vector>

#include "fast_vector/hnsw_index.h"
#include "fast_vector/hnsw_index_io.h"

int main(const int argc, char* argv[]) {
  if (argc != 2) {
    std::cerr << "usage: make_hnsw_snapshot PATH\n";
    return 1;
  }
  try {
    fast_vector::HnswIndex index({
        .dimension = 3,
        .max_connections = 4,
        .ef_construction = 16,
        .ef_search = 16,
        .random_seed = 17,
    });
    index.add(101, std::vector<float>{1.0F, 0.0F, 0.0F});
    index.add(102, std::vector<float>{0.0F, 1.0F, 0.0F});
    index.add(103, std::vector<float>{-1.0F, 0.0F, 0.0F});
    fast_vector::save_hnsw_index(index, argv[1]);
  } catch (const std::exception& error) {
    std::cerr << "failed to create HNSW test snapshot: " << error.what() << '\n';
    return 1;
  }
  return 0;
}
