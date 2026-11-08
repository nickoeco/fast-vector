#pragma once

#include <cstddef>
#include <span>
#include <unordered_map>
#include <vector>

#include "fast_vector/distance.h"
#include "fast_vector/vector_index.h"

namespace fast_vector {

class FlatIndexSerializer;
class WalIndexAccess;

/** Exact cosine-similarity index backed by row-major contiguous storage. */
class FlatIndex final : public VectorIndex {
 public:
  explicit FlatIndex(std::size_t dimension, DotProductKernel kernel = DotProductKernel::Scalar);

  void add(VectorId id, std::span<const float> vector) override;

  /** Remove an ID using swap-and-pop; return false if absent. Internal row order may change. */
  [[nodiscard]] bool erase(VectorId id);

  [[nodiscard]] std::vector<SearchResult> search(std::span<const float> query,
                                                 std::size_t k) const override;

  [[nodiscard]] std::size_t size() const noexcept override;
  [[nodiscard]] std::vector<SearchResult> search(std::span<const float> query, std::size_t k,
                                                 const SearchContext& context) const override;
  [[nodiscard]] std::size_t dimension() const noexcept override;
  [[nodiscard]] DotProductKernel kernel() const noexcept;

 private:
  friend class FlatIndexSerializer;
  friend class WalIndexAccess;
  void add_normalized(VectorId id, std::span<const float> normalized);

  std::size_t dimension_;
  DotProductKernel kernel_;
  std::vector<float> vectors_;
  std::vector<VectorId> ids_;
  std::unordered_map<VectorId, std::size_t> id_to_row_;
};

}  // namespace fast_vector
