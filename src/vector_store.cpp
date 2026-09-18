#include "fast_vector/vector_store.h"

#include <mutex>
#include <stdexcept>
#include <unordered_set>

#include "fast_vector/distance.h"

namespace fast_vector {

VectorStore::VectorStore(std::unique_ptr<VectorIndex> index, const std::size_t maximum_batch_size)
    : index_(std::move(index)), maximum_batch_size_(maximum_batch_size) {
  if (index_ == nullptr) {
    throw std::invalid_argument("vector store requires an index");
  }
  if (maximum_batch_size_ == 0) {
    throw std::invalid_argument("maximum batch size must be greater than zero");
  }
}

void VectorStore::add(const VectorId id, const std::span<const float> vector) {
  std::unique_lock lock(mutex_);
  index_->add(id, vector);
  inserted_vectors_.fetch_add(1, std::memory_order_relaxed);
}

void VectorStore::add_batch(const std::span<const VectorRecord> vectors) {
  validate_batch(vectors);
  std::unique_lock lock(mutex_);
  for (const VectorRecord& vector : vectors) {
    index_->add(vector.id, vector.values);
    inserted_vectors_.fetch_add(1, std::memory_order_relaxed);
  }
}

std::vector<SearchResult> VectorStore::search(const std::span<const float> query,
                                              const std::size_t k) const {
  try {
    std::shared_lock lock(mutex_);
    std::vector<SearchResult> results = index_->search(query, k);
    successful_queries_.fetch_add(1, std::memory_order_relaxed);
    return results;
  } catch (...) {
    failed_queries_.fetch_add(1, std::memory_order_relaxed);
    throw;
  }
}

StoreStats VectorStore::stats() const {
  std::shared_lock lock(mutex_);
  return StoreStats{
      index_->size(),
      index_->dimension(),
      successful_queries_.load(std::memory_order_relaxed),
      failed_queries_.load(std::memory_order_relaxed),
      inserted_vectors_.load(std::memory_order_relaxed),
  };
}

std::size_t VectorStore::maximum_batch_size() const noexcept { return maximum_batch_size_; }

void VectorStore::validate_batch(const std::span<const VectorRecord> vectors) const {
  if (vectors.size() > maximum_batch_size_) {
    throw std::length_error("batch exceeds maximum batch size");
  }

  std::unordered_set<VectorId> batch_ids;
  batch_ids.reserve(vectors.size());
  const std::size_t expected_dimension = index_->dimension();
  for (const VectorRecord& vector : vectors) {
    if (!batch_ids.insert(vector.id).second) {
      throw std::invalid_argument("batch contains a duplicate vector ID");
    }
    if (vector.values.size() != expected_dimension) {
      throw std::invalid_argument("vector dimension does not match index dimension");
    }
    static_cast<void>(normalize_l2(vector.values));
  }
}

}  // namespace fast_vector
