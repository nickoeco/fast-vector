#include "fast_vector/vector_store.h"

#include <mutex>
#include <stdexcept>
#include <unordered_set>

#include "fast_vector/distance.h"

namespace fast_vector {

VectorStore::VectorStore(std::unique_ptr<VectorIndex> index, const std::size_t maximum_batch_size,
                         const bool read_only)
    : writable_index_(std::move(index)),
      maximum_batch_size_(maximum_batch_size),
      read_only_(read_only) {
  if (writable_index_ == nullptr) {
    throw std::invalid_argument("vector store requires an index");
  }
  if (maximum_batch_size_ == 0) {
    throw std::invalid_argument("maximum batch size must be greater than zero");
  }
  if (read_only_) {
    publish_snapshot(std::shared_ptr<const VectorIndex>(std::move(writable_index_)));
  }
}

void VectorStore::add(const VectorId id, const std::span<const float> vector) {
  if (read_only_) {
    throw ReadOnlyStoreError();
  }
  std::unique_lock lock(mutex_);
  writable_index_->add(id, vector);
  inserted_vectors_.fetch_add(1, std::memory_order_relaxed);
}

void VectorStore::add_batch(const std::span<const VectorRecord> vectors) {
  if (read_only_) {
    throw ReadOnlyStoreError();
  }
  validate_batch(vectors);
  std::unique_lock lock(mutex_);
  for (const VectorRecord& vector : vectors) {
    writable_index_->add(vector.id, vector.values);
    inserted_vectors_.fetch_add(1, std::memory_order_relaxed);
  }
}

void VectorStore::replace_index(std::unique_ptr<VectorIndex> index) {
  if (!read_only_) {
    throw std::logic_error("index replacement requires a read-only vector store");
  }
  if (index == nullptr) {
    throw std::invalid_argument("replacement index must not be null");
  }
  // Allocate the control block before publication; failure leaves the old snapshot installed.
  auto replacement = std::shared_ptr<const VectorIndex>(std::move(index));
  publish_snapshot(std::move(replacement));
}

std::shared_ptr<const VectorIndex> VectorStore::acquire_snapshot() const {
#if defined(__cpp_lib_atomic_shared_ptr) && __cpp_lib_atomic_shared_ptr >= 201711L
  return snapshot_.load(std::memory_order_acquire);
#else
  return std::atomic_load_explicit(&snapshot_, std::memory_order_acquire);
#endif
}

void VectorStore::publish_snapshot(std::shared_ptr<const VectorIndex> snapshot) {
#if defined(__cpp_lib_atomic_shared_ptr) && __cpp_lib_atomic_shared_ptr >= 201711L
  snapshot_.store(std::move(snapshot), std::memory_order_release);
#else
  std::atomic_store_explicit(&snapshot_, std::move(snapshot), std::memory_order_release);
#endif
}

std::vector<SearchResult> VectorStore::search(const std::span<const float> query,
                                              const std::size_t k) const {
  try {
    std::vector<SearchResult> results;
    if (read_only_) {
      const auto snapshot = acquire_snapshot();
      results = snapshot->search(query, k);
    } else {
      std::shared_lock lock(mutex_);
      results = writable_index_->search(query, k);
    }
    successful_queries_.fetch_add(1, std::memory_order_relaxed);
    return results;
  } catch (...) {
    failed_queries_.fetch_add(1, std::memory_order_relaxed);
    throw;
  }
}

StoreStats VectorStore::stats() const {
  const auto observe = [this](const VectorIndex& index) {
    return StoreStats{
        index.size(),
        index.dimension(),
        successful_queries_.load(std::memory_order_relaxed),
        failed_queries_.load(std::memory_order_relaxed),
        inserted_vectors_.load(std::memory_order_relaxed),
        read_only_,
    };
  };
  if (read_only_) {
    const auto snapshot = acquire_snapshot();
    return observe(*snapshot);
  }
  std::shared_lock lock(mutex_);
  return observe(*writable_index_);
}

std::size_t VectorStore::maximum_batch_size() const noexcept { return maximum_batch_size_; }

bool VectorStore::read_only() const noexcept { return read_only_; }

void VectorStore::validate_batch(const std::span<const VectorRecord> vectors) const {
  if (vectors.size() > maximum_batch_size_) {
    throw std::length_error("batch exceeds maximum batch size");
  }

  std::unordered_set<VectorId> batch_ids;
  batch_ids.reserve(vectors.size());
  const std::size_t expected_dimension = writable_index_->dimension();
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
