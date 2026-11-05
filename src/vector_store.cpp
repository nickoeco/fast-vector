#include "fast_vector/vector_store.h"

#include <limits>
#include <mutex>
#include <stdexcept>
#include <unordered_set>

#include "fast_vector/distance.h"

namespace fast_vector {

VectorStore::VectorStore(std::unique_ptr<VectorIndex> index, const std::size_t maximum_batch_size,
                         const bool read_only, SnapshotIdentity identity)
    : writable_index_(std::move(index)),
      maximum_batch_size_(maximum_batch_size),
      read_only_(read_only) {
  if (writable_index_ == nullptr) {
    throw std::invalid_argument("vector store requires an index");
  }
  if (maximum_batch_size_ == 0) {
    throw std::invalid_argument("maximum batch size must be greater than zero");
  }
  if ((identity.snapshot_id.empty() != (identity.generation == 0)) ||
      (identity.snapshot_id.empty() && !identity.embedding_revision.empty()) ||
      identity.snapshot_id.size() > 256 || identity.embedding_revision.size() > 256) {
    throw std::invalid_argument("invalid initial snapshot identity");
  }
  if (read_only_) {
    publish_snapshot(std::shared_ptr<const VectorIndex>(std::move(writable_index_)),
                     std::move(identity));
  } else if (!identity.snapshot_id.empty() || identity.generation != 0 ||
             !identity.embedding_revision.empty()) {
    throw std::invalid_argument("snapshot identity requires read-only mode");
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

std::shared_ptr<const ServingSnapshot> VectorStore::acquire_snapshot() const {
#if defined(__cpp_lib_atomic_shared_ptr) && __cpp_lib_atomic_shared_ptr >= 201711L
  return snapshot_.load(std::memory_order_acquire);
#else
  return std::atomic_load_explicit(&snapshot_, std::memory_order_acquire);
#endif
}

void VectorStore::publish_snapshot(std::shared_ptr<const VectorIndex> snapshot,
                                   SnapshotIdentity identity) {
  // Publish version and index in one immutable object, never as separate atomics.
  std::lock_guard lock(publication_mutex_);
  const auto previous = acquire_snapshot();
  if (previous && previous->identity.generation != 0) throw SnapshotMismatchError();
  const auto version = previous ? previous->version : 0;
  if (version == std::numeric_limits<std::uint64_t>::max()) {
    throw std::length_error("snapshot version exhausted");
  }
  std::shared_ptr<const ServingSnapshot> publication = std::make_shared<ServingSnapshot>(
      ServingSnapshot{std::move(snapshot), version + 1, std::move(identity)});
#if defined(__cpp_lib_atomic_shared_ptr) && __cpp_lib_atomic_shared_ptr >= 201711L
  snapshot_.store(std::move(publication), std::memory_order_release);
#else
  std::atomic_store_explicit(&snapshot_, std::move(publication), std::memory_order_release);
#endif
}

std::shared_ptr<const ServingSnapshot> VectorStore::serving_snapshot() const {
  if (!read_only_) throw ReadOnlyStoreError();
  return acquire_snapshot();
}

std::shared_ptr<const ServingSnapshot> VectorStore::prepare_snapshot(
    std::unique_ptr<VectorIndex> index, SnapshotIdentity identity) {
  if (!read_only_) throw ReadOnlyStoreError();
  if (!index || identity.snapshot_id.empty() || identity.generation == 0 ||
      identity.snapshot_id.size() > 256 || identity.embedding_revision.size() > 256) {
    throw std::invalid_argument("invalid serving snapshot identity");
  }
  const auto current = acquire_snapshot();
  if (identity.generation <= current->identity.generation) throw SnapshotMismatchError();
  if (current->version == std::numeric_limits<std::uint64_t>::max()) {
    throw std::length_error("snapshot version exhausted");
  }
  return std::make_shared<ServingSnapshot>(
      ServingSnapshot{std::shared_ptr<const VectorIndex>(std::move(index)), current->version + 1,
                      std::move(identity)});
}

void VectorStore::activate_snapshot(std::shared_ptr<const ServingSnapshot> snapshot) {
  if (!read_only_) throw ReadOnlyStoreError();
  if (!snapshot || !snapshot->index || snapshot->identity.snapshot_id.empty() ||
      snapshot->identity.generation == 0)
    throw std::invalid_argument("invalid serving snapshot");
  std::lock_guard lock(publication_mutex_);
  const auto current = acquire_snapshot();
  if (snapshot->identity.generation <= current->identity.generation ||
      snapshot->version != current->version + 1)
    throw SnapshotMismatchError();
#if defined(__cpp_lib_atomic_shared_ptr) && __cpp_lib_atomic_shared_ptr >= 201711L
  snapshot_.store(std::move(snapshot), std::memory_order_release);
#else
  std::atomic_store_explicit(&snapshot_, std::move(snapshot), std::memory_order_release);
#endif
}

std::vector<SearchResult> VectorStore::search(const std::span<const float> query,
                                              const std::size_t k) const {
  return search(query, k, SearchContext{}).neighbors;
}

SnapshotSearchResult VectorStore::search(const std::span<const float> query, const std::size_t k,
                                         const SearchContext& context) const {
  try {
    context.validate();
    context.check();
    std::vector<SearchResult> results;
    std::uint64_t version = 0;
    SnapshotIdentity identity;
    if (read_only_) {
      const auto snapshot = acquire_snapshot();
      version = snapshot->version;
      if (!context.expected_snapshot_id.empty() &&
          context.expected_snapshot_id != snapshot->identity.snapshot_id)
        throw SnapshotMismatchError();
      identity = snapshot->identity;
      results = snapshot->index->search(query, k, context);
    } else {
      if (!context.expected_snapshot_id.empty()) throw SnapshotMismatchError();
      std::shared_lock lock(mutex_, std::defer_lock);
      if (context.enabled()) {
        while (!lock.try_lock_for(std::chrono::milliseconds(2))) context.check();
      } else {
        lock.lock();
      }
      context.check();
      results = writable_index_->search(query, k, context);
    }
    context.check();
    successful_queries_.fetch_add(1, std::memory_order_relaxed);
    return {std::move(results), version, std::move(identity)};
  } catch (...) {
    failed_queries_.fetch_add(1, std::memory_order_relaxed);
    throw;
  }
}

StoreStats VectorStore::stats() const {
  const auto observe = [this](const VectorIndex& index, const std::uint64_t version,
                              const SnapshotIdentity& identity = {}) {
    return StoreStats{
        index.size(),
        index.dimension(),
        successful_queries_.load(std::memory_order_relaxed),
        failed_queries_.load(std::memory_order_relaxed),
        inserted_vectors_.load(std::memory_order_relaxed),
        read_only_,
        version,
        identity,
    };
  };
  if (read_only_) {
    const auto snapshot = acquire_snapshot();
    return observe(*snapshot->index, snapshot->version, snapshot->identity);
  }
  std::shared_lock lock(mutex_);
  return observe(*writable_index_, 0);
}

std::size_t VectorStore::maximum_batch_size() const noexcept { return maximum_batch_size_; }

std::size_t VectorStore::dimension() const {
  if (read_only_) return acquire_snapshot()->index->dimension();
  // VectorIndex's fixed-dimension contract makes this independent of insertions.
  return writable_index_->dimension();
}

std::optional<std::size_t> VectorStore::try_size() const {
  if (read_only_) return acquire_snapshot()->index->size();
  std::shared_lock lock(mutex_, std::try_to_lock);
  if (!lock.owns_lock()) return std::nullopt;
  return writable_index_->size();
}

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
