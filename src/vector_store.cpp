#include "fast_vector/vector_store.h"

#include <limits>
#include <mutex>
#include <stdexcept>
#include <unordered_set>

#include "fast_vector/distance.h"
#include "fast_vector/hnsw_index.h"

namespace fast_vector {

VectorStore::VectorStore(std::unique_ptr<VectorIndex> index, const std::size_t maximum_batch_size,
                         const bool read_only, SnapshotIdentity identity, ResourceLimits limits,
                         std::shared_ptr<MemoryBudget> budget)
    : limits_(limits),
      budget_(budget ? std::move(budget)
                     : std::make_shared<MemoryBudget>(2ULL * 1024 * 1024 * 1024)),
      writable_index_(std::move(index)),
      maximum_batch_size_(maximum_batch_size),
      read_only_(read_only) {
  if (writable_index_ == nullptr) {
    throw std::invalid_argument("vector store requires an index");
  }
  if (maximum_batch_size_ == 0) {
    throw std::invalid_argument("maximum batch size must be greater than zero");
  }
  check_growth(0);
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

void VectorStore::validate_top_k(std::size_t k) const {
  if (k > limits_.maximum_top_k) throw ResourceExhausted();
}
std::shared_ptr<const VectorIndex> VectorStore::account_index(
    std::shared_ptr<const VectorIndex> index) const {
  struct Owner {
    std::shared_ptr<void> lease;
    std::shared_ptr<const VectorIndex> index;
  };
  auto owner = std::make_shared<Owner>(Owner{reserve_index(*index), std::move(index)});
  return std::shared_ptr<const VectorIndex>(owner, owner->index.get());
}
void VectorStore::check_growth(std::size_t additions) const {
  auto count = writable_index_->size();
  if (const auto* hnsw = dynamic_cast<const HnswIndex*>(writable_index_.get()))
    count = hnsw->stats().node_count;
  check_shape(limits_, writable_index_->dimension(), resource_add(count, additions));
  if (const auto* hnsw = dynamic_cast<const HnswIndex*>(writable_index_.get())) {
    const auto per_node = resource_add(
        resource_add(resource_multiply(hnsw->dimension(), 4), 512),
        resource_multiply(64,
                          resource_add(48, resource_multiply(hnsw->config().max_connections, 32))));
    if (resource_multiply(resource_add(count, additions), per_node) > limits_.maximum_index_bytes)
      throw ResourceExhausted();
  }
}
std::shared_ptr<void> VectorStore::reserve_index(const VectorIndex& index) const {
  auto count = index.size();
  std::uint64_t extra = 0;
  if (const auto* hnsw = dynamic_cast<const HnswIndex*>(&index)) {
    const auto stats = hnsw->stats();
    count = stats.node_count;
    extra = resource_add(
        resource_multiply(stats.directed_edge_count, 16),
        resource_multiply(count, resource_multiply(resource_add(stats.max_level, 1), 48)));
  }
  check_shape(limits_, index.dimension(), count);
  auto bytes = resource_add(
      resource_multiply(count, resource_add(resource_multiply(index.dimension(), 4), 512)), extra);
  if (bytes > limits_.maximum_index_bytes) throw ResourceExhausted();
  return budget_->reserve(bytes);
}
void VectorStore::add(const VectorId id, const std::span<const float> vector) {
  if (read_only_) {
    throw ReadOnlyStoreError();
  }
  std::unique_lock lock(mutex_);
  check_growth(1);
  writable_index_->add(id, vector);
  inserted_vectors_.fetch_add(1, std::memory_order_relaxed);
}

CommitResult VectorStore::add(const VectorId id, const std::span<const float> vector,
                              const std::string& operation_id) {
  if (read_only_) throw ReadOnlyStoreError();
  std::unique_lock lock(mutex_);
  if (auto* durable = dynamic_cast<DurableFlatIndex*>(writable_index_.get())) {
    auto result = durable->commit(operation_id, id, vector, &limits_);
    if (!result.replayed) inserted_vectors_.fetch_add(1, std::memory_order_relaxed);
    return result;
  }
  if (!operation_id.empty()) throw std::invalid_argument("client operation IDs require WAL mode");
  check_growth(1);
  writable_index_->add(id, vector);
  inserted_vectors_.fetch_add(1, std::memory_order_relaxed);
  return {0, writable_index_->size(), false};
}

std::uint64_t VectorStore::checkpoint() {
  if (read_only_) throw ReadOnlyStoreError();
  std::unique_lock lock(mutex_);
  auto* durable = dynamic_cast<DurableFlatIndex*>(writable_index_.get());
  if (!durable) throw DurabilityModeError("checkpoint requires WAL mode");
  return durable->checkpoint();
}

DurabilityStats VectorStore::durability_stats() const {
  if (read_only_) return {};
  std::shared_lock lock(mutex_);
  const auto* durable = dynamic_cast<const DurableFlatIndex*>(writable_index_.get());
  return durable ? durable->durability_stats() : DurabilityStats{};
}

void VectorStore::add_batch(const std::span<const VectorRecord> vectors) {
  if (read_only_) {
    throw ReadOnlyStoreError();
  }
  validate_batch(vectors);
  std::unique_lock lock(mutex_);
  if (dynamic_cast<DurableFlatIndex*>(writable_index_.get()))
    throw DurabilityModeError("batch insertion is not supported in WAL mode");
  check_growth(vectors.size());
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
      ServingSnapshot{account_index(std::move(snapshot)), version + 1, std::move(identity)});
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
      ServingSnapshot{account_index(std::shared_ptr<const VectorIndex>(std::move(index))),
                      current->version + 1, std::move(identity)});
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
    validate_top_k(k);
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
    const auto* durable = dynamic_cast<const DurableFlatIndex*>(&index);
    return StoreStats{
        index.size(),
        index.dimension(),
        successful_queries_.load(std::memory_order_relaxed),
        failed_queries_.load(std::memory_order_relaxed),
        inserted_vectors_.load(std::memory_order_relaxed),
        read_only_,
        version,
        identity,
        durable ? durable->durability_stats() : DurabilityStats{},
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
