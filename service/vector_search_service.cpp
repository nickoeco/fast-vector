#include "service/vector_search_service.h"

#include <chrono>
#include <cstddef>
#include <fstream>
#include <span>
#include <stdexcept>
#include <utility>
#include <vector>

#include "fast_vector/distance.h"

namespace fast_vector::service {
namespace {
constexpr std::array<std::string_view, 17> kStatusNames{
    "OK",        "CANCELLED",      "UNKNOWN",           "INVALID_ARGUMENT",   "DEADLINE_EXCEEDED",
    "NOT_FOUND", "ALREADY_EXISTS", "PERMISSION_DENIED", "RESOURCE_EXHAUSTED", "FAILED_PRECONDITION",
    "ABORTED",   "OUT_OF_RANGE",   "UNIMPLEMENTED",     "INTERNAL",           "UNAVAILABLE",
    "DATA_LOSS", "UNAUTHENTICATED"};
}

VectorSearchService::VectorSearchService(std::shared_ptr<VectorStore> store, std::string index_type,
                                         std::shared_ptr<StructuredLogger> logger,
                                         SearchExecutorConfig search_config,
                                         PublicationConfig publication_config)
    : store_(std::move(store)), index_type_(std::move(index_type)), logger_(std::move(logger)) {
  if (store_ == nullptr) {
    throw std::invalid_argument("gRPC vector service requires a vector store");
  }
  if (index_type_.empty()) {
    throw std::invalid_argument("gRPC vector service requires an index type");
  }
  executor_ = std::make_unique<SearchExecutor>(store_, search_config);
  publication_ =
      std::make_unique<SnapshotPublication>(store_, index_type_, std::move(publication_config));
}

VectorSearchService::RpcCall VectorSearchService::begin_rpc(
    const std::string_view method) noexcept {
  return {next_request_id_.fetch_add(1, std::memory_order_relaxed),
          std::chrono::steady_clock::now(), method};
}

grpc::Status VectorSearchService::finish_rpc(const RpcCall& call, grpc::Status status) {
  auto code = static_cast<std::size_t>(status.error_code());
  if (code >= kStatusNames.size()) {
    code = grpc::StatusCode::UNKNOWN;
  }
  status_counts_[code].fetch_add(1, std::memory_order_relaxed);
  if (logger_) {
    const auto elapsed = std::chrono::duration_cast<std::chrono::microseconds>(
                             std::chrono::steady_clock::now() - call.started)
                             .count();
    std::optional<std::uint64_t> count;
    try {
      count = store_->try_size();
    } catch (...) {
      // Observability must not replace the original handler outcome.
    }
    logger_->write(
        {.severity = status.ok() ? "INFO" : (code == grpc::StatusCode::INTERNAL ? "ERROR" : "WARN"),
         .event = "rpc_completed",
         .request_id = call.id,
         .method = call.method,
         .status = kStatusNames[code],
         .latency_us = static_cast<std::uint64_t>(elapsed),
         .index_type = index_type_,
         .vector_count = count},
        {{"status_code", static_cast<std::uint64_t>(code)}});
  }
  return status;
}

void VectorSearchService::log_rpc_summary() noexcept {
  if (!logger_ || summary_logged_.exchange(true, std::memory_order_relaxed)) {
    return;
  }
  std::optional<std::uint64_t> count;
  try {
    count = store_->stats().index_size;
  } catch (...) {
  }
  std::uint64_t total = 0;
  std::uint64_t errors = 0;
  const auto successful = status_counts_[0].load(std::memory_order_relaxed);
  for (std::size_t code = 0; code < status_counts_.size(); ++code) {
    const auto completed = status_counts_[code].load(std::memory_order_relaxed);
    total += completed;
    if (code != 0) {
      errors += completed;
      if (completed != 0) {
        logger_->write({.event = "rpc_error_summary",
                        .status = kStatusNames[code],
                        .index_type = index_type_,
                        .vector_count = count},
                       {{"status_code", static_cast<std::uint64_t>(code)}, {"count", completed}});
      }
    }
  }
  logger_->write({.event = "rpc_summary", .index_type = index_type_, .vector_count = count},
                 {{"completed_rpcs", total},
                  {"successful_rpcs", successful},
                  {"failed_rpcs", errors},
                  {"dropped_log_records", logger_->dropped_records()}});
  const auto scheduler = executor_->stats();
  logger_->write(
      {.event = "search_executor_summary", .index_type = index_type_, .vector_count = count},
      {{"admitted", scheduler.admitted},
       {"rejected", scheduler.rejected},
       {"completed", scheduler.completed},
       {"succeeded", scheduler.succeeded},
       {"failed", scheduler.failed},
       {"cancelled", scheduler.cancelled},
       {"expired", scheduler.expired},
       {"peak_queued", static_cast<std::uint64_t>(scheduler.peak_queued)},
       {"peak_running", static_cast<std::uint64_t>(scheduler.peak_running)},
       {"queue_us_total", scheduler.queue_us_total},
       {"execution_us_total", scheduler.execution_us_total}});
}

grpc::Status VectorSearchService::AddVector(grpc::ServerContext* context,
                                            const v1::AddVectorRequest* request,
                                            v1::AddVectorResponse* response) {
  const auto call = begin_rpc("AddVector");
  if (context->IsCancelled()) {
    return finish_rpc(call, cancelled_status(*context));
  }
  try {
    const v1::Vector& vector = request->vector();
    const auto result = store_->add(
        vector.id(), std::span<const float>(vector.values().data(), vector.values().size()),
        request->client_operation_id());
    response->set_index_size(result.index_size);
    response->set_lsn(result.lsn);
    response->set_replayed(result.replayed);
    return finish_rpc(call, grpc::Status::OK);
  } catch (const std::exception& error) {
    return finish_rpc(call, exception_status(error));
  }
}

grpc::Status VectorSearchService::Checkpoint(grpc::ServerContext* context,
                                             const v1::CheckpointRequest*,
                                             v1::CheckpointResponse* response) {
  const auto call = begin_rpc("Checkpoint");
  if (context->IsCancelled()) return finish_rpc(call, cancelled_status(*context));
  try {
    response->set_lsn(store_->checkpoint());
    return finish_rpc(call, grpc::Status::OK);
  } catch (const std::exception& error) {
    return finish_rpc(call, exception_status(error));
  }
}

grpc::Status VectorSearchService::BatchAdd(grpc::ServerContext* context,
                                           const v1::BatchAddRequest* request,
                                           v1::BatchAddResponse* response) {
  const auto call = begin_rpc("BatchAdd");
  if (context->IsCancelled()) {
    return finish_rpc(call, cancelled_status(*context));
  }
  try {
    std::vector<VectorRecord> vectors;
    vectors.reserve(static_cast<std::size_t>(request->vectors_size()));
    for (const v1::Vector& vector : request->vectors()) {
      vectors.push_back(VectorRecord{
          vector.id(),
          std::vector<float>(vector.values().begin(), vector.values().end()),
      });
    }
    store_->add_batch(vectors);
    const StoreStats stats = store_->stats();
    response->set_added_count(vectors.size());
    response->set_index_size(stats.index_size);
    return finish_rpc(call, grpc::Status::OK);
  } catch (const std::exception& error) {
    return finish_rpc(call, exception_status(error));
  }
}

grpc::Status VectorSearchService::Search(grpc::ServerContext* context,
                                         const v1::SearchRequest* request,
                                         v1::SearchResponse* response) {
  const auto call = begin_rpc("Search");
  if (context->IsCancelled()) {
    return finish_rpc(call, cancelled_status(*context));
  }
  try {
    const auto query = std::span<const float>(request->query().data(), request->query().size());
    if (request->k() != 0) {
      if (request->expected_snapshot_id().empty() && query.size() != store_->dimension()) {
        throw std::invalid_argument("query dimension does not match index dimension");
      }
      (void)normalize_l2(query);
    }
    if (request->expected_snapshot_id().size() > 256)
      throw std::invalid_argument("expected snapshot ID too long");
    auto deadline = SearchContext::Clock::time_point::max();
    if (context->deadline() != std::chrono::system_clock::time_point::max()) {
      const auto remaining = context->deadline() - std::chrono::system_clock::now();
      const auto now = SearchContext::Clock::now();
      if (remaining <= decltype(remaining)::zero()) {
        return finish_rpc(call, {grpc::StatusCode::DEADLINE_EXCEEDED, "request deadline exceeded"});
      }
      if (remaining < deadline - now) deadline = now + remaining;
    }
    auto ticket = executor_->submit(std::vector<float>(query.begin(), query.end()), request->k(),
                                    deadline, request->expected_snapshot_id());
    while (ticket.future.wait_for(std::chrono::milliseconds(2)) != std::future_status::ready) {
      if (context->IsCancelled()) {
        executor_->cancel(ticket.cancellation,
                          std::chrono::system_clock::now() >= context->deadline()
                              ? SearchStopReason::DeadlineExceeded
                              : SearchStopReason::Cancelled);
      }
    }
    auto outcome = ticket.future.get();
    if (outcome.error) std::rethrow_exception(outcome.error);
    if (context->IsCancelled()) {
      return finish_rpc(call, cancelled_status(*context));
    }
    for (const SearchResult& result : outcome.result.neighbors) {
      v1::Neighbor* neighbor = response->add_neighbors();
      neighbor->set_id(result.id);
      neighbor->set_score(result.score);
    }
    response->set_snapshot_version(outcome.result.snapshot_version);
    response->set_queue_us(outcome.queue_us);
    response->set_execution_us(outcome.execution_us);
    response->set_snapshot_id(outcome.result.identity.snapshot_id);
    response->set_generation(outcome.result.identity.generation);
    response->set_embedding_revision(outcome.result.identity.embedding_revision);
    return finish_rpc(call, grpc::Status::OK);
  } catch (const std::exception& error) {
    return finish_rpc(call, exception_status(error));
  } catch (...) {
    return finish_rpc(call, {grpc::StatusCode::INTERNAL, "internal vector service error"});
  }
}

grpc::Status VectorSearchService::PrepareSnapshot(
    grpc::ServerContext* context, grpc::ServerReader<v1::PrepareSnapshotRequest>* reader,
    v1::PublicationStatus* response) {
  const auto call = begin_rpc("PrepareSnapshot");
  std::uint64_t token = 0;
  v1::PrepareSnapshotRequest frame;
  v1::SnapshotManifest manifest;
  std::uint64_t generation = 0;
  try {
    if (!publication_->config().enabled) throw PublicationDisabledError();
    if (!reader->Read(&frame) || !frame.has_manifest() || !frame.data().empty() || frame.abort()) {
      throw std::invalid_argument("first frame requires manifest and generation only");
    }
    manifest = frame.manifest();
    generation = frame.generation();
    token = publication_->begin(manifest, generation);
    if (logger_)
      logger_->write(
          {.event = "snapshot_prepare_started", .index_type = index_type_},
          {{"generation", generation}, {"snapshot_id", std::string_view(manifest.snapshot_id())}});
    SnapshotStagingFile staging(publication_->config().staging_directory);
    std::ofstream output(staging.path(), std::ios::binary | std::ios::trunc);
    if (!output) throw std::runtime_error("snapshot staging unavailable");
    std::uint64_t received = 0;
    while (reader->Read(&frame)) {
      if (context->IsCancelled()) {
        publication_->fail(token);
        return finish_rpc(call, cancelled_status(*context));
      }
      if (frame.abort() || frame.has_manifest() || frame.generation() != 0 ||
          frame.data().empty() || frame.data().size() > 256 * 1024 ||
          frame.data().size() > manifest.file_size() - received) {
        throw std::invalid_argument("invalid snapshot data frame or length");
      }
      output.write(frame.data().data(), static_cast<std::streamsize>(frame.data().size()));
      if (!output) throw std::runtime_error("snapshot staging write failed");
      received += frame.data().size();
    }
    if (context->IsCancelled()) {
      publication_->fail(token);
      return finish_rpc(call, cancelled_status(*context));
    }
    output.flush();
    if (!output || received != manifest.file_size())
      throw std::invalid_argument("snapshot length mismatch");
    output.close();
    if (!output) throw std::runtime_error("snapshot staging close failed");
    publication_->complete(staging.path(), token);
    *response = publication_->status();
    if (logger_)
      logger_->write({.event = "snapshot_ready", .status = "OK", .index_type = index_type_},
                     {{"generation", generation}, {"prepare_us", response->prepare_us()}});
    return finish_rpc(call, grpc::Status::OK);
  } catch (const std::exception& error) {
    if (token)
      publication_->fail(token);
    else if (publication_->config().enabled) {
      try {
        publication_->rejected(manifest, generation);
      } catch (...) {
      }
    }
    if (logger_ && token)
      logger_->write(
          {.severity = "ERROR", .event = "snapshot_prepare_failed", .index_type = index_type_},
          {{"generation", generation}});
    if (dynamic_cast<const std::bad_alloc*>(&error)) {
      return finish_rpc(call,
                        {grpc::StatusCode::RESOURCE_EXHAUSTED, "snapshot resources unavailable"});
    }
    if (dynamic_cast<const PublicationBusyError*>(&error) ||
        dynamic_cast<const SnapshotMismatchError*>(&error) ||
        dynamic_cast<const ReadOnlyStoreError*>(&error) ||
        dynamic_cast<const PublicationDisabledError*>(&error))
      return finish_rpc(call, exception_status(error));
    return finish_rpc(
        call, {token ? grpc::StatusCode::FAILED_PRECONDITION : grpc::StatusCode::INVALID_ARGUMENT,
               "snapshot preparation failed"});
  }
}

grpc::Status VectorSearchService::ActivateSnapshot(grpc::ServerContext* context,
                                                   const v1::ActivateSnapshotRequest* request,
                                                   v1::PublicationStatus* response) {
  const auto call = begin_rpc("ActivateSnapshot");
  if (context->IsCancelled()) return finish_rpc(call, cancelled_status(*context));
  try {
    if (request->snapshot_id().size() > 256) throw std::invalid_argument("snapshot ID too long");
    publication_->activate(request->snapshot_id(), request->generation());
    *response = publication_->status();
    if (logger_)
      logger_->write(
          {.event = "snapshot_activated", .status = "OK", .index_type = index_type_},
          {{"generation", request->generation()}, {"activation_us", response->activation_us()}});
    return finish_rpc(call, grpc::Status::OK);
  } catch (const std::exception& error) {
    return finish_rpc(call, exception_status(error));
  }
}

grpc::Status VectorSearchService::GetStats(grpc::ServerContext* context, const v1::GetStatsRequest*,
                                           v1::GetStatsResponse* response) {
  const auto call = begin_rpc("GetStats");
  if (context->IsCancelled()) {
    return finish_rpc(call, cancelled_status(*context));
  }
  try {
    StoreStats stats{};
    auto publication = publication_->status(&stats);
    response->set_index_type(index_type_);
    response->set_vector_count(stats.index_size);
    response->set_dimension(stats.dimension);
    response->set_successful_queries(stats.successful_queries);
    response->set_failed_queries(stats.failed_queries);
    response->set_inserted_vectors(stats.inserted_vectors);
    response->set_read_only(stats.read_only);
    response->set_snapshot_version(stats.snapshot_version);
    *response->mutable_publication() = std::move(publication);
    const auto& durability = stats.durability;
    auto* durable = response->mutable_durability();
    durable->set_enabled(durability.enabled);
    durable->set_recovery_required(durability.recovery_required);
    durable->set_applied_lsn(durability.applied_lsn);
    durable->set_checkpoint_lsn(durability.checkpoint_lsn);
    durable->set_recovered_records(durability.recovered_records);
    durable->set_discarded_tail_bytes(durability.discarded_tail_bytes);
    durable->set_recovery_us(durability.recovery_us);
    durable->set_last_sync_us(durability.last_sync_us);
    durable->set_last_write_us(durability.last_write_us);
    durable->set_last_checkpoint_us(durability.last_checkpoint_us);
    const auto scheduler = executor_->stats();
    auto* scheduling = response->mutable_search_executor();
    scheduling->set_workers(scheduler.workers);
    scheduling->set_queue_capacity(scheduler.queue_capacity);
    scheduling->set_queued(scheduler.queued);
    scheduling->set_running(scheduler.running);
    scheduling->set_peak_queued(scheduler.peak_queued);
    scheduling->set_peak_running(scheduler.peak_running);
    scheduling->set_admitted(scheduler.admitted);
    scheduling->set_rejected(scheduler.rejected);
    scheduling->set_started(scheduler.started);
    scheduling->set_completed(scheduler.completed);
    scheduling->set_succeeded(scheduler.succeeded);
    scheduling->set_failed(scheduler.failed);
    scheduling->set_cancelled(scheduler.cancelled);
    scheduling->set_expired(scheduler.expired);
    scheduling->set_queue_us_total(scheduler.queue_us_total);
    scheduling->set_execution_us_total(scheduler.execution_us_total);
    return finish_rpc(call, grpc::Status::OK);
  } catch (const std::exception& error) {
    return finish_rpc(call, exception_status(error));
  }
}

grpc::Status VectorSearchService::cancelled_status(const grpc::ServerContext& context) {
  if (std::chrono::system_clock::now() >= context.deadline()) {
    return {grpc::StatusCode::DEADLINE_EXCEEDED, "request deadline exceeded"};
  }
  return {grpc::StatusCode::CANCELLED, "request cancelled"};
}

grpc::Status VectorSearchService::exception_status(const std::exception& error) {
  if (dynamic_cast<const RecoveryRequiredError*>(&error)) {
    return {grpc::StatusCode::FAILED_PRECONDITION, "durable writer requires recovery"};
  }
  if (dynamic_cast<const DurabilityModeError*>(&error)) {
    return {grpc::StatusCode::FAILED_PRECONDITION, error.what()};
  }
  if (dynamic_cast<const PublicationDisabledError*>(&error)) {
    return {grpc::StatusCode::FAILED_PRECONDITION, "snapshot publication is disabled"};
  }
  if (dynamic_cast<const SnapshotMismatchError*>(&error)) {
    return {grpc::StatusCode::FAILED_PRECONDITION, "snapshot identity or generation mismatch"};
  }
  if (dynamic_cast<const PublicationBusyError*>(&error)) {
    return {grpc::StatusCode::RESOURCE_EXHAUSTED, "snapshot preparation already in progress"};
  }
  if (const auto* rejected = dynamic_cast<const SearchRejected*>(&error)) {
    return {
        rejected->stopping() ? grpc::StatusCode::UNAVAILABLE : grpc::StatusCode::RESOURCE_EXHAUSTED,
        rejected->stopping() ? "search executor stopping" : "search queue full"};
  }
  if (const auto* interrupted = dynamic_cast<const SearchInterrupted*>(&error)) {
    switch (interrupted->reason()) {
      case SearchStopReason::DeadlineExceeded:
        return {grpc::StatusCode::DEADLINE_EXCEEDED, "search deadline exceeded"};
      case SearchStopReason::Shutdown:
        return {grpc::StatusCode::UNAVAILABLE, "search executor stopping"};
      default:
        return {grpc::StatusCode::CANCELLED, "search cancelled"};
    }
  }
  if (dynamic_cast<const ReadOnlyStoreError*>(&error) != nullptr) {
    return {grpc::StatusCode::FAILED_PRECONDITION, error.what()};
  }
  if (dynamic_cast<const std::length_error*>(&error) != nullptr) {
    return {grpc::StatusCode::RESOURCE_EXHAUSTED, error.what()};
  }
  if (dynamic_cast<const std::invalid_argument*>(&error) != nullptr) {
    return {grpc::StatusCode::INVALID_ARGUMENT, error.what()};
  }
  return {grpc::StatusCode::INTERNAL, "internal vector service error"};
}

}  // namespace fast_vector::service
