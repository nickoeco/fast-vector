#include "service/vector_search_service.h"

#include <chrono>
#include <cstddef>
#include <span>
#include <stdexcept>
#include <utility>
#include <vector>

namespace fast_vector::service {
namespace {
constexpr std::array<std::string_view, 17> kStatusNames{
    "OK",        "CANCELLED",      "UNKNOWN",           "INVALID_ARGUMENT",   "DEADLINE_EXCEEDED",
    "NOT_FOUND", "ALREADY_EXISTS", "PERMISSION_DENIED", "RESOURCE_EXHAUSTED", "FAILED_PRECONDITION",
    "ABORTED",   "OUT_OF_RANGE",   "UNIMPLEMENTED",     "INTERNAL",           "UNAVAILABLE",
    "DATA_LOSS", "UNAUTHENTICATED"};
}

VectorSearchService::VectorSearchService(std::shared_ptr<VectorStore> store, std::string index_type,
                                         std::shared_ptr<StructuredLogger> logger)
    : store_(std::move(store)), index_type_(std::move(index_type)), logger_(std::move(logger)) {
  if (store_ == nullptr) {
    throw std::invalid_argument("gRPC vector service requires a vector store");
  }
  if (index_type_.empty()) {
    throw std::invalid_argument("gRPC vector service requires an index type");
  }
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
      count = store_->stats().index_size;
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
    store_->add(vector.id(),
                std::span<const float>(vector.values().data(), vector.values().size()));
    response->set_index_size(store_->stats().index_size);
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
    const std::vector<SearchResult> results = store_->search(
        std::span<const float>(request->query().data(), request->query().size()), request->k());
    if (context->IsCancelled()) {
      return finish_rpc(call, cancelled_status(*context));
    }
    for (const SearchResult& result : results) {
      v1::Neighbor* neighbor = response->add_neighbors();
      neighbor->set_id(result.id);
      neighbor->set_score(result.score);
    }
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
    const StoreStats stats = store_->stats();
    response->set_index_type(index_type_);
    response->set_vector_count(stats.index_size);
    response->set_dimension(stats.dimension);
    response->set_successful_queries(stats.successful_queries);
    response->set_failed_queries(stats.failed_queries);
    response->set_inserted_vectors(stats.inserted_vectors);
    response->set_read_only(stats.read_only);
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
