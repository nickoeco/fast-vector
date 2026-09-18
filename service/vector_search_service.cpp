#include "service/vector_search_service.h"

#include <chrono>
#include <cstddef>
#include <span>
#include <stdexcept>
#include <utility>
#include <vector>

namespace fast_vector::service {

VectorSearchService::VectorSearchService(std::shared_ptr<VectorStore> store, std::string index_type)
    : store_(std::move(store)), index_type_(std::move(index_type)) {
  if (store_ == nullptr) {
    throw std::invalid_argument("gRPC vector service requires a vector store");
  }
  if (index_type_.empty()) {
    throw std::invalid_argument("gRPC vector service requires an index type");
  }
}

grpc::Status VectorSearchService::AddVector(grpc::ServerContext* context,
                                            const v1::AddVectorRequest* request,
                                            v1::AddVectorResponse* response) {
  if (context->IsCancelled()) {
    return cancelled_status(*context);
  }
  try {
    const v1::Vector& vector = request->vector();
    store_->add(vector.id(),
                std::span<const float>(vector.values().data(), vector.values().size()));
    response->set_index_size(store_->stats().index_size);
    return grpc::Status::OK;
  } catch (const std::exception& error) {
    return exception_status(error);
  }
}

grpc::Status VectorSearchService::BatchAdd(grpc::ServerContext* context,
                                           const v1::BatchAddRequest* request,
                                           v1::BatchAddResponse* response) {
  if (context->IsCancelled()) {
    return cancelled_status(*context);
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
    return grpc::Status::OK;
  } catch (const std::exception& error) {
    return exception_status(error);
  }
}

grpc::Status VectorSearchService::Search(grpc::ServerContext* context,
                                         const v1::SearchRequest* request,
                                         v1::SearchResponse* response) {
  if (context->IsCancelled()) {
    return cancelled_status(*context);
  }
  try {
    const std::vector<SearchResult> results = store_->search(
        std::span<const float>(request->query().data(), request->query().size()), request->k());
    if (context->IsCancelled()) {
      return cancelled_status(*context);
    }
    for (const SearchResult& result : results) {
      v1::Neighbor* neighbor = response->add_neighbors();
      neighbor->set_id(result.id);
      neighbor->set_score(result.score);
    }
    return grpc::Status::OK;
  } catch (const std::exception& error) {
    return exception_status(error);
  }
}

grpc::Status VectorSearchService::GetStats(grpc::ServerContext* context, const v1::GetStatsRequest*,
                                           v1::GetStatsResponse* response) {
  if (context->IsCancelled()) {
    return cancelled_status(*context);
  }
  try {
    const StoreStats stats = store_->stats();
    response->set_index_type(index_type_);
    response->set_vector_count(stats.index_size);
    response->set_dimension(stats.dimension);
    response->set_successful_queries(stats.successful_queries);
    response->set_failed_queries(stats.failed_queries);
    response->set_inserted_vectors(stats.inserted_vectors);
    return grpc::Status::OK;
  } catch (const std::exception& error) {
    return exception_status(error);
  }
}

grpc::Status VectorSearchService::cancelled_status(const grpc::ServerContext& context) {
  if (std::chrono::system_clock::now() >= context.deadline()) {
    return {grpc::StatusCode::DEADLINE_EXCEEDED, "request deadline exceeded"};
  }
  return {grpc::StatusCode::CANCELLED, "request cancelled"};
}

grpc::Status VectorSearchService::exception_status(const std::exception& error) {
  if (dynamic_cast<const std::length_error*>(&error) != nullptr) {
    return {grpc::StatusCode::RESOURCE_EXHAUSTED, error.what()};
  }
  if (dynamic_cast<const std::invalid_argument*>(&error) != nullptr) {
    return {grpc::StatusCode::INVALID_ARGUMENT, error.what()};
  }
  return {grpc::StatusCode::INTERNAL, "internal vector service error"};
}

}  // namespace fast_vector::service
