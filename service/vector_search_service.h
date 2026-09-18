#pragma once

#include <grpcpp/grpcpp.h>

#include <memory>
#include <string>

#include "fast_vector/v1/vector_search.grpc.pb.h"
#include "fast_vector/vector_store.h"

namespace fast_vector::service {

inline constexpr char kVectorSearchServiceName[] = "fast_vector.v1.VectorSearchService";

/** Synchronous gRPC adapter for the transport-independent VectorStore. */
class VectorSearchService final : public v1::VectorSearchService::Service {
 public:
  VectorSearchService(std::shared_ptr<VectorStore> store, std::string index_type);

  grpc::Status AddVector(grpc::ServerContext* context, const v1::AddVectorRequest* request,
                         v1::AddVectorResponse* response) override;
  grpc::Status BatchAdd(grpc::ServerContext* context, const v1::BatchAddRequest* request,
                        v1::BatchAddResponse* response) override;
  grpc::Status Search(grpc::ServerContext* context, const v1::SearchRequest* request,
                      v1::SearchResponse* response) override;
  grpc::Status GetStats(grpc::ServerContext* context, const v1::GetStatsRequest* request,
                        v1::GetStatsResponse* response) override;

 private:
  [[nodiscard]] static grpc::Status cancelled_status(const grpc::ServerContext& context);
  [[nodiscard]] static grpc::Status exception_status(const std::exception& error);

  std::shared_ptr<VectorStore> store_;
  std::string index_type_;
};

}  // namespace fast_vector::service
