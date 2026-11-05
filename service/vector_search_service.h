#pragma once

#include <grpcpp/grpcpp.h>

#include <array>
#include <atomic>
#include <chrono>
#include <memory>
#include <string>

#include "fast_vector/search_executor.h"
#include "fast_vector/v1/vector_search.grpc.pb.h"
#include "fast_vector/vector_store.h"
#include "service/snapshot_publication.h"
#include "service/structured_logger.h"

namespace fast_vector::service {

inline constexpr char kVectorSearchServiceName[] = "fast_vector.v1.VectorSearchService";

/** Synchronous gRPC adapter for the transport-independent VectorStore. */
class VectorSearchService final : public v1::VectorSearchService::Service {
 public:
  VectorSearchService(std::shared_ptr<VectorStore> store, std::string index_type,
                      std::shared_ptr<StructuredLogger> logger = nullptr,
                      SearchExecutorConfig search_config = {},
                      PublicationConfig publication_config = {});
  ~VectorSearchService() override {
    shutdown_searches();
    log_rpc_summary();
  }
  void stop_searches() { executor_->stop(); }
  void shutdown_searches() { executor_->shutdown(); }

  /** Emit final process-local handler status counts after all RPCs have drained. */
  void log_rpc_summary() noexcept;

  grpc::Status AddVector(grpc::ServerContext* context, const v1::AddVectorRequest* request,
                         v1::AddVectorResponse* response) override;
  grpc::Status BatchAdd(grpc::ServerContext* context, const v1::BatchAddRequest* request,
                        v1::BatchAddResponse* response) override;
  grpc::Status Search(grpc::ServerContext* context, const v1::SearchRequest* request,
                      v1::SearchResponse* response) override;
  grpc::Status GetStats(grpc::ServerContext* context, const v1::GetStatsRequest* request,
                        v1::GetStatsResponse* response) override;
  grpc::Status PrepareSnapshot(grpc::ServerContext* context,
                               grpc::ServerReader<v1::PrepareSnapshotRequest>* reader,
                               v1::PublicationStatus* response) override;
  grpc::Status ActivateSnapshot(grpc::ServerContext* context,
                                const v1::ActivateSnapshotRequest* request,
                                v1::PublicationStatus* response) override;

 private:
  struct RpcCall {
    std::uint64_t id;
    std::chrono::steady_clock::time_point started;
    std::string_view method;
  };
  [[nodiscard]] RpcCall begin_rpc(std::string_view method) noexcept;
  [[nodiscard]] grpc::Status finish_rpc(const RpcCall& call, grpc::Status status);
  [[nodiscard]] static grpc::Status cancelled_status(const grpc::ServerContext& context);
  [[nodiscard]] static grpc::Status exception_status(const std::exception& error);

  std::shared_ptr<VectorStore> store_;
  std::string index_type_;
  std::shared_ptr<StructuredLogger> logger_;
  std::unique_ptr<SearchExecutor> executor_;
  std::unique_ptr<SnapshotPublication> publication_;
  std::atomic<std::uint64_t> next_request_id_{1};
  std::array<std::atomic<std::uint64_t>, 17> status_counts_{};
  std::atomic<bool> summary_logged_{false};
};

}  // namespace fast_vector::service
