#include <grpcpp/grpcpp.h>
#include <gtest/gtest.h>

#include <chrono>
#include <cstdint>
#include <memory>
#include <string>

#include "fast_vector/flat_index.h"
#include "fast_vector/v1/vector_search.grpc.pb.h"
#include "fast_vector/vector_store.h"
#include "service/vector_search_service.h"

namespace {

class GrpcServiceTest : public ::testing::Test {
 protected:
  void SetUp() override {
    auto index = std::make_unique<fast_vector::FlatIndex>(2);
    store_ = std::make_shared<fast_vector::VectorStore>(std::move(index), 2);
    service_ = std::make_unique<fast_vector::service::VectorSearchService>(store_, "flat");

    grpc::EnableDefaultHealthCheckService(true);
    grpc::ServerBuilder builder;
    int port = 0;
    builder.AddListeningPort("127.0.0.1:0", grpc::InsecureServerCredentials(), &port);
    builder.RegisterService(service_.get());
    server_ = builder.BuildAndStart();
    ASSERT_NE(server_, nullptr);
    ASSERT_GT(port, 0);

    channel_ = grpc::CreateChannel("127.0.0.1:" + std::to_string(port),
                                   grpc::InsecureChannelCredentials());
    ASSERT_TRUE(
        channel_->WaitForConnected(std::chrono::system_clock::now() + std::chrono::seconds(5)));
    stub_ = fast_vector::v1::VectorSearchService::NewStub(channel_);
  }

  void TearDown() override {
    if (server_ != nullptr) {
      server_->Shutdown();
      server_->Wait();
    }
  }

  std::shared_ptr<fast_vector::VectorStore> store_;
  std::unique_ptr<fast_vector::service::VectorSearchService> service_;
  std::unique_ptr<grpc::Server> server_;
  std::shared_ptr<grpc::Channel> channel_;
  std::unique_ptr<fast_vector::v1::VectorSearchService::Stub> stub_;
};

TEST_F(GrpcServiceTest, AddsSearchesAndReturnsStatsOverLocalhost) {
  fast_vector::v1::AddVectorRequest add_request;
  add_request.mutable_vector()->set_id(42);
  add_request.mutable_vector()->add_values(1.0F);
  add_request.mutable_vector()->add_values(0.0F);
  fast_vector::v1::AddVectorResponse add_response;
  grpc::ClientContext add_context;
  const grpc::Status add_status = stub_->AddVector(&add_context, add_request, &add_response);
  ASSERT_TRUE(add_status.ok()) << add_status.error_message();
  EXPECT_EQ(add_response.index_size(), 1U);

  fast_vector::v1::SearchRequest search_request;
  search_request.add_query(1.0F);
  search_request.add_query(0.0F);
  search_request.set_k(1);
  fast_vector::v1::SearchResponse search_response;
  grpc::ClientContext search_context;
  const grpc::Status search_status =
      stub_->Search(&search_context, search_request, &search_response);
  ASSERT_TRUE(search_status.ok()) << search_status.error_message();
  ASSERT_EQ(search_response.neighbors_size(), 1);
  EXPECT_EQ(search_response.neighbors(0).id(), 42U);
  EXPECT_NEAR(search_response.neighbors(0).score(), 1.0F, 1.0e-5F);

  fast_vector::v1::GetStatsRequest stats_request;
  fast_vector::v1::GetStatsResponse stats_response;
  grpc::ClientContext stats_context;
  const grpc::Status stats_status = stub_->GetStats(&stats_context, stats_request, &stats_response);
  ASSERT_TRUE(stats_status.ok()) << stats_status.error_message();
  EXPECT_EQ(stats_response.index_type(), "flat");
  EXPECT_EQ(stats_response.vector_count(), 1U);
  EXPECT_EQ(stats_response.dimension(), 2U);
  EXPECT_EQ(stats_response.successful_queries(), 1U);
  EXPECT_EQ(stats_response.failed_queries(), 0U);
  EXPECT_EQ(stats_response.inserted_vectors(), 1U);
}

TEST_F(GrpcServiceTest, MapsInputAndResourceErrorsToGrpcStatuses) {
  fast_vector::v1::AddVectorRequest invalid_add;
  invalid_add.mutable_vector()->set_id(1);
  invalid_add.mutable_vector()->add_values(1.0F);
  fast_vector::v1::AddVectorResponse add_response;
  grpc::ClientContext add_context;
  const grpc::Status invalid_status = stub_->AddVector(&add_context, invalid_add, &add_response);
  EXPECT_EQ(invalid_status.error_code(), grpc::StatusCode::INVALID_ARGUMENT);

  fast_vector::v1::BatchAddRequest batch;
  for (std::uint64_t id = 1; id <= 3; ++id) {
    auto* vector = batch.add_vectors();
    vector->set_id(id);
    vector->add_values(1.0F);
    vector->add_values(static_cast<float>(id));
  }
  fast_vector::v1::BatchAddResponse batch_response;
  grpc::ClientContext batch_context;
  const grpc::Status batch_status = stub_->BatchAdd(&batch_context, batch, &batch_response);
  EXPECT_EQ(batch_status.error_code(), grpc::StatusCode::RESOURCE_EXHAUSTED);
  EXPECT_EQ(store_->stats().index_size, 0U);
}

TEST_F(GrpcServiceTest, HonorsExpiredDeadline) {
#if defined(FAST_VECTOR_GRPC_SYSTEM_LIBRARY_SANITIZER_BOUNDARY)
  GTEST_SKIP() << "Ubuntu 22.04 gRPC 1.30 ClientContext deadline cleanup is incompatible "
                  "with an ASan/UBSan-instrumented caller";
#else
  fast_vector::v1::SearchRequest request;
  request.add_query(1.0F);
  request.add_query(0.0F);
  request.set_k(1);
  fast_vector::v1::SearchResponse response;
  grpc::ClientContext context;
  context.set_deadline(std::chrono::system_clock::now() - std::chrono::seconds(1));

  const grpc::Status status = stub_->Search(&context, request, &response);
  EXPECT_EQ(status.error_code(), grpc::StatusCode::DEADLINE_EXCEEDED);
#endif
}

}  // namespace
