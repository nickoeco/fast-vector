#include <grpcpp/grpcpp.h>
#include <gtest/gtest.h>

#include <chrono>
#include <cstdint>
#include <future>
#include <memory>
#include <sstream>
#include <string>
#include <thread>

#include "fast_vector/flat_index.h"
#include "fast_vector/v1/vector_search.grpc.pb.h"
#include "fast_vector/vector_store.h"
#include "service/vector_search_service.h"

namespace {

class GrpcServiceTest : public ::testing::Test {
 protected:
  virtual fast_vector::SearchExecutorConfig search_config() { return {}; }
  virtual std::unique_ptr<fast_vector::VectorIndex> create_index() {
    return std::make_unique<fast_vector::FlatIndex>(2);
  }
  void SetUp() override {
    auto index = create_index();
    store_ = std::make_shared<fast_vector::VectorStore>(std::move(index), 2);
    logger_ = std::make_shared<fast_vector::service::StructuredLogger>(logs_);
    service_ = std::make_unique<fast_vector::service::VectorSearchService>(store_, "flat", logger_,
                                                                           search_config());

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

  std::ostringstream logs_;
  std::shared_ptr<fast_vector::service::StructuredLogger> logger_;
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
  EXPECT_FALSE(stats_response.read_only());
}

TEST(GrpcServiceStandaloneTest, MapsReadOnlyWritesToFailedPrecondition) {
  auto index = std::make_unique<fast_vector::FlatIndex>(2);
  index->add(42, std::vector<float>{1.0F, 0.0F});
  auto store = std::make_shared<fast_vector::VectorStore>(std::move(index), 2, true);
  fast_vector::service::VectorSearchService service(store, "flat");

  grpc::ServerBuilder builder;
  int port = 0;
  builder.AddListeningPort("127.0.0.1:0", grpc::InsecureServerCredentials(), &port);
  builder.RegisterService(&service);
  std::unique_ptr<grpc::Server> server = builder.BuildAndStart();
  ASSERT_NE(server, nullptr);

  auto channel =
      grpc::CreateChannel("127.0.0.1:" + std::to_string(port), grpc::InsecureChannelCredentials());
  auto stub = fast_vector::v1::VectorSearchService::NewStub(channel);
  fast_vector::v1::AddVectorRequest request;
  request.mutable_vector()->set_id(100);
  request.mutable_vector()->add_values(0.0F);
  request.mutable_vector()->add_values(1.0F);
  fast_vector::v1::AddVectorResponse response;
  grpc::ClientContext context;
  const grpc::Status status = stub->AddVector(&context, request, &response);

  EXPECT_EQ(status.error_code(), grpc::StatusCode::FAILED_PRECONDITION);

  fast_vector::v1::BatchAddRequest batch_request;
  auto* vector = batch_request.add_vectors();
  vector->set_id(101);
  vector->add_values(0.0F);
  vector->add_values(1.0F);
  fast_vector::v1::BatchAddResponse batch_response;
  grpc::ClientContext batch_context;
  const grpc::Status batch_status = stub->BatchAdd(&batch_context, batch_request, &batch_response);
  EXPECT_EQ(batch_status.error_code(), grpc::StatusCode::FAILED_PRECONDITION);

  fast_vector::v1::SearchRequest search_request;
  search_request.add_query(1.0F);
  search_request.add_query(0.0F);
  search_request.set_k(1);
  fast_vector::v1::SearchResponse search_response;
  grpc::ClientContext search_context;
  const grpc::Status search_status =
      stub->Search(&search_context, search_request, &search_response);
  ASSERT_TRUE(search_status.ok()) << search_status.error_message();
  ASSERT_EQ(search_response.neighbors_size(), 1);
  EXPECT_EQ(search_response.neighbors(0).id(), 42U);

  fast_vector::v1::GetStatsRequest stats_request;
  fast_vector::v1::GetStatsResponse stats_response;
  grpc::ClientContext stats_context;
  const grpc::Status stats_status = stub->GetStats(&stats_context, stats_request, &stats_response);
  ASSERT_TRUE(stats_status.ok()) << stats_status.error_message();
  EXPECT_TRUE(stats_response.read_only());
  EXPECT_EQ(stats_response.vector_count(), 1U);
  EXPECT_EQ(stats_response.inserted_vectors(), 0U);
  EXPECT_EQ(store->stats().index_size, 1U);
  server->Shutdown();
  server->Wait();
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
  service_->log_rpc_summary();
  EXPECT_NE(logs_.str().find("\"event\":\"rpc_error_summary\""), std::string::npos);
  EXPECT_NE(logs_.str().find("\"status\":\"INVALID_ARGUMENT\""), std::string::npos);
  EXPECT_NE(logs_.str().find("\"status\":\"RESOURCE_EXHAUSTED\""), std::string::npos);
  EXPECT_NE(logs_.str().find("\"failed_rpcs\":2"), std::string::npos);
}

TEST_F(GrpcServiceTest, LoggingFailureDoesNotChangeSuccessfulRpc) {
  logs_.setstate(std::ios::badbit);
  fast_vector::v1::AddVectorRequest request;
  request.mutable_vector()->set_id(42);
  request.mutable_vector()->add_values(1.0F);
  request.mutable_vector()->add_values(0.0F);
  fast_vector::v1::AddVectorResponse response;
  grpc::ClientContext context;
  const auto status = stub_->AddVector(&context, request, &response);
  EXPECT_TRUE(status.ok());
  EXPECT_EQ(response.index_size(), 1U);
  EXPECT_EQ(logger_->dropped_records(), 1U);
  logs_.clear();
  service_->log_rpc_summary();
  EXPECT_NE(logs_.str().find("\"completed_rpcs\":1"), std::string::npos);
  EXPECT_NE(logs_.str().find("\"dropped_log_records\":1"), std::string::npos);
}

class SensitiveFailureIndex final : public fast_vector::VectorIndex {
 public:
  void add(fast_vector::VectorId, std::span<const float>) override {
    throw std::runtime_error("PRIVATE_DOCUMENT_CONTENT");
  }
  std::vector<fast_vector::SearchResult> search(std::span<const float>,
                                                std::size_t) const override {
    throw std::runtime_error("PRIVATE_DOCUMENT_CONTENT");
  }
  std::size_t size() const noexcept override { return 1; }
  std::size_t dimension() const noexcept override { return 2; }
};

class InternalFailureGrpcTest : public GrpcServiceTest {
 protected:
  std::unique_ptr<fast_vector::VectorIndex> create_index() override {
    return std::make_unique<SensitiveFailureIndex>();
  }
};

TEST_F(InternalFailureGrpcTest, LogsInternalStatusWithoutExceptionMessage) {
  fast_vector::v1::SearchRequest request;
  request.add_query(1.0F);
  request.add_query(0.0F);
  request.set_k(1);
  fast_vector::v1::SearchResponse response;
  grpc::ClientContext context;
  const auto status = stub_->Search(&context, request, &response);
  EXPECT_EQ(status.error_code(), grpc::StatusCode::INTERNAL);
  EXPECT_EQ(logs_.str().find("PRIVATE_DOCUMENT_CONTENT"), std::string::npos);
  EXPECT_NE(logs_.str().find("\"status\":\"INTERNAL\""), std::string::npos);
  service_->log_rpc_summary();
  EXPECT_NE(logs_.str().find("\"failed_rpcs\":1"), std::string::npos);
}

struct RpcGate {
  std::atomic<bool> released{false};
  std::atomic<unsigned> calls{0};
};

class RpcControlledIndex final : public fast_vector::VectorIndex {
 public:
  explicit RpcControlledIndex(std::shared_ptr<RpcGate> gate) : gate_(std::move(gate)) {}
  void add(fast_vector::VectorId, std::span<const float>) override {}
  std::size_t size() const noexcept override { return 1; }
  std::size_t dimension() const noexcept override { return 2; }
  std::vector<fast_vector::SearchResult> search(std::span<const float> query,
                                                std::size_t k) const override {
    return search(query, k, fast_vector::SearchContext{});
  }
  std::vector<fast_vector::SearchResult> search(
      std::span<const float> query, std::size_t,
      const fast_vector::SearchContext& context) const override {
    ++gate_->calls;
    while (query[0] == 1 && !gate_->released.load()) {
      context.check();
      std::this_thread::sleep_for(std::chrono::milliseconds(1));
    }
    context.check();
    if (query[0] < 0) throw 7;  // Even a non-standard task exception must not escape the handler.
    return {{42, 1.0F}};
  }

 private:
  std::shared_ptr<RpcGate> gate_;
};

class ScheduledGrpcTest : public GrpcServiceTest {
 protected:
  fast_vector::SearchExecutorConfig search_config() override { return {1, 1, 1}; }
  std::unique_ptr<fast_vector::VectorIndex> create_index() override {
    return std::make_unique<RpcControlledIndex>(gate_);
  }
  void TearDown() override {
    gate_->released.store(true);
    service_->stop_searches();
    GrpcServiceTest::TearDown();
    service_->shutdown_searches();
  }
  fast_vector::v1::GetStatsResponse statistics() {
    fast_vector::v1::GetStatsRequest request;
    fast_vector::v1::GetStatsResponse response;
    grpc::ClientContext context;
#if !defined(FAST_VECTOR_GRPC_SYSTEM_LIBRARY_SANITIZER_BOUNDARY)
    context.set_deadline(std::chrono::system_clock::now() + std::chrono::seconds(5));
#endif
    EXPECT_TRUE(stub_->GetStats(&context, request, &response).ok());
    return response;
  }
  bool wait_for_queue(std::uint64_t queued, std::uint64_t running = 1) {
    const auto limit = std::chrono::steady_clock::now() + std::chrono::seconds(5);
    while (std::chrono::steady_clock::now() < limit) {
      auto stats = statistics().search_executor();
      if (stats.queued() == queued && stats.running() == running) return true;
      std::this_thread::sleep_for(std::chrono::milliseconds(2));
    }
    return false;
  }
  std::future<grpc::Status> search_async(grpc::ClientContext& context, float value = 1) {
#if !defined(FAST_VECTOR_GRPC_SYSTEM_LIBRARY_SANITIZER_BOUNDARY)
    context.set_deadline(std::chrono::system_clock::now() + std::chrono::seconds(5));
#endif
    return std::async(std::launch::async, [&, value] {
      fast_vector::v1::SearchRequest request;
      request.add_query(value);
      request.add_query(0);
      request.set_k(1);
      fast_vector::v1::SearchResponse response;
      return stub_->Search(&context, request, &response);
    });
  }
  std::shared_ptr<RpcGate> gate_ = std::make_shared<RpcGate>();
};

TEST_F(ScheduledGrpcTest, RejectsFullQueueCancelsWaitingRpcAndRecovers) {
  grpc::ClientContext first_context, waiting_context, rejected_context;
  auto first = search_async(first_context);
  EXPECT_TRUE(wait_for_queue(0));
  auto waiting = search_async(waiting_context, 2);
  EXPECT_TRUE(wait_for_queue(1));
  auto rejected = search_async(rejected_context, 2);
  EXPECT_EQ(rejected.get().error_code(), grpc::StatusCode::RESOURCE_EXHAUSTED);
  waiting_context.TryCancel();
  EXPECT_EQ(waiting.get().error_code(), grpc::StatusCode::CANCELLED);
  EXPECT_TRUE(wait_for_queue(0));
  EXPECT_EQ(gate_->calls.load(), 1U);
  first_context.TryCancel();
  EXPECT_EQ(first.get().error_code(), grpc::StatusCode::CANCELLED);
  EXPECT_TRUE(wait_for_queue(0, 0));
  grpc::ClientContext failure_context, recovery_context;
  EXPECT_EQ(search_async(failure_context, -1).get().error_code(), grpc::StatusCode::INTERNAL);
  EXPECT_TRUE(search_async(recovery_context, 2).get().ok());
  const auto stats = statistics().search_executor();
  EXPECT_LE(stats.peak_running(), 1U);
  EXPECT_LE(stats.peak_queued(), 1U);
  EXPECT_EQ(stats.rejected(), 1U);
  EXPECT_EQ(stats.cancelled(), 2U);
}

TEST_F(ScheduledGrpcTest, ShutdownSettlesQueuedAndActiveRpc) {
  grpc::ClientContext first_context, waiting_context, later_context;
  auto first = search_async(first_context);
  EXPECT_TRUE(wait_for_queue(0));
  auto waiting = search_async(waiting_context, 2);
  EXPECT_TRUE(wait_for_queue(1));
  service_->stop_searches();
  service_->shutdown_searches();
  EXPECT_EQ(first.get().error_code(), grpc::StatusCode::UNAVAILABLE);
  EXPECT_EQ(waiting.get().error_code(), grpc::StatusCode::UNAVAILABLE);
  EXPECT_EQ(search_async(later_context, 2).get().error_code(), grpc::StatusCode::UNAVAILABLE);
  const auto stats = statistics().search_executor();
  EXPECT_EQ(stats.queued(), 0U);
  EXPECT_EQ(stats.running(), 0U);
  EXPECT_EQ(stats.completed(), stats.admitted());
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
