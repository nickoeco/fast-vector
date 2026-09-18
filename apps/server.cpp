#include <grpcpp/grpcpp.h>

#include <atomic>
#include <charconv>
#include <chrono>
#include <csignal>
#include <cstddef>
#include <cstdint>
#include <iostream>
#include <limits>
#include <memory>
#include <stdexcept>
#include <string>
#include <string_view>
#include <thread>

#include "fast_vector/flat_index.h"
#include "fast_vector/hnsw_index.h"
#include "fast_vector/vector_store.h"
#include "service/vector_search_service.h"

namespace {

volatile std::sig_atomic_t stop_requested = 0;

void handle_signal(int) { stop_requested = 1; }

struct ServerConfig {
  std::string address = "0.0.0.0:50051";
  std::string index_type = "hnsw";
  std::size_t dimension = 128;
  std::size_t maximum_batch_size = 1'000;
  int maximum_message_bytes = 16 * 1024 * 1024;
  fast_vector::DotProductKernel kernel = fast_vector::DotProductKernel::Scalar;
  std::size_t max_connections = 16;
  std::size_t ef_construction = 200;
  std::size_t ef_search = 100;
  std::uint64_t hnsw_seed = 42;
  fast_vector::HnswNeighborSelection neighbor_selection =
      fast_vector::HnswNeighborSelection::Heuristic;
};

std::uint64_t parse_unsigned(const std::string_view value, const std::string& option) {
  std::uint64_t parsed = 0;
  const auto [end, error] = std::from_chars(value.data(), value.data() + value.size(), parsed);
  if (error != std::errc{} || end != value.data() + value.size()) {
    throw std::invalid_argument("invalid value for " + option);
  }
  return parsed;
}

std::size_t parse_positive_size(const std::string_view value, const std::string& option) {
  const std::uint64_t parsed = parse_unsigned(value, option);
  if (parsed == 0 || parsed > std::numeric_limits<std::size_t>::max()) {
    throw std::invalid_argument(option + " must be a positive platform-sized integer");
  }
  return static_cast<std::size_t>(parsed);
}

ServerConfig parse_arguments(const int argc, char* argv[]) {
  ServerConfig config;
  for (int i = 1; i < argc; i += 2) {
    if (i + 1 >= argc) {
      throw std::invalid_argument("every server option requires a value");
    }
    const std::string option = argv[i];
    const std::string_view value = argv[i + 1];
    if (option == "--address") {
      config.address = value;
    } else if (option == "--index") {
      config.index_type = value;
      if (value != "flat" && value != "hnsw") {
        throw std::invalid_argument("--index must be flat or hnsw");
      }
    } else if (option == "--dimension") {
      config.dimension = parse_positive_size(value, option);
    } else if (option == "--max-batch-size") {
      config.maximum_batch_size = parse_positive_size(value, option);
    } else if (option == "--max-message-bytes") {
      const std::uint64_t bytes = parse_unsigned(value, option);
      if (bytes == 0 || bytes > static_cast<std::uint64_t>(std::numeric_limits<int>::max())) {
        throw std::invalid_argument("--max-message-bytes must fit in a positive int");
      }
      config.maximum_message_bytes = static_cast<int>(bytes);
    } else if (option == "--m") {
      config.max_connections = parse_positive_size(value, option);
    } else if (option == "--ef-construction") {
      config.ef_construction = parse_positive_size(value, option);
    } else if (option == "--ef-search") {
      config.ef_search = parse_positive_size(value, option);
    } else if (option == "--hnsw-seed") {
      config.hnsw_seed = parse_unsigned(value, option);
    } else if (option == "--kernel") {
      if (value == "scalar") {
        config.kernel = fast_vector::DotProductKernel::Scalar;
      } else if (value == "auto") {
        config.kernel = fast_vector::DotProductKernel::AutoVectorized;
      } else if (value == "avx2") {
        config.kernel = fast_vector::DotProductKernel::Avx2;
      } else {
        throw std::invalid_argument("--kernel must be scalar, auto, or avx2");
      }
    } else if (option == "--neighbor-selection") {
      if (value == "simple") {
        config.neighbor_selection = fast_vector::HnswNeighborSelection::Simple;
      } else if (value == "heuristic") {
        config.neighbor_selection = fast_vector::HnswNeighborSelection::Heuristic;
      } else {
        throw std::invalid_argument("--neighbor-selection must be simple or heuristic");
      }
    } else {
      throw std::invalid_argument("unknown server option: " + option);
    }
  }
  if (config.address.empty()) {
    throw std::invalid_argument("--address must not be empty");
  }
  return config;
}

std::unique_ptr<fast_vector::VectorIndex> make_index(const ServerConfig& config) {
  if (config.index_type == "flat") {
    return std::make_unique<fast_vector::FlatIndex>(config.dimension, config.kernel);
  }
  return std::make_unique<fast_vector::HnswIndex>(fast_vector::HnswConfig{
      .dimension = config.dimension,
      .max_connections = config.max_connections,
      .ef_construction = config.ef_construction,
      .ef_search = config.ef_search,
      .random_seed = config.hnsw_seed,
      .kernel = config.kernel,
      .neighbor_selection = config.neighbor_selection,
  });
}

void print_usage() {
  std::cerr << "Usage: fast_vector_server [--address HOST:PORT] [--index flat|hnsw] "
               "[--dimension D] [--max-batch-size N] [--max-message-bytes N] "
               "[--kernel scalar|auto|avx2] [--m M] [--ef-construction N] "
               "[--ef-search N] [--hnsw-seed S] "
               "[--neighbor-selection simple|heuristic]\n";
}

}  // namespace

int main(const int argc, char* argv[]) {
  try {
    const ServerConfig config = parse_arguments(argc, argv);
    auto store =
        std::make_shared<fast_vector::VectorStore>(make_index(config), config.maximum_batch_size);
    fast_vector::service::VectorSearchService vector_service(store, config.index_type);

    grpc::EnableDefaultHealthCheckService(true);
    grpc::ServerBuilder builder;
    int selected_port = 0;
    builder.AddListeningPort(config.address, grpc::InsecureServerCredentials(), &selected_port);
    builder.RegisterService(&vector_service);
    builder.SetMaxReceiveMessageSize(config.maximum_message_bytes);
    builder.SetMaxSendMessageSize(config.maximum_message_bytes);
    std::unique_ptr<grpc::Server> server = builder.BuildAndStart();
    if (server == nullptr || selected_port == 0) {
      throw std::runtime_error("failed to start gRPC server on " + config.address);
    }

    std::signal(SIGINT, handle_signal);
    std::signal(SIGTERM, handle_signal);
    std::thread shutdown_monitor([&server] {
      while (stop_requested == 0) {
        std::this_thread::sleep_for(std::chrono::milliseconds(50));
      }
      server->Shutdown();
    });

    std::cout << "fast-vector gRPC server listening on " << config.address << " (selected port "
              << selected_port << ", index " << config.index_type << ", dimension "
              << config.dimension << ")\n";
    server->Wait();
    shutdown_monitor.join();
  } catch (const std::exception& error) {
    std::cerr << "Server failed: " << error.what() << '\n';
    print_usage();
    return 1;
  }
  return 0;
}
