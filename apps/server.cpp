#include <grpcpp/grpcpp.h>

#include <array>
#include <atomic>
#include <charconv>
#include <chrono>
#include <csignal>
#include <cstddef>
#include <cstdint>
#include <iostream>
#include <limits>
#include <memory>
#include <optional>
#include <span>
#include <stdexcept>
#include <string>
#include <string_view>
#include <thread>

#include "fast_vector/flat_index.h"
#include "fast_vector/flat_index_io.h"
#include "fast_vector/hnsw_index.h"
#include "fast_vector/hnsw_index_io.h"
#include "fast_vector/vector_store.h"
#include "service/vector_search_service.h"

namespace {

std::atomic<std::sig_atomic_t> stop_requested{0};
std::atomic<std::sig_atomic_t> reload_requested{0};
static_assert(std::atomic<std::sig_atomic_t>::is_always_lock_free);

void handle_stop_signal(const int signal) {
  stop_requested.store(signal, std::memory_order_relaxed);
}
void handle_reload_signal(int) { reload_requested.store(1, std::memory_order_relaxed); }

using fast_vector::service::LogField;
using fast_vector::service::StructuredLogger;

std::string_view shutdown_reason() {
  const auto signal = stop_requested.load(std::memory_order_relaxed);
  return signal == SIGINT ? "SIGINT" : (signal == SIGTERM ? "SIGTERM" : "internal_stop");
}

std::uint64_t elapsed_us(const std::chrono::steady_clock::time_point start) {
  return static_cast<std::uint64_t>(std::chrono::duration_cast<std::chrono::microseconds>(
                                        std::chrono::steady_clock::now() - start)
                                        .count());
}

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
  std::string load_index_path;
  bool read_only = false;
  bool reload_on_sighup = false;
};

bool parse_bool(const std::string_view value, const std::string& option) {
  if (value == "true") {
    return true;
  }
  if (value == "false") {
    return false;
  }
  throw std::invalid_argument(option + " must be true or false");
}

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
    } else if (option == "--load-index") {
      config.load_index_path = value;
      if (config.load_index_path.empty()) {
        throw std::invalid_argument("--load-index must not be empty");
      }
    } else if (option == "--read-only") {
      config.read_only = parse_bool(value, option);
    } else if (option == "--reload-on-sighup") {
      config.reload_on_sighup = parse_bool(value, option);
    } else {
      throw std::invalid_argument("unknown server option: " + option);
    }
  }
  if (config.address.empty()) {
    throw std::invalid_argument("--address must not be empty");
  }
  if (config.reload_on_sighup &&
      (config.index_type != "flat" || config.load_index_path.empty() || !config.read_only)) {
    throw std::invalid_argument(
        "--reload-on-sighup true requires a read-only FlatIndex loaded from a snapshot");
  }
#if !defined(SIGHUP)
  if (config.reload_on_sighup) {
    throw std::invalid_argument("--reload-on-sighup is not supported on this platform");
  }
#endif
  return config;
}

std::unique_ptr<fast_vector::VectorIndex> load_snapshot(
    const ServerConfig& config, StructuredLogger& logger, const std::string_view operation,
    const std::optional<std::uint64_t> previous_count = std::nullopt) {
  logger.write(
      {.event = "index_load_started", .status = "IN_PROGRESS", .index_type = config.index_type},
      {{"operation", operation}});
  const auto start = std::chrono::steady_clock::now();
  try {
    std::unique_ptr<fast_vector::VectorIndex> index;
    if (config.index_type == "flat") {
      index = std::make_unique<fast_vector::FlatIndex>(
          fast_vector::load_flat_index(config.load_index_path, config.kernel));
    } else {
      index = std::make_unique<fast_vector::HnswIndex>(
          fast_vector::load_hnsw_index(config.load_index_path, config.kernel));
    }
    logger.write(
        {.event = "index_load_completed",
         .status = "OK",
         .latency_us = elapsed_us(start),
         .index_type = config.index_type,
         .vector_count = index->size()},
        {{"operation", operation}, {"dimension", static_cast<std::uint64_t>(index->dimension())}});
    return index;
  } catch (...) {
    logger.write({.severity = "ERROR",
                  .event = "index_load_failed",
                  .status = "FAILED_PRECONDITION",
                  .latency_us = elapsed_us(start),
                  .index_type = config.index_type,
                  .vector_count = previous_count},
                 {{"operation", operation},
                  {"kept_previous_index", previous_count.has_value()},
                  {"reason", std::string_view("snapshot_unavailable_or_invalid")}});
    throw;
  }
}

std::unique_ptr<fast_vector::VectorIndex> make_index(const ServerConfig& config,
                                                     StructuredLogger& logger) {
  if (!config.load_index_path.empty()) {
    return load_snapshot(config, logger, "startup");
  }
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

constexpr char kUsage[] =
    "fast_vector_server [--address HOST:PORT] [--index flat|hnsw] "
    "[--dimension D] [--max-batch-size N] [--max-message-bytes N] "
    "[--kernel scalar|auto|avx2] [--m M] [--ef-construction N] "
    "[--ef-search N] [--hnsw-seed S] "
    "[--neighbor-selection simple|heuristic] [--load-index PATH] "
    "[--read-only true|false] [--reload-on-sighup true|false]";

void log_config(const ServerConfig& config, const fast_vector::VectorIndex& index,
                StructuredLogger& logger) {
  const auto* hnsw = dynamic_cast<const fast_vector::HnswIndex*>(&index);
  const std::array<LogField, 12> fields{{
      {"dimension", static_cast<std::uint64_t>(index.dimension())},
      {"kernel", std::string_view(fast_vector::dot_product_kernel_name(config.kernel))},
      {"read_only", config.read_only},
      {"config_source", std::string_view(config.load_index_path.empty() ? "cli" : "snapshot")},
      {"reload_on_sighup", config.reload_on_sighup},
      {"max_batch_size", static_cast<std::uint64_t>(config.maximum_batch_size)},
      {"max_message_bytes", static_cast<std::uint64_t>(config.maximum_message_bytes)},
      {"m", static_cast<std::uint64_t>(hnsw ? hnsw->config().max_connections : 0)},
      {"ef_construction", static_cast<std::uint64_t>(hnsw ? hnsw->config().ef_construction : 0)},
      {"ef_search", static_cast<std::uint64_t>(hnsw ? hnsw->config().ef_search : 0)},
      {"hnsw_seed", hnsw ? hnsw->config().random_seed : std::uint64_t{0}},
      {"neighbor_selection",
       std::string_view(hnsw && hnsw->config().neighbor_selection ==
                                    fast_vector::HnswNeighborSelection::Heuristic
                            ? "heuristic"
                            : "simple")},
  }};
  logger.write({.event = "startup_config",
                .status = "OK",
                .index_type = config.index_type,
                .vector_count = index.size()},
               std::span<const LogField>(fields).first(hnsw ? 12 : 7));
}

}  // namespace

int main(const int argc, char* argv[]) {
  auto logger = std::make_shared<StructuredLogger>(std::cerr);
  std::string_view phase = "invalid_configuration";
  std::string index_type;
  bool snapshot_requested = false;
  std::signal(SIGINT, handle_stop_signal);
  std::signal(SIGTERM, handle_stop_signal);
  try {
    const ServerConfig config = parse_arguments(argc, argv);
    index_type = config.index_type;
    snapshot_requested = !config.load_index_path.empty();
    phase = "index_initialization_failed";
    auto index = make_index(config, *logger);
    log_config(config, *index, *logger);
    auto store = std::make_shared<fast_vector::VectorStore>(
        std::move(index), config.maximum_batch_size, config.read_only);
    fast_vector::service::VectorSearchService vector_service(store, config.index_type, logger);
    if (stop_requested.load(std::memory_order_relaxed) != 0) {
      logger->write({.event = "shutdown_completed",
                     .status = "OK",
                     .index_type = config.index_type,
                     .vector_count = store->stats().index_size},
                    {{"reason", shutdown_reason()}});
      return 0;
    }

    phase = "listen_failed";
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

#if defined(SIGHUP)
    if (config.reload_on_sighup) {
      std::signal(SIGHUP, handle_reload_signal);
    }
#endif
    std::jthread shutdown_monitor([&config, &server, &store, &logger](std::stop_token stop) {
      while (stop_requested.load(std::memory_order_relaxed) == 0 && !stop.stop_requested()) {
#if defined(SIGHUP)
        if (reload_requested.exchange(0, std::memory_order_relaxed) != 0) {
          try {
            auto replacement = load_snapshot(config, *logger, "reload", store->stats().index_size);
            const std::size_t count = replacement->size();
            const std::size_t dimension = replacement->dimension();
            store->replace_index(std::move(replacement));
            logger->write({.event = "index_replaced",
                           .status = "OK",
                           .index_type = config.index_type,
                           .vector_count = count},
                          {{"dimension", static_cast<std::uint64_t>(dimension)}});
          } catch (...) {
            // load_snapshot already logged a sanitized failure; retain the current index.
          }
        }
#endif
        std::this_thread::sleep_for(std::chrono::milliseconds(50));
      }
      logger->write({.event = "shutdown_requested", .index_type = config.index_type},
                    {{"reason", shutdown_reason()}});
      server->Shutdown();
    });

    const fast_vector::StoreStats stats = store->stats();
    logger->write({.event = "server_started",
                   .status = "OK",
                   .index_type = config.index_type,
                   .vector_count = stats.index_size},
                  {{"selected_port", static_cast<std::uint64_t>(selected_port)}});
    phase = "serving_failed";
    server->Wait();
    shutdown_monitor.join();
    vector_service.log_rpc_summary();
    logger->write({.event = "shutdown_completed",
                   .status = "OK",
                   .index_type = config.index_type,
                   .vector_count = store->stats().index_size},
                  {{"reason", shutdown_reason()}});
  } catch (const std::exception& error) {
    logger->write({.severity = "ERROR",
                   .event = phase == "serving_failed" ? "server_failed" : "startup_failed",
                   .status = dynamic_cast<const std::invalid_argument*>(&error)
                                 ? "INVALID_ARGUMENT"
                                 : (snapshot_requested && phase == "index_initialization_failed"
                                        ? "FAILED_PRECONDITION"
                                        : "INTERNAL"),
                   .index_type = index_type},
                  {{"reason", phase}, {"usage", std::string_view(kUsage)}});
    return 1;
  }
  return 0;
}
