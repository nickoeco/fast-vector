#include "fast_vector/hnsw_index_io.h"

#include <algorithm>
#include <array>
#include <bit>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <fstream>
#include <limits>
#include <stdexcept>
#include <string>
#include <type_traits>
#include <unordered_set>
#include <utility>
#include <vector>

namespace fast_vector {
namespace {

constexpr std::array<unsigned char, 8> kMagic{'F', 'V', 'H', 'N', 'S', 'W', '\0', '\0'};
constexpr std::uint32_t kVersion = 1;
constexpr std::uint32_t kHeaderSize = 112;
constexpr std::uint32_t kEndianMarker = 0x01020304U;
constexpr std::uint32_t kNormalizedCosineFlag = 1U;
constexpr std::uint32_t kMaximumLevel = 63;
constexpr std::uint64_t kEmptyEntry = std::numeric_limits<std::uint64_t>::max();
constexpr std::uint64_t kFnvOffset = 14695981039346656037ULL;
constexpr std::uint64_t kFnvPrime = 1099511628211ULL;
constexpr double kNormTolerance = 1.0e-3;

[[noreturn]] void fail(const std::string& reason) {
  throw std::runtime_error("invalid HNSW index: " + reason);
}

std::uint64_t checked_add(const std::uint64_t left, const std::uint64_t right,
                          const std::string& field) {
  if (right > std::numeric_limits<std::uint64_t>::max() - left) {
    fail(field + " overflows");
  }
  return left + right;
}

std::uint64_t checked_multiply(const std::uint64_t left, const std::uint64_t right,
                               const std::string& field) {
  if (left != 0 && right > std::numeric_limits<std::uint64_t>::max() / left) {
    fail(field + " overflows");
  }
  return left * right;
}

std::size_t checked_size(const std::uint64_t value, const std::string& field) {
  if (value > std::numeric_limits<std::size_t>::max()) {
    fail(field + " exceeds platform limits");
  }
  return static_cast<std::size_t>(value);
}

void update_checksum(std::uint64_t& checksum, const unsigned char byte) {
  checksum ^= byte;
  checksum *= kFnvPrime;
}

template <typename UInt>
void checksum_integer(std::uint64_t& checksum, UInt value) {
  static_assert(std::is_unsigned_v<UInt>);
  for (std::size_t byte = 0; byte < sizeof(UInt); ++byte) {
    update_checksum(checksum, static_cast<unsigned char>(value & 0xFFU));
    value >>= 8U;
  }
}

template <typename UInt>
void write_integer(std::ostream& output, UInt value) {
  static_assert(std::is_unsigned_v<UInt>);
  std::array<char, sizeof(UInt)> bytes{};
  for (char& byte : bytes) {
    byte = static_cast<char>(value & 0xFFU);
    value >>= 8U;
  }
  output.write(bytes.data(), static_cast<std::streamsize>(bytes.size()));
  if (!output) {
    throw std::runtime_error("failed to write HNSW index");
  }
}

template <typename UInt>
UInt read_integer(std::istream& input, std::uint64_t* checksum) {
  static_assert(std::is_unsigned_v<UInt>);
  std::array<unsigned char, sizeof(UInt)> bytes{};
  input.read(reinterpret_cast<char*>(bytes.data()), static_cast<std::streamsize>(bytes.size()));
  if (!input) {
    fail("file is truncated");
  }
  UInt value = 0;
  for (std::size_t byte = 0; byte < bytes.size(); ++byte) {
    if (checksum != nullptr) {
      update_checksum(*checksum, bytes[byte]);
    }
    value |= static_cast<UInt>(bytes[byte]) << (8U * byte);
  }
  return value;
}

struct Header {
  std::uint64_t dimension;
  std::uint64_t count;
  std::uint64_t max_connections;
  std::uint64_t ef_construction;
  std::uint64_t ef_search;
  std::uint64_t seed;
  std::uint32_t selection;
  std::uint64_t maximum_level;
  std::uint64_t entry_point;
  std::uint64_t payload_bytes;
};

void checksum_header(std::uint64_t& checksum, const Header& header) {
  for (const unsigned char byte : kMagic) {
    update_checksum(checksum, byte);
  }
  checksum_integer(checksum, kVersion);
  checksum_integer(checksum, kHeaderSize);
  checksum_integer(checksum, kEndianMarker);
  checksum_integer(checksum, kNormalizedCosineFlag);
  checksum_integer(checksum, header.dimension);
  checksum_integer(checksum, header.count);
  checksum_integer(checksum, header.max_connections);
  checksum_integer(checksum, header.ef_construction);
  checksum_integer(checksum, header.ef_search);
  checksum_integer(checksum, header.seed);
  checksum_integer(checksum, header.selection);
  checksum_integer(checksum, std::uint32_t{0});
  checksum_integer(checksum, header.maximum_level);
  checksum_integer(checksum, header.entry_point);
  checksum_integer(checksum, header.payload_bytes);
}

void write_header(std::ostream& output, const Header& header, const std::uint64_t checksum) {
  output.write(reinterpret_cast<const char*>(kMagic.data()),
               static_cast<std::streamsize>(kMagic.size()));
  write_integer(output, kVersion);
  write_integer(output, kHeaderSize);
  write_integer(output, kEndianMarker);
  write_integer(output, kNormalizedCosineFlag);
  write_integer(output, header.dimension);
  write_integer(output, header.count);
  write_integer(output, header.max_connections);
  write_integer(output, header.ef_construction);
  write_integer(output, header.ef_search);
  write_integer(output, header.seed);
  write_integer(output, header.selection);
  write_integer(output, std::uint32_t{0});
  write_integer(output, header.maximum_level);
  write_integer(output, header.entry_point);
  write_integer(output, header.payload_bytes);
  write_integer(output, checksum);
}

Header read_header(std::istream& input, std::uint64_t& checksum, std::uint64_t& stored_checksum) {
  std::array<unsigned char, kMagic.size()> magic{};
  input.read(reinterpret_cast<char*>(magic.data()), static_cast<std::streamsize>(magic.size()));
  if (!input || magic != kMagic) {
    fail("magic does not match");
  }
  for (const unsigned char byte : magic) {
    update_checksum(checksum, byte);
  }
  if (read_integer<std::uint32_t>(input, &checksum) != kVersion) {
    fail("unsupported format version");
  }
  if (read_integer<std::uint32_t>(input, &checksum) != kHeaderSize) {
    fail("header size does not match version 1");
  }
  if (read_integer<std::uint32_t>(input, &checksum) != kEndianMarker) {
    fail("endianness marker does not match");
  }
  if (read_integer<std::uint32_t>(input, &checksum) != kNormalizedCosineFlag) {
    fail("unsupported index flags");
  }
  Header header{};
  header.dimension = read_integer<std::uint64_t>(input, &checksum);
  header.count = read_integer<std::uint64_t>(input, &checksum);
  header.max_connections = read_integer<std::uint64_t>(input, &checksum);
  header.ef_construction = read_integer<std::uint64_t>(input, &checksum);
  header.ef_search = read_integer<std::uint64_t>(input, &checksum);
  header.seed = read_integer<std::uint64_t>(input, &checksum);
  header.selection = read_integer<std::uint32_t>(input, &checksum);
  if (read_integer<std::uint32_t>(input, &checksum) != 0) {
    fail("reserved header field is not zero");
  }
  header.maximum_level = read_integer<std::uint64_t>(input, &checksum);
  header.entry_point = read_integer<std::uint64_t>(input, &checksum);
  header.payload_bytes = read_integer<std::uint64_t>(input, &checksum);
  stored_checksum = read_integer<std::uint64_t>(input, nullptr);
  return header;
}

void validate_header(const Header& header, const std::uintmax_t file_bytes) {
  if (header.dimension == 0 || header.max_connections < 2 ||
      header.max_connections > std::numeric_limits<std::uint32_t>::max() / 2U ||
      header.ef_construction < header.max_connections || header.ef_search == 0 ||
      header.selection > 1U || header.count > std::numeric_limits<std::uint32_t>::max() ||
      header.maximum_level > kMaximumLevel) {
    fail("invalid configuration or node count");
  }
  checked_size(header.dimension, "dimension");
  checked_size(header.max_connections, "M");
  checked_size(header.ef_construction, "efConstruction");
  checked_size(header.ef_search, "efSearch");
  const std::uint64_t elements = checked_multiply(header.count, header.dimension, "vector count");
  checked_size(elements, "vector count");
  const std::uint64_t vector_bytes = checked_multiply(elements, sizeof(float), "vector bytes");
  const std::uint64_t id_bytes = checked_multiply(header.count, sizeof(VectorId), "ID bytes");
  const std::uint64_t minimum_graph_bytes = checked_multiply(header.count, 8, "graph bytes");
  const std::uint64_t minimum_payload =
      checked_add(checked_add(id_bytes, vector_bytes, "payload"), minimum_graph_bytes, "payload");
  if (header.payload_bytes < minimum_payload) {
    fail("payload is too small for the declared dimensions and node count");
  }
  const std::uint64_t maximum_degree_bytes =
      checked_multiply(header.max_connections * 2, sizeof(std::uint32_t), "degree bytes");
  const std::uint64_t maximum_layer_bytes =
      checked_add(sizeof(std::uint32_t), maximum_degree_bytes, "layer bytes");
  const std::uint64_t maximum_graph_bytes = checked_multiply(
      header.count,
      checked_add(sizeof(std::uint32_t),
                  checked_multiply(kMaximumLevel + 1, maximum_layer_bytes, "graph bytes"),
                  "graph bytes"),
      "graph bytes");
  const std::uint64_t maximum_payload =
      checked_add(checked_add(id_bytes, vector_bytes, "payload"), maximum_graph_bytes, "payload");
  if (header.payload_bytes > maximum_payload ||
      checked_add(kHeaderSize, header.payload_bytes, "file size") != file_bytes) {
    fail("file size does not match the header");
  }
  if (header.count == 0) {
    if (header.entry_point != kEmptyEntry || header.maximum_level != 0) {
      fail("empty graph has an invalid entry point or maximum level");
    }
  } else if (header.entry_point >= header.count) {
    fail("entry point is out of range");
  }
}

}  // namespace

class HnswIndexSerializer {
 public:
  static void save(const HnswIndex& index, const std::filesystem::path& path) {
    const std::uint64_t count = index.ids_.size();
    const std::uint64_t dimension = index.config_.dimension;
    const std::uint64_t elements = checked_multiply(count, dimension, "vector count");
    if (count > std::numeric_limits<std::uint32_t>::max() || elements != index.vectors_.size() ||
        index.graph_.size() != count ||
        index.config_.max_connections > std::numeric_limits<std::uint32_t>::max() / 2U) {
      fail("in-memory index exceeds snapshot format limits");
    }
    std::uint64_t payload_bytes =
        checked_add(checked_multiply(count, sizeof(VectorId), "ID bytes"),
                    checked_multiply(elements, sizeof(float), "vector bytes"), "payload");
    for (const HnswIndex::Node& node : index.graph_) {
      payload_bytes = checked_add(payload_bytes, sizeof(std::uint32_t), "payload");
      for (const std::vector<HnswIndex::NodeIndex>& neighbors : node.levels) {
        payload_bytes = checked_add(payload_bytes, sizeof(std::uint32_t), "payload");
        payload_bytes = checked_add(
            payload_bytes, checked_multiply(neighbors.size(), sizeof(std::uint32_t), "edge bytes"),
            "payload");
      }
    }
    const Header header{
        dimension,
        count,
        index.config_.max_connections,
        index.config_.ef_construction,
        index.config_.ef_search,
        index.config_.random_seed,
        index.config_.neighbor_selection == HnswNeighborSelection::Simple ? 0U : 1U,
        count == 0 ? 0U : index.max_level_,
        count == 0 ? kEmptyEntry : index.entry_point_,
        payload_bytes,
    };
    std::uint64_t checksum = kFnvOffset;
    checksum_header(checksum, header);
    for (const VectorId id : index.ids_) {
      checksum_integer(checksum, id);
    }
    for (const float value : index.vectors_) {
      checksum_integer(checksum, std::bit_cast<std::uint32_t>(value));
    }
    for (const HnswIndex::Node& node : index.graph_) {
      checksum_integer(checksum, static_cast<std::uint32_t>(node.levels.size() - 1));
      for (const std::vector<HnswIndex::NodeIndex>& neighbors : node.levels) {
        checksum_integer(checksum, static_cast<std::uint32_t>(neighbors.size()));
        for (const HnswIndex::NodeIndex neighbor : neighbors) {
          checksum_integer(checksum, neighbor);
        }
      }
    }

    std::ofstream output(path, std::ios::binary | std::ios::trunc);
    if (!output) {
      throw std::runtime_error("cannot open HNSW index for writing: " + path.string());
    }
    write_header(output, header, checksum);
    for (const VectorId id : index.ids_) {
      write_integer(output, id);
    }
    for (const float value : index.vectors_) {
      write_integer(output, std::bit_cast<std::uint32_t>(value));
    }
    for (const HnswIndex::Node& node : index.graph_) {
      write_integer(output, static_cast<std::uint32_t>(node.levels.size() - 1));
      for (const std::vector<HnswIndex::NodeIndex>& neighbors : node.levels) {
        write_integer(output, static_cast<std::uint32_t>(neighbors.size()));
        for (const HnswIndex::NodeIndex neighbor : neighbors) {
          write_integer(output, neighbor);
        }
      }
    }
    output.flush();
    if (!output) {
      throw std::runtime_error("failed to flush HNSW index: " + path.string());
    }
  }

  static HnswIndex load(const std::filesystem::path& path, const DotProductKernel kernel) {
    std::ifstream input(path, std::ios::binary);
    if (!input) {
      throw std::runtime_error("cannot open HNSW index for reading: " + path.string());
    }
    std::uint64_t checksum = kFnvOffset;
    std::uint64_t stored_checksum = 0;
    const Header header = read_header(input, checksum, stored_checksum);
    validate_header(header, std::filesystem::file_size(path));

    const std::size_t count = checked_size(header.count, "node count");
    const std::size_t dimension = checked_size(header.dimension, "dimension");
    HnswIndex index(HnswConfig{
        .dimension = dimension,
        .max_connections = checked_size(header.max_connections, "M"),
        .ef_construction = checked_size(header.ef_construction, "efConstruction"),
        .ef_search = checked_size(header.ef_search, "efSearch"),
        .random_seed = header.seed,
        .kernel = kernel,
        .neighbor_selection = header.selection == 0 ? HnswNeighborSelection::Simple
                                                    : HnswNeighborSelection::Heuristic,
    });
    const std::size_t element_count = checked_size(
        checked_multiply(header.count, header.dimension, "vector count"), "vector count");
    if (count > index.ids_.max_size() || count > index.graph_.max_size() ||
        count > index.id_to_node_.max_size() || element_count > index.vectors_.max_size()) {
      fail("declared index exceeds container capacity");
    }
    index.ids_.reserve(count);
    index.id_to_node_.reserve(count);
    for (std::size_t node = 0; node < count; ++node) {
      const VectorId id = read_integer<VectorId>(input, &checksum);
      if (!index.id_to_node_.emplace(id, static_cast<HnswIndex::NodeIndex>(node)).second) {
        fail("duplicate external vector ID");
      }
      index.ids_.push_back(id);
    }

    index.vectors_.reserve(element_count);
    for (std::size_t node = 0; node < count; ++node) {
      double squared_norm = 0.0;
      for (std::size_t component = 0; component < dimension; ++component) {
        const float value = std::bit_cast<float>(read_integer<std::uint32_t>(input, &checksum));
        if (!std::isfinite(value)) {
          fail("non-finite vector component");
        }
        squared_norm += static_cast<double>(value) * static_cast<double>(value);
        index.vectors_.push_back(value);
      }
      if (std::abs(squared_norm - 1.0) > kNormTolerance) {
        fail("vector is not L2-normalized");
      }
    }

    index.graph_.reserve(count);
    std::uint32_t observed_maximum_level = 0;
    for (std::size_t node = 0; node < count; ++node) {
      const std::uint32_t level = read_integer<std::uint32_t>(input, &checksum);
      if (level > kMaximumLevel || level > header.maximum_level) {
        fail("node level exceeds the declared maximum");
      }
      observed_maximum_level = std::max(observed_maximum_level, level);
      HnswIndex::Node graph_node{std::vector<std::vector<HnswIndex::NodeIndex>>(level + 1)};
      for (std::size_t layer = 0; layer <= level; ++layer) {
        const std::uint32_t degree = read_integer<std::uint32_t>(input, &checksum);
        const std::uint64_t degree_limit =
            layer == 0 ? header.max_connections * 2 : header.max_connections;
        if (degree > degree_limit || degree >= header.count) {
          fail("adjacency degree exceeds its limit");
        }
        const auto position = input.tellg();
        const std::uint64_t file_end = checked_add(kHeaderSize, header.payload_bytes, "file size");
        if (position < 0 || static_cast<std::uint64_t>(position) > file_end ||
            checked_multiply(degree, sizeof(std::uint32_t), "adjacency bytes") >
                file_end - static_cast<std::uint64_t>(position)) {
          fail("adjacency list exceeds remaining payload bytes");
        }
        std::vector<HnswIndex::NodeIndex>& neighbors = graph_node.levels[layer];
        neighbors.reserve(degree);
        std::unordered_set<HnswIndex::NodeIndex> seen;
        seen.reserve(degree);
        for (std::uint32_t edge = 0; edge < degree; ++edge) {
          const auto neighbor = read_integer<std::uint32_t>(input, &checksum);
          if (neighbor >= count) {
            fail("neighbor node index is out of range");
          }
          if (neighbor == node) {
            fail("self-loop is not allowed");
          }
          if (!seen.insert(neighbor).second) {
            fail("duplicate neighbor in adjacency list");
          }
          neighbors.push_back(neighbor);
        }
      }
      index.graph_.push_back(std::move(graph_node));
    }
    if (observed_maximum_level != header.maximum_level) {
      fail("maximum level does not match node levels");
    }
    if (count > 0 && index.graph_[header.entry_point].levels.size() - 1 != header.maximum_level) {
      fail("entry point is not present on the maximum level");
    }
    for (std::size_t node = 0; node < count; ++node) {
      for (std::size_t layer = 0; layer < index.graph_[node].levels.size(); ++layer) {
        for (const HnswIndex::NodeIndex neighbor : index.graph_[node].levels[layer]) {
          if (index.graph_[neighbor].levels.size() <= layer) {
            fail("neighbor is not present on the referenced layer");
          }
        }
      }
    }
    if (checksum != stored_checksum) {
      fail("checksum does not match");
    }
    index.entry_point_ = count == 0 ? 0 : static_cast<HnswIndex::NodeIndex>(header.entry_point);
    index.max_level_ = checked_size(header.maximum_level, "maximum level");
    return index;
  }
};

void save_hnsw_index(const HnswIndex& index, const std::filesystem::path& path) {
  HnswIndexSerializer::save(index, path);
}

HnswIndex load_hnsw_index(const std::filesystem::path& path, const DotProductKernel kernel) {
  return HnswIndexSerializer::load(path, kernel);
}

}  // namespace fast_vector
