#include "fast_vector/flat_index_io.h"

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
#include <vector>

namespace fast_vector {
namespace {

constexpr std::array<char, 8> kMagic{'F', 'V', 'I', 'N', 'D', 'E', 'X', '\0'};
constexpr std::uint32_t kFormatVersion = 1;
constexpr std::uint32_t kHeaderSize = 56;
constexpr std::uint32_t kEndianMarker = 0x01020304U;
constexpr std::uint32_t kNormalizedCosineFlag = 1U;
constexpr std::uint64_t kFnvOffsetBasis = 14695981039346656037ULL;
constexpr std::uint64_t kFnvPrime = 1099511628211ULL;
constexpr double kNormalizedSquaredNormTolerance = 1.0e-3;

[[noreturn]] void fail(const std::string& message) {
    throw std::runtime_error("invalid fast-vector index: " + message);
}

void update_checksum(std::uint64_t& checksum, const unsigned char byte) {
    checksum ^= byte;
    checksum *= kFnvPrime;
}

template <typename UInt>
void checksum_little_endian(std::uint64_t& checksum, UInt value) {
    static_assert(std::is_unsigned_v<UInt>);
    for (std::size_t i = 0; i < sizeof(UInt); ++i) {
        update_checksum(checksum, static_cast<unsigned char>(value & 0xFFU));
        value >>= 8U;
    }
}

void write_exact(std::ostream& output, const char* data, const std::size_t size) {
    output.write(data, static_cast<std::streamsize>(size));
    if (!output) {
        throw std::runtime_error("failed to write fast-vector index");
    }
}

template <typename UInt>
void write_little_endian(std::ostream& output, UInt value) {
    static_assert(std::is_unsigned_v<UInt>);
    std::array<char, sizeof(UInt)> bytes{};
    for (std::size_t i = 0; i < bytes.size(); ++i) {
        bytes[i] = static_cast<char>(value & 0xFFU);
        value >>= 8U;
    }
    write_exact(output, bytes.data(), bytes.size());
}

void read_exact(std::istream& input, char* data, const std::size_t size) {
    input.read(data, static_cast<std::streamsize>(size));
    if (!input) {
        fail("file is truncated");
    }
}

template <typename UInt>
UInt read_little_endian(std::istream& input) {
    static_assert(std::is_unsigned_v<UInt>);
    std::array<unsigned char, sizeof(UInt)> bytes{};
    read_exact(input, reinterpret_cast<char*>(bytes.data()), bytes.size());
    UInt value = 0;
    for (std::size_t i = 0; i < bytes.size(); ++i) {
        value |= static_cast<UInt>(bytes[i]) << (8U * i);
    }
    return value;
}

template <typename UInt>
UInt read_payload_value(std::istream& input, std::uint64_t& checksum) {
    static_assert(std::is_unsigned_v<UInt>);
    std::array<unsigned char, sizeof(UInt)> bytes{};
    read_exact(input, reinterpret_cast<char*>(bytes.data()), bytes.size());
    UInt value = 0;
    for (std::size_t i = 0; i < bytes.size(); ++i) {
        update_checksum(checksum, bytes[i]);
        value |= static_cast<UInt>(bytes[i]) << (8U * i);
    }
    return value;
}

std::uint64_t checked_multiply(
    const std::uint64_t lhs, const std::uint64_t rhs, const std::string& field) {
    if (lhs != 0 && rhs > std::numeric_limits<std::uint64_t>::max() / lhs) {
        fail(field + " overflows");
    }
    return lhs * rhs;
}

std::uint64_t checked_add(
    const std::uint64_t lhs, const std::uint64_t rhs, const std::string& field) {
    if (rhs > std::numeric_limits<std::uint64_t>::max() - lhs) {
        fail(field + " overflows");
    }
    return lhs + rhs;
}

std::uint64_t payload_size(const std::uint64_t count, const std::uint64_t dimension) {
    const std::uint64_t id_bytes = checked_multiply(count, sizeof(VectorId), "ID data size");
    const std::uint64_t element_count = checked_multiply(count, dimension, "vector element count");
    const std::uint64_t vector_bytes =
        checked_multiply(element_count, sizeof(float), "vector data size");
    return checked_add(id_bytes, vector_bytes, "payload size");
}

std::size_t to_size_t(const std::uint64_t value, const std::string& field) {
    if (value > std::numeric_limits<std::size_t>::max()) {
        fail(field + " exceeds this platform's limits");
    }
    return static_cast<std::size_t>(value);
}

}  // namespace

class FlatIndexSerializer {
public:
    static void save(const FlatIndex& index, const std::filesystem::path& path) {
        const auto count = static_cast<std::uint64_t>(index.ids_.size());
        const auto dimension = static_cast<std::uint64_t>(index.dimension_);
        const std::uint64_t stored_payload_size = payload_size(count, dimension);

        std::uint64_t checksum = kFnvOffsetBasis;
        for (const VectorId id : index.ids_) {
            checksum_little_endian(checksum, id);
        }
        for (const float value : index.vectors_) {
            checksum_little_endian(checksum, std::bit_cast<std::uint32_t>(value));
        }

        std::ofstream output(path, std::ios::binary | std::ios::trunc);
        if (!output) {
            throw std::runtime_error("cannot open index for writing: " + path.string());
        }

        write_exact(output, kMagic.data(), kMagic.size());
        write_little_endian(output, kFormatVersion);
        write_little_endian(output, kHeaderSize);
        write_little_endian(output, kEndianMarker);
        write_little_endian(output, kNormalizedCosineFlag);
        write_little_endian(output, dimension);
        write_little_endian(output, count);
        write_little_endian(output, stored_payload_size);
        write_little_endian(output, checksum);
        for (const VectorId id : index.ids_) {
            write_little_endian(output, id);
        }
        for (const float value : index.vectors_) {
            write_little_endian(output, std::bit_cast<std::uint32_t>(value));
        }
        output.flush();
        if (!output) {
            throw std::runtime_error("failed to flush fast-vector index: " + path.string());
        }
    }

    static FlatIndex load(const std::filesystem::path& path) {
        std::ifstream input(path, std::ios::binary);
        if (!input) {
            throw std::runtime_error("cannot open index for reading: " + path.string());
        }

        std::array<char, kMagic.size()> magic{};
        read_exact(input, magic.data(), magic.size());
        if (magic != kMagic) {
            fail("magic does not match");
        }
        if (read_little_endian<std::uint32_t>(input) != kFormatVersion) {
            fail("unsupported format version");
        }
        if (read_little_endian<std::uint32_t>(input) != kHeaderSize) {
            fail("header size does not match version 1");
        }
        if (read_little_endian<std::uint32_t>(input) != kEndianMarker) {
            fail("endianness marker does not match");
        }
        if (read_little_endian<std::uint32_t>(input) != kNormalizedCosineFlag) {
            fail("unsupported index flags");
        }

        const std::uint64_t stored_dimension = read_little_endian<std::uint64_t>(input);
        const std::uint64_t stored_count = read_little_endian<std::uint64_t>(input);
        const std::uint64_t stored_payload_size = read_little_endian<std::uint64_t>(input);
        const std::uint64_t stored_checksum = read_little_endian<std::uint64_t>(input);
        if (stored_dimension == 0) {
            fail("dimension must be greater than zero");
        }

        const std::uint64_t expected_payload_size = payload_size(stored_count, stored_dimension);
        if (stored_payload_size != expected_payload_size) {
            fail("payload size does not match dimension and count");
        }
        const std::uint64_t expected_file_size =
            checked_add(kHeaderSize, stored_payload_size, "file size");
        if (std::filesystem::file_size(path) != expected_file_size) {
            fail("file size does not match the header");
        }

        const std::size_t dimension = to_size_t(stored_dimension, "dimension");
        const std::size_t count = to_size_t(stored_count, "vector count");
        const std::size_t element_count =
            to_size_t(checked_multiply(stored_count, stored_dimension, "vector element count"),
                      "vector element count");

        std::vector<VectorId> ids;
        std::unordered_set<VectorId> id_set;
        ids.reserve(count);
        id_set.reserve(count);
        std::uint64_t checksum = kFnvOffsetBasis;
        for (std::size_t i = 0; i < count; ++i) {
            const VectorId id = read_payload_value<VectorId>(input, checksum);
            if (!id_set.insert(id).second) {
                fail("payload contains duplicate vector IDs");
            }
            ids.push_back(id);
        }

        std::vector<float> vectors;
        vectors.reserve(element_count);
        for (std::size_t row = 0; row < count; ++row) {
            double squared_norm = 0.0;
            for (std::size_t column = 0; column < dimension; ++column) {
                const auto bits = read_payload_value<std::uint32_t>(input, checksum);
                const float value = std::bit_cast<float>(bits);
                if (!std::isfinite(value)) {
                    fail("payload contains a non-finite vector value");
                }
                vectors.push_back(value);
                const double wide_value = static_cast<double>(value);
                squared_norm += wide_value * wide_value;
            }
            if (std::abs(squared_norm - 1.0) > kNormalizedSquaredNormTolerance) {
                fail("payload contains a vector that is not L2-normalized");
            }
        }
        if (checksum != stored_checksum) {
            fail("payload checksum does not match");
        }

        FlatIndex index(dimension);
        index.ids_ = std::move(ids);
        index.id_set_ = std::move(id_set);
        index.vectors_ = std::move(vectors);
        return index;
    }
};

void save_flat_index(const FlatIndex& index, const std::filesystem::path& path) {
    FlatIndexSerializer::save(index, path);
}

FlatIndex load_flat_index(const std::filesystem::path& path) {
    return FlatIndexSerializer::load(path);
}

}  // namespace fast_vector
