#include <atomic>
#include <chrono>
#include <cstddef>
#include <cstdint>
#include <filesystem>
#include <fstream>
#include <limits>
#include <stdexcept>
#include <string>
#include <vector>

#include <gtest/gtest.h>

#include "fast_vector/flat_index_io.h"

namespace {

class TemporaryIndexFile {
public:
    TemporaryIndexFile() {
        static std::atomic<unsigned int> sequence{0};
        const auto timestamp = std::chrono::steady_clock::now().time_since_epoch().count();
        path_ = std::filesystem::temp_directory_path() /
                ("fast-vector-test-" + std::to_string(timestamp) + "-" +
                 std::to_string(sequence.fetch_add(1)) + ".fv");
    }

    ~TemporaryIndexFile() {
        std::error_code error;
        std::filesystem::remove(path_, error);
    }

    TemporaryIndexFile(const TemporaryIndexFile&) = delete;
    TemporaryIndexFile& operator=(const TemporaryIndexFile&) = delete;

    [[nodiscard]] const std::filesystem::path& path() const noexcept { return path_; }

private:
    std::filesystem::path path_;
};

void overwrite_u32(const std::filesystem::path& path, const std::streamoff offset,
                   const std::uint32_t value) {
    std::fstream file(path, std::ios::binary | std::ios::in | std::ios::out);
    ASSERT_TRUE(file);
    const char bytes[] = {
        static_cast<char>(value & 0xFFU),
        static_cast<char>((value >> 8U) & 0xFFU),
        static_cast<char>((value >> 16U) & 0xFFU),
        static_cast<char>((value >> 24U) & 0xFFU),
    };
    file.seekp(offset);
    file.write(bytes, sizeof(bytes));
    ASSERT_TRUE(file);
}

void overwrite_u64(const std::filesystem::path& path, const std::streamoff offset,
                   std::uint64_t value) {
    std::fstream file(path, std::ios::binary | std::ios::in | std::ios::out);
    ASSERT_TRUE(file);
    char bytes[8]{};
    for (std::size_t i = 0; i < sizeof(bytes); ++i) {
        bytes[i] = static_cast<char>(value & 0xFFU);
        value >>= 8U;
    }
    file.seekp(offset);
    file.write(bytes, sizeof(bytes));
    ASSERT_TRUE(file);
}

void rewrite_payload_checksum(const std::filesystem::path& path) {
    constexpr std::streamoff kHeaderSize = 56;
    constexpr std::streamoff kChecksumOffset = 48;
    constexpr std::uint64_t kFnvOffsetBasis = 14695981039346656037ULL;
    constexpr std::uint64_t kFnvPrime = 1099511628211ULL;

    std::ifstream input(path, std::ios::binary);
    ASSERT_TRUE(input);
    input.seekg(kHeaderSize);
    std::uint64_t checksum = kFnvOffsetBasis;
    char byte = 0;
    while (input.get(byte)) {
        checksum ^= static_cast<unsigned char>(byte);
        checksum *= kFnvPrime;
    }
    overwrite_u64(path, kChecksumOffset, checksum);
}

fast_vector::FlatIndex make_index() {
    fast_vector::FlatIndex index(3);
    index.add(90, std::vector<float>{1.0F, 2.0F, 3.0F});
    index.add(10, std::vector<float>{-2.0F, 1.0F, 0.5F});
    index.add(40, std::vector<float>{0.0F, -1.0F, 2.0F});
    return index;
}

TEST(FlatIndexIoTest, RoundTripPreservesExactResultsAndIndexState) {
    TemporaryIndexFile file;
    const auto original = make_index();
    const std::vector<float> query{0.5F, 1.0F, 2.0F};
    const auto expected = original.search(query, 3);

    fast_vector::save_flat_index(original, file.path());
    auto restored = fast_vector::load_flat_index(file.path());

    EXPECT_EQ(restored.dimension(), original.dimension());
    EXPECT_EQ(restored.size(), original.size());
    EXPECT_EQ(restored.search(query, 3), expected);
    EXPECT_THROW(restored.add(90, std::vector<float>{1.0F, 0.0F, 0.0F}),
                 std::invalid_argument);
    restored.add(100, std::vector<float>{1.0F, 0.0F, 0.0F});
    EXPECT_EQ(restored.size(), 4U);
}

TEST(FlatIndexIoTest, RoundTripSupportsEmptyIndex) {
    TemporaryIndexFile file;
    const fast_vector::FlatIndex original(7);

    fast_vector::save_flat_index(original, file.path());
    const auto restored = fast_vector::load_flat_index(file.path());

    EXPECT_EQ(restored.dimension(), 7U);
    EXPECT_EQ(restored.size(), 0U);
}

TEST(FlatIndexIoTest, RejectsBadMagic) {
    TemporaryIndexFile file;
    fast_vector::save_flat_index(make_index(), file.path());
    overwrite_u32(file.path(), 0, 0U);
    EXPECT_THROW(static_cast<void>(fast_vector::load_flat_index(file.path())),
                 std::runtime_error);
}

TEST(FlatIndexIoTest, RejectsUnsupportedVersion) {
    TemporaryIndexFile file;
    fast_vector::save_flat_index(make_index(), file.path());
    overwrite_u32(file.path(), 8, 2U);
    EXPECT_THROW(static_cast<void>(fast_vector::load_flat_index(file.path())),
                 std::runtime_error);
}

TEST(FlatIndexIoTest, RejectsInvalidDimension) {
    TemporaryIndexFile file;
    fast_vector::save_flat_index(make_index(), file.path());
    overwrite_u32(file.path(), 24, 0U);
    overwrite_u32(file.path(), 28, 0U);
    EXPECT_THROW(static_cast<void>(fast_vector::load_flat_index(file.path())),
                 std::runtime_error);
}

TEST(FlatIndexIoTest, RejectsOverflowingPayloadSize) {
    TemporaryIndexFile file;
    fast_vector::save_flat_index(make_index(), file.path());
    overwrite_u64(file.path(), 32, std::numeric_limits<std::uint64_t>::max());
    EXPECT_THROW(static_cast<void>(fast_vector::load_flat_index(file.path())),
                 std::runtime_error);
}

TEST(FlatIndexIoTest, RejectsDuplicateIdsEvenWithAValidChecksum) {
    constexpr std::streamoff kPayloadOffset = 56;
    TemporaryIndexFile file;
    fast_vector::save_flat_index(make_index(), file.path());

    std::fstream stream(file.path(), std::ios::binary | std::ios::in | std::ios::out);
    ASSERT_TRUE(stream);
    char first_id[sizeof(fast_vector::VectorId)]{};
    stream.seekg(kPayloadOffset);
    stream.read(first_id, sizeof(first_id));
    ASSERT_TRUE(stream);
    stream.seekp(kPayloadOffset + static_cast<std::streamoff>(sizeof(first_id)));
    stream.write(first_id, sizeof(first_id));
    ASSERT_TRUE(stream);
    stream.close();
    rewrite_payload_checksum(file.path());

    EXPECT_THROW(static_cast<void>(fast_vector::load_flat_index(file.path())),
                 std::runtime_error);
}

TEST(FlatIndexIoTest, RejectsTruncatedAndTrailingData) {
    TemporaryIndexFile truncated;
    fast_vector::save_flat_index(make_index(), truncated.path());
    std::filesystem::resize_file(truncated.path(), std::filesystem::file_size(truncated.path()) - 1);
    EXPECT_THROW(static_cast<void>(fast_vector::load_flat_index(truncated.path())),
                 std::runtime_error);

    TemporaryIndexFile trailing;
    fast_vector::save_flat_index(make_index(), trailing.path());
    std::ofstream output(trailing.path(), std::ios::binary | std::ios::app);
    output.put('\0');
    output.close();
    EXPECT_THROW(static_cast<void>(fast_vector::load_flat_index(trailing.path())),
                 std::runtime_error);
}

TEST(FlatIndexIoTest, RejectsChecksumMismatch) {
    TemporaryIndexFile file;
    fast_vector::save_flat_index(make_index(), file.path());
    const auto last_byte = static_cast<std::streamoff>(std::filesystem::file_size(file.path()) - 1);
    std::fstream stream(file.path(), std::ios::binary | std::ios::in | std::ios::out);
    stream.seekg(last_byte);
    char value = 0;
    stream.get(value);
    stream.seekp(last_byte);
    stream.put(static_cast<char>(value ^ 0x01));
    stream.close();

    EXPECT_THROW(static_cast<void>(fast_vector::load_flat_index(file.path())),
                 std::runtime_error);
}

}  // namespace
