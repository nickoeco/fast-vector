#include <cerrno>
#include <charconv>
#include <chrono>
#include <cstddef>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <limits>
#include <optional>
#include <stdexcept>
#include <string>
#include <string_view>
#include <system_error>

#if defined(__linux__)
#include <fcntl.h>
#include <unistd.h>
#endif

#include "fast_vector/flat_index.h"
#include "fast_vector/flat_index_io.h"
#include "tools/vector_jsonl.h"

namespace {

struct BuildOptions {
  std::filesystem::path input_path;
  std::filesystem::path output_path;
  std::size_t dimension = 0;
  std::optional<std::size_t> expected_count;
  bool overwrite = false;
};

std::size_t parse_size(const std::string_view value, const std::string& option,
                       const bool allow_zero) {
  std::uint64_t parsed = 0;
  const auto [end, error] = std::from_chars(value.data(), value.data() + value.size(), parsed);
  if (error != std::errc{} || end != value.data() + value.size() || (!allow_zero && parsed == 0) ||
      parsed > std::numeric_limits<std::size_t>::max()) {
    throw std::invalid_argument("invalid value for " + option);
  }
  return static_cast<std::size_t>(parsed);
}

bool parse_bool(const std::string_view value, const std::string& option) {
  if (value == "true") {
    return true;
  }
  if (value == "false") {
    return false;
  }
  throw std::invalid_argument("invalid value for " + option + ": expected true or false");
}

BuildOptions parse_arguments(const int argc, char* argv[]) {
  BuildOptions options;
  if (argc == 2 && std::string_view(argv[1]) == "--help") {
    return options;
  }
  for (int index = 1; index < argc; index += 2) {
    if (index + 1 >= argc) {
      throw std::invalid_argument("every option requires a value");
    }
    const std::string option = argv[index];
    const std::string_view value = argv[index + 1];
    if (option == "--input") {
      options.input_path = value;
    } else if (option == "--output") {
      options.output_path = value;
    } else if (option == "--dimension") {
      options.dimension = parse_size(value, option, false);
    } else if (option == "--expected-count") {
      options.expected_count = parse_size(value, option, true);
    } else if (option == "--overwrite") {
      options.overwrite = parse_bool(value, option);
    } else {
      throw std::invalid_argument("unknown option: " + option);
    }
  }
  if (options.input_path.empty() || options.output_path.empty() || options.dimension == 0) {
    throw std::invalid_argument("--input, --output, and --dimension are required");
  }
  return options;
}

void print_usage() {
  std::cerr << "Usage: fast_vector_build_index --input vectors.jsonl --output index.fv "
               "--dimension D [--expected-count N] [--overwrite true|false]\n";
}

#if defined(__linux__)
class FileDescriptor {
 public:
  explicit FileDescriptor(const int descriptor) : descriptor_(descriptor) {}
  ~FileDescriptor() {
    if (descriptor_ >= 0) {
      ::close(descriptor_);
    }
  }

  FileDescriptor(const FileDescriptor&) = delete;
  FileDescriptor& operator=(const FileDescriptor&) = delete;

  [[nodiscard]] int get() const noexcept { return descriptor_; }

 private:
  int descriptor_;
};

[[noreturn]] void throw_system_error(const std::string& action, const std::filesystem::path& path) {
  const int error = errno;
  throw std::runtime_error(action + ": " + path.string() + ": " + std::strerror(error));
}

void sync_file(const std::filesystem::path& path) {
  const FileDescriptor descriptor(::open(path.c_str(), O_RDONLY | O_CLOEXEC));
  if (descriptor.get() < 0) {
    throw_system_error("failed to open snapshot for fsync", path);
  }
  if (::fsync(descriptor.get()) != 0) {
    throw_system_error("failed to fsync snapshot", path);
  }
}

void sync_parent_directory(const std::filesystem::path& path) {
  std::filesystem::path parent = path.parent_path();
  if (parent.empty()) {
    parent = ".";
  }
  const FileDescriptor descriptor(::open(parent.c_str(), O_RDONLY | O_DIRECTORY | O_CLOEXEC));
  if (descriptor.get() < 0) {
    throw_system_error("failed to open output directory for fsync", parent);
  }
  if (::fsync(descriptor.get()) != 0) {
    throw_system_error("failed to fsync output directory", parent);
  }
}

void publish_snapshot(const std::filesystem::path& temporary, const std::filesystem::path& output,
                      const bool overwrite) {
  sync_file(temporary);
  if (overwrite) {
    if (::rename(temporary.c_str(), output.c_str()) != 0) {
      throw_system_error("failed to atomically replace output", output);
    }
  } else {
    if (::link(temporary.c_str(), output.c_str()) != 0) {
      if (errno == EEXIST) {
        throw std::runtime_error("output already exists: " + output.string());
      }
      throw_system_error("failed to atomically publish output", output);
    }
    std::error_code remove_error;
    if (!std::filesystem::remove(temporary, remove_error)) {
      throw std::runtime_error("failed to remove published temporary file: " +
                               remove_error.message());
    }
  }
  sync_parent_directory(output);
}
#else
void publish_snapshot(const std::filesystem::path& temporary, const std::filesystem::path& output,
                      const bool overwrite) {
  if (overwrite) {
    throw std::runtime_error("--overwrite true is supported only on Linux");
  }
  if (std::filesystem::exists(output)) {
    throw std::runtime_error("output already exists: " + output.string());
  }
  std::filesystem::rename(temporary, output);
}
#endif

class TemporaryOutput {
 public:
  explicit TemporaryOutput(const std::filesystem::path& output) {
    const auto stamp = std::chrono::steady_clock::now().time_since_epoch().count();
    path_ = output;
    path_ += ".tmp-" + std::to_string(stamp);
    if (std::filesystem::exists(path_)) {
      throw std::runtime_error("temporary output already exists: " + path_.string());
    }
  }

  ~TemporaryOutput() {
    if (!committed_) {
      std::error_code error;
      std::filesystem::remove(path_, error);
    }
  }

  TemporaryOutput(const TemporaryOutput&) = delete;
  TemporaryOutput& operator=(const TemporaryOutput&) = delete;

  [[nodiscard]] const std::filesystem::path& path() const noexcept { return path_; }
  void commit() noexcept { committed_ = true; }

 private:
  std::filesystem::path path_;
  bool committed_ = false;
};

void build_index(const BuildOptions& options) {
  if (!std::filesystem::is_regular_file(options.input_path)) {
    throw std::runtime_error("input is not a regular file: " + options.input_path.string());
  }
  if (!options.overwrite && std::filesystem::exists(options.output_path)) {
    throw std::runtime_error("output already exists: " + options.output_path.string());
  }

  std::ifstream input(options.input_path);
  if (!input) {
    throw std::runtime_error("failed to open input: " + options.input_path.string());
  }

  const auto start = std::chrono::steady_clock::now();
  fast_vector::FlatIndex index(options.dimension);
  std::string line;
  std::size_t line_number = 0;
  while (std::getline(input, line)) {
    ++line_number;
    if (fast_vector::tools::is_json_whitespace_only(line)) {
      continue;
    }
    try {
      fast_vector::tools::ParsedVectorRecord record =
          fast_vector::tools::parse_vector_json_line(line);
      index.add(record.id, record.values);
    } catch (const std::exception& error) {
      throw std::runtime_error("line " + std::to_string(line_number) + ": " + error.what());
    }
  }
  if (!input.eof()) {
    throw std::runtime_error("failed while reading input");
  }
  if (options.expected_count.has_value() && index.size() != *options.expected_count) {
    throw std::runtime_error("vector count does not match --expected-count");
  }

  TemporaryOutput temporary(options.output_path);
  fast_vector::save_flat_index(index, temporary.path());
  const fast_vector::FlatIndex verified = fast_vector::load_flat_index(temporary.path());
  if (verified.size() != index.size() || verified.dimension() != index.dimension()) {
    throw std::runtime_error("written snapshot failed structural verification");
  }
  const std::uintmax_t file_size = std::filesystem::file_size(temporary.path());
  publish_snapshot(temporary.path(), options.output_path, options.overwrite);
  temporary.commit();

  const auto elapsed =
      std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - start);
  std::cout << "input=" << options.input_path.string() << '\n'
            << "output=" << options.output_path.string() << '\n'
            << "vectors=" << index.size() << '\n'
            << "dimension=" << index.dimension() << '\n'
            << "file_bytes=" << file_size << '\n'
            << "build_ms=" << elapsed.count() << '\n';
}

}  // namespace

int main(const int argc, char* argv[]) {
  try {
    if (argc == 2 && std::string_view(argv[1]) == "--help") {
      print_usage();
      return 0;
    }
    build_index(parse_arguments(argc, argv));
    return 0;
  } catch (const std::exception& error) {
    std::cerr << "Index build failed: " << error.what() << '\n';
    print_usage();
    return 1;
  }
}
