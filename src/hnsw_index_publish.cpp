#include <filesystem>
#include <stdexcept>

#include "fast_vector/hnsw_index_io.h"

#if defined(__linux__)
#include <fcntl.h>
#include <unistd.h>

#include <cerrno>
#include <cstdio>
#include <cstdlib>
#include <string>
#include <system_error>
#include <vector>
#endif

namespace fast_vector {

#if defined(__linux__)
namespace {

[[noreturn]] void fail_system(const char* operation) {
  throw std::system_error(errno, std::generic_category(), operation);
}

class FileDescriptor {
 public:
  explicit FileDescriptor(const int descriptor) : descriptor_(descriptor) {}
  ~FileDescriptor() { close(descriptor_); }
  FileDescriptor(const FileDescriptor&) = delete;
  FileDescriptor& operator=(const FileDescriptor&) = delete;

  [[nodiscard]] int get() const noexcept { return descriptor_; }

 private:
  int descriptor_;
};

class TemporaryPath {
 public:
  explicit TemporaryPath(const std::filesystem::path& destination) {
    std::string pattern = destination.string() + ".tmp-XXXXXX";
    std::vector<char> buffer(pattern.begin(), pattern.end());
    buffer.push_back('\0');
    const int descriptor = mkstemp(buffer.data());
    if (descriptor < 0) {
      fail_system("create temporary HNSW snapshot");
    }
    FileDescriptor file(descriptor);
    try {
      path_ = buffer.data();
    } catch (...) {
      unlink(buffer.data());
      throw;
    }
  }

  ~TemporaryPath() {
    std::error_code error;
    std::filesystem::remove(path_, error);
  }
  TemporaryPath(const TemporaryPath&) = delete;
  TemporaryPath& operator=(const TemporaryPath&) = delete;

  [[nodiscard]] const std::filesystem::path& path() const noexcept { return path_; }

 private:
  std::filesystem::path path_;
};

void sync_path(const std::filesystem::path& path, const int flags) {
  const int descriptor = open(path.c_str(), flags);
  if (descriptor < 0) {
    fail_system("open HNSW snapshot for fsync");
  }
  FileDescriptor file(descriptor);
  if (fsync(file.get()) != 0) {
    fail_system("fsync HNSW snapshot");
  }
}

}  // namespace
#endif

void publish_hnsw_index(const HnswIndex& index, const std::filesystem::path& path,
                        const bool overwrite) {
#if defined(__linux__)
  if (path.empty() || path.filename().empty()) {
    throw std::invalid_argument("HNSW snapshot path must name a file");
  }
  const std::filesystem::path parent = path.has_parent_path() ? path.parent_path() : ".";
  TemporaryPath temporary(path);
  save_hnsw_index(index, temporary.path());
  const HnswIndex verified = load_hnsw_index(temporary.path());
  if (verified.size() != index.size() ||
      verified.stats().directed_edge_count != index.stats().directed_edge_count) {
    throw std::runtime_error("published HNSW snapshot failed verification");
  }
  sync_path(temporary.path(), O_RDONLY);
  if (overwrite) {
    if (rename(temporary.path().c_str(), path.c_str()) != 0) {
      fail_system("replace HNSW snapshot");
    }
  } else {
    if (link(temporary.path().c_str(), path.c_str()) != 0) {
      fail_system("publish HNSW snapshot without overwrite");
    }
    if (unlink(temporary.path().c_str()) != 0) {
      fail_system("remove temporary HNSW snapshot link");
    }
  }
  sync_path(parent, O_RDONLY | O_DIRECTORY);
#else
  (void)index;
  (void)path;
  (void)overwrite;
  throw std::runtime_error("durable HNSW snapshot publication is supported only on Linux");
#endif
}

}  // namespace fast_vector
