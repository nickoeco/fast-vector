# fast-vector

`fast-vector` is a small C++20 vector search engine built to make index design,
correctness, and performance measurable. The current implementation is an exact
in-memory flat index using cosine similarity.

## Implemented features

- Fixed-dimension `VectorIndex` interface and `FlatIndex` implementation
- User-provided 64-bit vector IDs with duplicate rejection
- L2 normalization on insertion and query
- Contiguous row-major `std::vector<float>` storage
- Exact scalar dot-product scan
- Deterministic heap-based Top-K selection
- Unit tests, command-line demo, and a reproducible single-thread benchmark
- Cross-language correctness validation against exact NumPy ground truth
- Versioned binary save/load for `FlatIndex`, with structural and checksum validation
- Ordered batch search backed by a reusable fixed-size worker pool

HNSW, deletion, batch operations, explicit SIMD, service APIs, and metadata storage are
not implemented yet. Persistence currently means complete snapshot save/load; incremental
persistence is not implemented.

## Requirements

- CMake 3.20 or newer
- A C++20 compiler (GCC or Clang)
- Network access during the first test configuration: CMake FetchContent downloads
  GoogleTest 1.15.2. The library and demo have no runtime third-party dependencies.

## Build and test

```bash
cmake -S . -B build -DCMAKE_BUILD_TYPE=Debug
cmake --build build --parallel
ctest --test-dir build --output-on-failure
./build/fast_vector_demo
```

To build the library and demo without downloading GoogleTest:

```bash
cmake -S . -B build -DFAST_VECTOR_BUILD_TESTS=OFF
cmake --build build --parallel
```

On Linux with GCC or Clang, enable AddressSanitizer and UndefinedBehaviorSanitizer:

```bash
cmake -S . -B build-sanitize \
  -DCMAKE_BUILD_TYPE=Debug \
  -DFAST_VECTOR_ENABLE_SANITIZERS=ON \
  -DFAST_VECTOR_BUILD_BENCHMARKS=OFF
cmake --build build-sanitize --parallel
ctest --test-dir build-sanitize --output-on-failure
```

## Usage

```cpp
#include <vector>

#include "fast_vector/flat_index.h"

fast_vector::FlatIndex index(3);
index.add(1001, std::vector<float>{1.0F, 0.0F, 0.0F});
index.add(1002, std::vector<float>{0.8F, 0.2F, 0.0F});

const auto results = index.search(std::vector<float>{1.0F, 0.0F, 0.0F}, 2);
```

Results are ordered by decreasing score. Equal scores are ordered by increasing ID.
`k == 0` returns an empty result without validating the query, and `k > size()` returns
all stored vectors. Dimension mismatches, duplicate IDs, zero vectors, and non-finite
values are rejected with `std::invalid_argument`.

## Storage and algorithm

All normalized vectors occupy one contiguous row-major array. For dimension `d`, the
vector at row `i` begins at `vectors.data() + i * d`. Compared with a vector of vectors,
this removes per-row allocations and improves locality during a full scan. The tradeoffs
are fixed dimensions, whole-buffer reallocations during growth, and more complicated
deletion; deletion is outside the current scope.

For a vector `x`, L2 normalization is:

```text
x_hat = x / ||x||_2
```

Cosine similarity becomes a dot product after both stored and query vectors are
normalized:

```text
cos(q, x) = q_hat dot x_hat
```

Searching `N` vectors of dimension `d` costs `O(Nd)`. A heap containing at most `K`
candidates adds `O(N log K)` selection work and uses `O(K)` extra memory. Only those
`K` candidates are sorted at the end, so the implementation does not allocate and sort
all `N` scores.

## Thread safety

`FlatIndex` contains no internal synchronization. Concurrent calls to `search()` are
safe only while no thread is calling `add()`. Calling `add()` concurrently with any
other operation, or calling `add()` from multiple threads, is unsupported and requires
external synchronization. `BatchSearcher` owns persistent workers and performs concurrent
read-only calls while preserving input query order. The referenced index must outlive the
searcher and remain immutable. Destroying a `BatchSearcher` concurrently with `search()` is
unsupported.

## Benchmark

Build and run in Release mode:

```bash
cmake -S . -B build-release -DCMAKE_BUILD_TYPE=Release
cmake --build build-release --parallel
./build-release/fast_vector_benchmark
```

The default run uses a fixed seed, 10,000 vectors, 128 dimensions, 1,000 queries, Top-10,
and a warm-up period. Parameters can be changed without editing source code:

```bash
./build-release/fast_vector_benchmark \
  --vectors 50000 \
  --dimension 256 \
  --queries 500 \
  --k 20 \
  --warmup 20 \
  --threads 4 \
  --seed 20250908
```

The output separates index construction, snapshot save, snapshot load, average and
P50/P95/P99 scalar-query latency, scalar QPS, batch wall time, parallel batch QPS, and
speedup. It also reports the snapshot file size and, on Linux, approximate
resident-set-size deltas. RSS deltas include allocator and container overhead and are not
an exact index-memory measurement. Batch QPS measures an in-process fixed query set and
does not claim per-request tail latency under concurrent service load. No performance
numbers are pre-recorded here because they depend on the machine, compiler, storage, and
build flags.

## NumPy correctness validation

The validation script generates deterministic float32 database and query vectors, writes
them as inspectable CSV files, computes exact cosine Top-K results with NumPy, runs the C++
`FlatIndex` on the same inputs, and reports Recall@K plus the maximum score difference.
NumPy is a validation-only dependency:

```bash
python3 -m pip install -r scripts/requirements.txt
python3 scripts/validate_recall.py \
  --runner build-release/fast_vector_validate \
  --output-dir validation-output
```

The generated directory contains the inputs, NumPy ground truth, C++ results, and a JSON
summary. It is ignored by Git. Exact `FlatIndex` validation requires Recall@K of `1.0` for
every query, an identical ranked ID sequence, and a maximum absolute score difference no
greater than `1e-5`. Both sides use descending score and ascending vector ID as the
deterministic tie-break rule.

## Binary index persistence

`FlatIndex` snapshots preserve normalized float32 vectors and IDs exactly:

```cpp
#include "fast_vector/flat_index_io.h"

fast_vector::save_flat_index(index, "example.fv");
fast_vector::FlatIndex restored = fast_vector::load_flat_index("example.fv");
```

Version 1 uses a fixed little-endian layout:

```text
8 bytes   magic: "FVINDEX\\0"
4 bytes   format version
4 bytes   header size
4 bytes   endianness marker
4 bytes   normalized-cosine flags
8 bytes   dimension
8 bytes   vector count
8 bytes   payload size
8 bytes   FNV-1a payload checksum
N * 8     vector IDs
N * d * 4 normalized float32 components
```

The loader rejects unknown versions or flags, invalid dimensions and sizes, truncated or
trailing data, checksum mismatches, duplicate IDs, non-finite components, and vectors that
are not L2-normalized. The checksum detects accidental corruption; FNV-1a is not a
cryptographic authenticity mechanism. The format does not serialize C++ container objects
or pointers, and loading rebuilds the duplicate-ID set. Snapshot writes are not crash-atomic
in version 1, so callers should write to a new path before replacing an important snapshot.

## Roadmap

Planned phases add persistence, then concurrency and SIMD, a from-scratch HNSW index,
gRPC, embedding pipelines, and container deployment. The exact `FlatIndex` remains the
correctness and recall baseline for those implementations.

## License

MIT
