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

HNSW, persistence, deletion, batch operations, explicit SIMD, service APIs, and
metadata storage are not implemented yet.

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
external synchronization.

## Benchmark

Build and run in Release mode:

```bash
cmake -S . -B build-release -DCMAKE_BUILD_TYPE=Release
cmake --build build-release --parallel
./build-release/fast_vector_benchmark
```

The benchmark uses a fixed seed, 10,000 vectors, 128 dimensions, 1,000 queries, Top-10,
and a warm-up period. It reports index construction time separately from average,
P50/P95/P99 per-query latency and overall QPS. It is a single-process, single-thread,
in-memory baseline, not a production service benchmark. No performance numbers are
pre-recorded here because they depend on the machine, compiler, and build flags.

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

## Roadmap

Planned phases add persistence, then concurrency and SIMD, a from-scratch HNSW index,
gRPC, embedding pipelines, and container deployment. The exact `FlatIndex` remains the
correctness and recall baseline for those implementations.

## License

MIT
