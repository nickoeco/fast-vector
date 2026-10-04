# fast-vector

`fast-vector` is a small C++20 vector search engine built to make index design,
correctness, and performance measurable. It includes an exact in-memory flat index and
a from-scratch approximate HNSW index using cosine similarity.

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
- Selectable scalar, compiler-auto-vectorized, and optional AVX2 dot-product kernels
- In-memory single-threaded HNSW construction and approximate cosine search
- Thread-safe `VectorStore` application layer with serialized insertion and concurrent search
- Optional synchronous gRPC server with protobuf code generation and standard health checks

HNSW persistence, deletion, authentication, TLS, and metadata storage are not implemented yet.
Persistence currently means complete `FlatIndex` snapshot save/load; incremental
persistence is not implemented. HNSW supports both simple nearest-M and diversity-aware
neighbor selection, with reproducible Recall@K validation against NumPy ground truth.
The gRPC transport supports insertion, non-transactional batch insertion, search, statistics,
deadlines, bounded request messages, and the standard gRPC health protocol.

## Requirements

- CMake 3.20 or newer
- A C++20 compiler (GCC or Clang)
- Network access during the first test configuration: CMake FetchContent downloads
  GoogleTest 1.15.2. The library and demo have no runtime third-party dependencies.
- Optional gRPC build: protobuf compiler and development files, the gRPC C++ development
  package, and `grpc_cpp_plugin`. On Ubuntu 22.04/WSL2 these are available through
  `protobuf-compiler`, `protobuf-compiler-grpc`, `libprotobuf-dev`, and `libgrpc++-dev`.

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

## HNSW index

`HnswIndex` stores normalized vectors in the same contiguous row-major representation as
`FlatIndex`, while graph nodes refer to them through compact 32-bit internal indexes.
User-provided 64-bit IDs remain separate from graph topology. Random levels are generated
from a fixed seed, high layers use greedy descent, and layer zero uses bounded candidate
and result queues controlled by `ef_search`.

The implementation follows the algorithm described by Malkov and Yashunin in
[Efficient and robust approximate nearest neighbor search using Hierarchical Navigable
Small World graphs](https://arxiv.org/abs/1603.09320). This repository implements its own
level generation, graph traversal, bidirectional linking, bounded candidate search, and
degree pruning; it does not wrap `hnswlib`, FAISS, or another vector-search library.

The configuration can select either simple nearest-`M` connections or the paper's
diversity-aware rule. Layer zero permits up to `2M` connections while sparse upper layers
permit `M`. `M`, `ef_construction`, and `ef_search` are explicit configuration values.
The per-call search overload raises an `ef_search` below `k` to `k`, because at least `k`
retained candidates are required to return `k` results.

## Thread safety

`FlatIndex` contains no internal synchronization. Concurrent calls to `search()` are
safe only while no thread is calling `add()`. Calling `add()` concurrently with any
other operation, or calling `add()` from multiple threads, is unsupported and requires
external synchronization. `BatchSearcher` owns persistent workers and performs concurrent
read-only calls while preserving input query order. The referenced index must outlive the
searcher and remain immutable. Destroying a `BatchSearcher` concurrently with `search()` is
unsupported.

`HnswIndex` also contains no internal synchronization. Concurrent read-only searches are
safe only after construction has finished. Insertion concurrent with search or another
insertion is unsupported.

## Application layer

`VectorStore` owns one `VectorIndex` and supplies the synchronization boundary needed by a
future service adapter. Searches take a shared lock, so multiple read-only requests can run
concurrently. Single and batch insertion take an exclusive lock and therefore block all
searches for the duration of the mutation. Counters track successful and failed searches
plus vectors inserted through the store; they do not infer vectors that may have existed in
an index before ownership was transferred.

Batch insertion checks the configured batch limit, dimensions, finite values, zero vectors,
and duplicate IDs within the request before taking the write lock. This validation performs
an extra normalization pass but prevents those request-local errors from partially changing
the index. A collision with an ID already stored in the index can still fail after earlier
batch elements were inserted, so BatchAdd is explicitly not transactional in this version.

The transport-neutral protobuf contract lives in
`proto/fast_vector/v1/vector_search.proto`. Generated protobuf files are intentionally not
committed. The core build keeps gRPC disabled by default and therefore requires no protobuf
or gRPC installation.

## gRPC service

Install the optional build dependencies on Ubuntu 22.04/WSL2, then configure explicitly:

```bash
sudo apt update
sudo apt install protobuf-compiler protobuf-compiler-grpc libprotobuf-dev libgrpc++-dev

cmake -S . -B build-grpc \
  -DCMAKE_BUILD_TYPE=Release \
  -DFAST_VECTOR_BUILD_GRPC=ON
cmake --build build-grpc --parallel
ctest --test-dir build-grpc --output-on-failure
```

Generated C++ protobuf sources stay under the build directory. CMake first uses installed
gRPC package targets when available and otherwise falls back to `pkg-config`, which supports
the Ubuntu 22.04 packages. With those system packages, the expired-deadline integration test
is skipped only in an ASan/UBSan build because gRPC 1.30's uninstrumented `ClientContext`
cleanup is incompatible with the instrumented caller; Debug and Release builds execute it.

Start an empty HNSW-backed server with a fixed dimension:

```bash
./build-grpc/fast_vector_server \
  --address 0.0.0.0:50051 \
  --index hnsw \
  --dimension 128 \
  --max-batch-size 1000 \
  --max-message-bytes 16777216 \
  --kernel auto \
  --m 16 \
  --ef-construction 200 \
  --ef-search 100 \
  --hnsw-seed 42 \
  --neighbor-selection heuristic
```

Use `--index flat` for exact search; HNSW-only options are then ignored. The synchronous
server lets gRPC schedule independent requests concurrently. `VectorStore` permits concurrent
searches but serializes each insertion or batch against all other operations. RPC input errors
map to `INVALID_ARGUMENT`, oversized configured batches to `RESOURCE_EXHAUSTED`, and expired
deadlines to `DEADLINE_EXCEEDED`. Writes to a read-only store map to `FAILED_PRECONDITION`.
Message size is bounded separately by
`--max-message-bytes`. Batch insertion is not transactional when an ID already exists in the
index, as described above.

The server can load a version 1 FlatIndex snapshot before accepting requests. The loaded file
defines the index dimension; `--dimension` is used only when constructing an empty index.
Read-only mode rejects both insertion RPCs while retaining search and statistics:

```bash
./build-grpc/fast_vector_server \
  --address 0.0.0.0:50051 \
  --index flat \
  --load-index /data/index.fv \
  --read-only true \
  --reload-on-sighup true \
  --kernel auto
```

`--load-index` currently requires `--index flat`. A loaded index remains writable unless
`--read-only true` is supplied. The server does not save later mutations automatically.
`GetStats` reports the read-only state; snapshot vectors contribute to `vector_count` but not
to the process-local `inserted_vectors` counter.

On Linux, `--reload-on-sighup true` enables zero-downtime snapshot replacement for this
read-only configuration. Publish a verified replacement to the same `--load-index` path with
`fast_vector_build_index --overwrite true`, then signal the server:

```bash
kill -HUP <server-pid>
```

The server loads and validates the file before taking the store's exclusive lock, so disk I/O
does not block searches. The lock only protects the final index-pointer swap: searches already
holding the shared lock finish on the old index, and later searches use the new index. Query
counters survive the reload; snapshot vectors still do not count as process-local insertions.
If loading fails, the server logs the error, keeps serving the previous in-memory index, and
remains healthy. Multiple SIGHUP notifications may coalesce into one reload. Hot reload is not
available for writable stores, empty indexes, HNSW, or non-Linux platforms.

The server enables gRPC's standard health-check service. Transport security, authentication,
HNSW loading, persistence during service operation, reflection, and production observability
remain outside this phase.

The Python integration suite also exercises the complete snapshot-serving path: it builds a
FlatIndex snapshot from `tests/data/vectors.jsonl`, starts a server on a dynamic local port,
checks the standard gRPC health RPC, verifies deterministic search and statistics, and confirms
that both single and batch writes fail with `FAILED_PRECONDITION` in read-only mode. It then
atomically replaces the snapshot, signals a live reload, and verifies that a corrupt subsequent
snapshot cannot replace the valid in-memory index. Set
`FAST_VECTOR_SERVER` and `FAST_VECTOR_BUILDER` when the executables are outside `build-grpc`.

### Python client and service load test

Keep the Python gRPC toolchain isolated and generate the client modules from the checked-in
contract:

```bash
python3 -m venv build/python-grpc-venv
build/python-grpc-venv/bin/python -m pip install -r scripts/requirements-grpc.txt
build/python-grpc-venv/bin/python scripts/generate_grpc_python.py \
  --output-dir build/python-grpc
```

With `fast_vector_server` running on `127.0.0.1:50051`, the command-line client can insert,
batch insert from JSON, search, and read service statistics:

```bash
build/python-grpc-venv/bin/python scripts/grpc_client.py add \
  --id 101 --vector 1,0,0

build/python-grpc-venv/bin/python scripts/grpc_client.py search \
  --query 1,0,0 --k 10

build/python-grpc-venv/bin/python scripts/grpc_client.py stats
```

`batch-add --input vectors.json` expects a JSON array such as
`[{"id": 101, "values": [1.0, 0.0, 0.0]}]`. Global options such as `--address`,
`--generated-dir`, and `--timeout` must appear before the subcommand.

The load-test client can optionally own the local server process, which makes one command
reproducible and guarantees cleanup:

```bash
build/python-grpc-venv/bin/python scripts/benchmark_grpc.py \
  --server-executable build-grpc-release/fast_vector_server \
  --address 127.0.0.1:50052 \
  --generated-dir build/python-grpc \
  --index hnsw \
  --vectors 10000 \
  --dimension 128 \
  --queries 1000 \
  --k 10 \
  --workers 4 \
  --batch-size 500 \
  --warmup 20 \
  --seed 20250908
```

The JSON report separates index population time from concurrent query wall time and reports
successful and failed requests, errors grouped by gRPC status, QPS, and average/P50/P95/P99
client-observed latency. The measurements include serialization, localhost transport, gRPC
scheduling, lock contention, and search. This is a controlled development benchmark, not a
production capacity claim, and the README intentionally contains no pre-recorded numbers.

## Embedding dataset pipeline

Phase 6 begins with an offline, reproducible boundary between text processing and vector
serving. Input is UTF-8 JSONL with one unique `document_id`, `text`, and optional `source`
per line. Install the optional model dependency in a separate environment and build an
example dataset:

```bash
python3 -m venv build/embedding-venv
build/embedding-venv/bin/python -m pip install -r scripts/requirements-embedding.txt
build/embedding-venv/bin/python scripts/build_embedding_dataset.py \
  --input examples/documents.jsonl \
  --output-dir build/embedding-example \
  --model sentence-transformers/all-MiniLM-L6-v2 \
  --maximum-words 120 \
  --overlap-words 20 \
  --batch-size 32 \
  --device cpu
```

Whitespace-normalized text is split into deterministic overlapping word windows. Each vector
ID is a stable 64-bit BLAKE2b digest of `(document_id, chunk_id)`; collisions are checked
within the generated dataset. Sentence Transformers returns normalized float embeddings,
which are validated again before export. The default
[`all-MiniLM-L6-v2`](https://huggingface.co/sentence-transformers/all-MiniLM-L6-v2)
model produces 384-dimensional embeddings.

The output keeps vector-index data and application chunk records separate:

- `vectors.jsonl` contains only vector IDs and float arrays for ingestion.
- `chunks.jsonl` maps IDs to chunk records. Each record contains the text payload plus
  descriptive metadata such as document ID, chunk number, and source.
- `manifest.json` records format version, model, dimension, chunk settings, and filenames.

Model weights are downloaded from Hugging Face on first use and are not committed. Word-based
chunk sizes are intentionally simple and measurable; they are not model-token limits.

Build the Python gRPC modules once, then run the complete semantic-search path. Supplying a
server executable creates an empty index with the manifest dimension, ingests `vectors.jsonl`,
embeds the query with the manifest model, searches, maps result IDs through `chunks.jsonl`, and
stops the server:

```bash
build/embedding-venv/bin/python -m pip install -r scripts/requirements-grpc.txt
build/embedding-venv/bin/python scripts/generate_grpc_python.py \
  --output-dir build/python-grpc
build/embedding-venv/bin/python scripts/semantic_search.py \
  --dataset-dir build/embedding-example \
  --query "Which index provides exact search results?" \
  --k 3 \
  --batch-size 500 \
  --address 127.0.0.1:50054 \
  --server-executable build-grpc-release/fast_vector_server \
  --index flat \
  --generated-dir build/python-grpc \
  --device cpu
```

The loader rejects unsupported manifests, duplicate IDs, wrong dimensions, count mismatches,
and any difference between the vector and chunk ID sets before modifying the service. Use
`--skip-ingest` only when connecting to a service already populated with exactly the same
dataset; repeating ingestion against the same in-memory index produces duplicate-ID errors.
The output is retrieval evidence—scores and matched chunks—not a generated answer. LLM-based
RAG remains outside the implemented scope.

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
  --kernel avx2 \
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

The default `scalar` kernel accumulates in `double` and remains the strict correctness
reference. The `auto` kernel is isolated in one translation unit compiled with fast
floating-point optimization so GCC or Clang may vectorize its float reduction. The optional
`avx2` kernel uses explicit eight-float SIMD operations and scalar tail handling. AVX2 is
compiled only for x86 targets and selected only after a runtime CPU capability check; asking
for it when unavailable returns an error. Optimized reductions may differ from the scalar
reference by small floating-point rounding amounts, so tests compare scores with a tolerance
and require identical Top-K IDs on deterministic inputs.

## NumPy correctness validation

The validation script generates deterministic float32 database and query vectors, writes
them as inspectable CSV files, computes exact cosine Top-K results with NumPy, runs the C++
`FlatIndex` on the same inputs, and reports Recall@K plus the maximum score difference.
NumPy is a validation-only dependency:

```bash
python3 -m pip install -r scripts/requirements.txt
python3 scripts/validate_recall.py \
  --runner build-release/fast_vector_validate \
  --output-dir validation-output \
  --kernel scalar
```

The generated directory contains the inputs, NumPy ground truth, C++ results, and a JSON
summary. It is ignored by Git. Exact `FlatIndex` validation requires Recall@K of `1.0` for
every query, an identical ranked ID sequence, and a maximum absolute score difference no
greater than `1e-5`. Both sides use descending score and ascending vector ID as the
deterministic tie-break rule.

The same script can evaluate approximate HNSW results against the exact NumPy baseline:

```bash
python3 scripts/validate_recall.py \
  --runner build-release/fast_vector_validate \
  --output-dir validation-output/hnsw \
  --index hnsw \
  --vector-count 10000 \
  --query-count 100 \
  --dimension 128 \
  --k 10 \
  --m 16 \
  --ef-construction 200 \
  --ef-search 200 \
  --neighbor-selection heuristic \
  --minimum-recall 0.85
```

Recall@K is the fraction of exact Top-K IDs also present in the approximate Top-K,
averaged across queries. It ignores rank within the Top-K set. The optional minimum is an
explicit validation threshold, not a universal HNSW guarantee; data distribution, insertion
order, parameters, and seed all affect recall. Scores are independently checked against the
exact cosine score of each ID actually returned by HNSW.

## HNSW benchmark

The dedicated Release benchmark builds `FlatIndex` and `HnswIndex` from the same generated
vectors. Flat search supplies both the exact Top-K baseline and its own latency measurement.
HNSW is then queried at each requested `efSearch` value:

```bash
cmake -S . -B build-release -DCMAKE_BUILD_TYPE=Release
cmake --build build-release --parallel
./build-release/fast_vector_hnsw_benchmark \
  --vectors 10000 \
  --dimension 128 \
  --queries 1000 \
  --k 10 \
  --m 16 \
  --ef-construction 200 \
  --ef-search 10,50,100,200 \
  --neighbor-selection heuristic \
  --seed 20250908 \
  --hnsw-seed 42
```

The output separates Flat and HNSW construction time, Flat latency and QPS, and an HNSW
table containing mean/minimum Recall@K, average latency, P50/P95/P99, and QPS for every
`efSearch`. It also reports graph degree counters, vector/ID payload bytes, adjacency-edge
payload bytes, and approximate process RSS deltas. Payload values intentionally exclude
container, hash-table, allocator, and alignment overhead; RSS includes unrelated process
memory and allocator behavior. Neither value is an exact retained-memory measurement.

Queries run sequentially in-process, so these figures measure a single-thread algorithmic
tradeoff rather than concurrent service latency. Results are not pre-recorded because they
depend on CPU, compiler, kernel, workload, parameters, and build flags.

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
or pointers, and loading rebuilds the duplicate-ID set. The low-level `save_flat_index()` API
writes exactly the requested path and does not provide crash-atomic replacement. Use the offline
builder when publishing or replacing a deployable snapshot.

### Offline FlatIndex builder

`fast_vector_build_index` converts the embedding pipeline's `vectors.jsonl` into a deployable
FlatIndex snapshot without loading the whole JSONL file into a second in-memory collection:

```bash
./build-release/fast_vector_build_index \
  --input embedding-output/vectors.jsonl \
  --output embedding-output/index.fv \
  --dimension 384 \
  --expected-count 10000
```

The input is one strict JSON object per non-empty line with exactly `id` and `values` members.
Member order and JSON whitespace may vary, but unknown or duplicate members, escaped member
names, invalid JSON numbers, IDs outside `uint64`, dimension mismatches, duplicate IDs,
non-finite values, and zero vectors are rejected with a line number. The dimension and expected
count should come from the embedding manifest.

The builder uses `FlatIndex::add()` for validation and normalization and writes a uniquely named
temporary file beside the requested output. It reloads that file with the production loader
before publishing it. By default, publication fails if the output already exists, including
when another builder wins a concurrent publication race.

On Linux, explicitly replace an existing snapshot with `--overwrite true`:

```bash
./build-release/fast_vector_build_index \
  --input embedding-output/vectors.jsonl \
  --output embedding-output/index.fv \
  --dimension 384 \
  --expected-count 10000 \
  --overwrite true
```

The Linux publication path flushes the verified temporary file with `fsync()`, atomically
renames it over the previous snapshot, and then `fsync()`s the parent directory. A crash before
the rename leaves the previous snapshot intact; after the rename, readers opening the path see
the complete new snapshot. Existing open file descriptors may continue reading the old inode.
These guarantees require a local filesystem that correctly implements atomic rename and
`fsync()` durability. `--overwrite true` is rejected on non-Linux platforms.

Successful output reports vector count, dimension, file size, and total build time. The timing
includes JSONL parsing, index construction, snapshot writing, and verification; it is not a
query-performance benchmark.

## Container deployment

The multi-stage Docker build compiles the optional gRPC server on Ubuntu 22.04 and copies only
the server plus its runtime libraries into the final image. The process runs as the non-root
UID `10001`; build tools, source files, tests, and generated protobuf sources are absent from
the runtime stage.

```bash
docker build -t fast-vector:local .
docker run --rm -p 50051:50051 fast-vector:local \
  --address 0.0.0.0:50051 \
  --index flat \
  --dimension 384 \
  --max-batch-size 1000 \
  --kernel auto
```

Serve an existing FlatIndex snapshot without allowing divergent writes:

```bash
docker run --rm -p 50051:50051 \
  --mount type=bind,source="$(pwd)/index.fv",target=/data/index.fv,readonly \
  fast-vector:local \
  --address 0.0.0.0:50051 \
  --index flat \
  --load-index /data/index.fv \
  --read-only true \
  --kernel auto
```

The image includes a TCP liveness check. Docker Compose also supplies the documented HNSW
defaults, graceful SIGTERM shutdown, and restart policy:

```bash
docker compose --project-name fast-vector config --quiet
docker compose --project-name fast-vector up --build
docker compose --project-name fast-vector down
```

The health check verifies that the server accepts TCP connections on port 50051; it does not
issue the standard gRPC health RPC. The default image command starts an empty in-memory HNSW
index and uses insecure gRPC. Snapshot files must be mounted explicitly and are never written
by read-only mode.

## Kubernetes deployment

The manifests in `deploy/kubernetes` define a single-replica Deployment and an internal
ClusterIP Service. Before applying them, build or publish the image and change
`fast-vector:local` to an image reference available to the cluster. For a local cluster, load
the locally built image using the command provided by that cluster implementation.

```bash
kubectl kustomize deploy/kubernetes
kubectl apply -k deploy/kubernetes
kubectl rollout status deployment/fast-vector
kubectl port-forward service/fast-vector 50051:50051
```

Kubernetes uses the server's standard gRPC health service for startup, readiness, and liveness
probes. The Pod runs as UID/GID `10001`, drops Linux capabilities, disables privilege
escalation and service-account token mounting, uses a read-only root filesystem, and declares
initial CPU and memory requests and limits. These resource values are safe starting points,
not measured production sizing recommendations.

The Deployment intentionally uses one replica with the `Recreate` strategy. Each process owns
an independent in-memory index, and writes are not replicated between Pods, so scaling this
manifest would produce inconsistent query results. A production multi-replica deployment
requires immutable snapshot loading or an external replication/coordinator design. The native
gRPC probe requires Kubernetes 1.27 or newer.

To serve a FlatIndex snapshot, mount it from storage as a read-only file and replace the
container arguments with `--index flat --load-index /data/index.fv --read-only true` plus the
address, batch-size, and kernel options. Storage provisioning is cluster-specific, so the base
manifest does not assume a PersistentVolumeClaim or storage class. Replicas that mount the
same immutable snapshot may answer consistent reads, but runtime insertion must remain disabled.

## Architecture and reproducibility

- [`docs/architecture.md`](docs/architecture.md) explains ownership, module boundaries, data
  flow, persistence, deployment, verification, and known limitations.
- [`docs/reproducible-experiments.md`](docs/reproducible-experiments.md) provides an experiment
  protocol and reporting checklist without pre-filled performance claims.
- [`docs/project-showcase.md`](docs/project-showcase.md) contains an evidence-based resume
  description, demonstration sequence, and interview questions.

Version tags matching the CMake project version publish the runtime image to GitHub Container
Registry. For example, after changing and committing `project(... VERSION 0.2.0)`, creating the
tag `v0.2.0` triggers `.github/workflows/release.yml`. A mismatched tag fails before publishing.
No release tag is created automatically.

## Roadmap

Planned work adds transport security, HNSW persistence, production observability, and writable
multi-replica consistency. The exact `FlatIndex` remains the correctness and recall baseline
for approximate indexes.

## License

MIT
