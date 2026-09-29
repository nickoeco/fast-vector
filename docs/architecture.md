# Architecture

## Purpose

`fast-vector` separates vector-search algorithms from transport and data preparation so each
layer can be tested and measured independently. The C++ core does not know about text,
protobuf, containers, or Kubernetes. Python tooling does not implement the search index.

```text
documents.jsonl
      |
      v
Python embedding pipeline ---> vectors.jsonl + chunks.jsonl + manifest.json
      |                                      |
      | vectors                              | VectorId lookup
      v                                      v
gRPC client ---> VectorSearch service ---> VectorStore ---> VectorIndex
                                                       |--> FlatIndex
                                                       `--> HnswIndex
```

## C++ core

`VectorIndex` is the common ownership-neutral interface for insertion, search, size, and
dimension. `FlatIndex` is the exact correctness baseline. `HnswIndex` implements approximate
search behind the same interface, so callers can select an algorithm without changing the
service contract.

Both indexes own normalized vectors in a contiguous row-major `std::vector<float>`. External
64-bit vector IDs are stored separately from compact internal positions. Search inputs are
non-owning `std::span` views; an index owns every vector after insertion.

`FlatIndex` performs an `O(Nd)` scan and retains at most `K` candidates in a heap. `HnswIndex`
uses randomized layers, greedy upper-layer descent, and a bounded layer-zero candidate search.
The exact index supplies ground truth for approximate-index recall measurements.

`VectorStore` owns one `VectorIndex`. It uses a shared mutex so multiple searches may run
concurrently while insertion remains exclusive. Index implementations themselves do not
promise concurrent insertion and search.

## Service boundary

The synchronous gRPC service validates transport input and delegates index operations to
`VectorStore`. Protobuf messages contain vector IDs, float values, results, and service
statistics; original document text remains outside the C++ process. The server enables the
standard gRPC health service and handles SIGINT/SIGTERM shutdown.

Batch insertion is prevalidated where possible but is not transactional if the underlying
index rejects an item after earlier items were inserted. Deadlines are checked at operation
boundaries; an in-progress insertion is not rolled back.

## Offline and online data flow

The embedding pipeline deterministically chunks documents, derives stable vector IDs, calls a
named Sentence Transformers model, and writes three artifacts:

- `vectors.jsonl`: vector IDs and float embeddings for service ingestion.
- `chunks.jsonl`: vector IDs and source text for result reconstruction.
- `manifest.json`: schema, model identity, dimension, normalization policy, counts, and file
  names.

The semantic-search client verifies the manifest and both data files before ingestion. Query
text is embedded with the manifest model, sent to the service, and result IDs are mapped back
to chunks. The output is retrieval evidence, not an LLM-generated answer.

## Persistence

Version 1 persistence supports complete `FlatIndex` snapshots. The format has fixed-width
little-endian fields, a magic value, version, dimensions, counts, flags, payload size, and an
FNV-1a checksum. Loading validates all structural limits before allocating large buffers.

The gRPC server can load a FlatIndex snapshot before opening its listening port and can reject
all later writes in read-only mode. The snapshot dimension is authoritative. HNSW persistence,
automatic saving, and online snapshot replacement are not implemented. The default container
and Kubernetes commands still start with an empty index unless an operator mounts a snapshot
and supplies the loading arguments.

## Deployment boundary

The multi-stage Docker build produces a non-root runtime image. Kubernetes uses native gRPC
startup, readiness, and liveness probes and a ClusterIP Service. The reference Deployment has
one replica and uses `Recreate`, because independent in-memory indexes do not provide data
replication or consistency.

## Verification layers

- GoogleTest verifies mathematical behavior, deterministic ordering, persistence validation,
  concurrency policy, HNSW invariants, and the gRPC boundary.
- NumPy supplies independent exact cosine Top-K ground truth and Recall@K validation.
- Release benchmarks separate construction, persistence, latency percentiles, QPS, recall,
  and approximate memory observations.
- Sanitizer builds check address and undefined behavior; ThreadSanitizer has a separate build.
- GitHub Actions builds GCC and Clang configurations, exercises gRPC/Python integration, and
  builds and starts the runtime container.

## Known limitations

- No deletion or incremental persistence.
- No HNSW persistence or concurrent HNSW construction.
- No transactional batch insertion.
- No TLS, authentication, authorization, or rate limiting.
- No multi-replica consistency for writable indexes; immutable read-only Flat snapshots can be
  mounted consistently only when the platform supplies the same file to every Pod.
- No crash-atomic snapshot replacement in format version 1.
- Resource limits are starting values, not production sizing recommendations.
