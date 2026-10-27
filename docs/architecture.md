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

`FlatIndex::erase()` uses an ID-to-row hash map and swap-and-pop to keep storage dense. The
moved row keeps its external ID, and score/ID result ordering does not depend on row order.
Flat snapshots retain their version 1 layout and store only live rows. `HnswIndex::erase()`
keeps physical nodes and edges as tombstones. Navigation can use them while the result queue
admits only live nodes. Both concrete indexes support deletion, but the common interface and
service do not expose it. HNSW size is the live count; graph statistics distinguish physical,
live, and deleted nodes. Offline rebuild preserves live IDs and normalized vector bits while
creating new topology in ID order. See [deletion.md](deletion.md) for the workflow.

HNSW insertion keeps a rollback journal of original adjacency lists before each mutation,
including reverse-link edits. Exceptions restore these lists, remove any appended node/ID/vector,
erase the newly inserted map entry, and restore RNG/entry state without allocating. This is
a strong logical guarantee; container capacity may change. Rebuild returns a new index and
leaves its source unchanged even if construction fails.

`VectorStore` owns one `VectorIndex`. It uses a shared mutex so multiple searches may run
concurrently while insertion remains exclusive. Index implementations themselves do not
promise concurrent mutation and search, including deletion.

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

The offline `fast_vector_build_index` tool streams `vectors.jsonl`, delegates vector validation
and normalization to `FlatIndex`, writes a same-directory temporary snapshot, and reloads it
before publication. On Linux, publication flushes the file, uses an atomic filesystem operation
that either refuses an existing target or explicitly replaces it, and flushes the parent
directory. This creates the immutable artifact consumed by read-only service deployments
without teaching the C++ search core about documents or chunk metadata.

The semantic-search client verifies the manifest and both data files before ingestion. Query
text is embedded with the manifest model, sent to the service, and result IDs are mapped back
to chunks. The output is retrieval evidence, not an LLM-generated answer.

## Persistence

FlatIndex version 1 and HNSW version 3 persistence use distinct magic values, fixed-width
little-endian fields, and an FNV-1a checksum. The HNSW loader also accepts versions 1/2. The HNSW
format preserves graph parameters, seed, neighbor policy, external IDs, normalized vector rows,
levels, adjacency lists, maximum level, and entry point. Version 2 additionally preserves the
current RNG state, allowing deterministic insertion continuation with a compatible standard
library; version 1 resumes from the initial seed instead. Version 3 adds live count and
tombstone flags under the checksum; old formats load as all-live. The loader validates bounded sizes
before allocation and checks every graph reference and layer relationship. On Linux,
`publish_hnsw_index` validates a same-directory temporary snapshot before durable publication;
explicit overwrite atomically replaces the destination.

The gRPC server can load a FlatIndex or HNSW snapshot before opening its listening port and can
reject all later writes in read-only mode. The snapshot dimension and HNSW configuration are
authoritative. On Linux, an explicit Flat-only SIGHUP mode loads and validates a replacement
outside the store lock, then swaps index ownership under an exclusive lock. Failed reloads
preserve the current in-memory index. Automatic saving and HNSW hot reload are not implemented.
The default container and Kubernetes commands still start with an empty index unless an operator
mounts a snapshot and supplies the loading arguments.

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

- No online HNSW compaction, service deletion API, or incremental persistence.
- No HNSW hot reload or concurrent HNSW construction.
- No transactional batch insertion.
- No TLS, authentication, authorization, or rate limiting.
- No multi-replica consistency for writable indexes; immutable read-only snapshots can be
  mounted consistently only when the platform supplies the same file to every Pod.
- Direct calls to the low-level snapshot save API do not provide atomic replacement; the
  offline builder provides the durable Linux publication workflow.
- Resource limits are starting values, not production sizing recommendations.
