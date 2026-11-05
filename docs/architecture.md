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

`VectorStore` uses a shared mutex for writable indexes: searches share it and insertion is
exclusive. Read-only stores atomically publish an immutable shared index pointer instead.
Each query and statistics call acquires one owning snapshot without holding the store mutex;
old queries retain the previous snapshot across replacement. C++20 atomic shared-pointer
specialization is used when available, with atomic shared-pointer functions on libstdc++ 11.
Atomic pointer operations are not necessarily lock-free. Index implementations themselves do not
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
authoritative. On Linux, an explicit read-only SIGHUP mode loads and validates a Flat or HNSW
replacement and then atomically publishes shared immutable ownership. Publication does not
wait for active queries, which retain their previous snapshot until completion. Failed reloads
preserve the current in-memory index. The configured index type stays fixed; new snapshot
dimensions and HNSW parameters are authoritative. Old and new snapshots may coexist in memory;
final-reference release performs destruction. Automatic saving and writable hot reload are not implemented.
The default container and Kubernetes commands still start with an empty index unless an operator
mounts a snapshot and supplies the loading arguments.

## Search scheduling

Search handlers validate/copy input and submit owned tasks to a fixed-size computation pool
with bounded waiting capacity. Workers acquire snapshots only on execution. Flat/HNSW cooperate
through `SearchContext` batch checks; transport contexts remain in handlers. Shutdown refuses
Search admission, cancels waiting/active tasks, drains gRPC and joins workers. Insertions retain
their direct path. An immutable publication bundles its index with a process-local version
returned by Search. See [service concurrency](service_concurrency.md) for timing and limitations.

## Read-only replica publication

Managed read-only stores atomically own a `ServingSnapshot` containing index, local counter,
content snapshot ID, release generation and embedding revision. A dedicated manager admits
one candidate, receives bounded gRPC chunks, checks SHA-256/length/format/shape, and uses the
complete binary loader before marking READY. Activation publishes the preallocated object.
Queries and their expected-ID guard use the actual snapshot acquired at execution.

The explicit replica-list publisher rolls prepares/activations and re-observes every target.
Rollback raises generation while reusing old content. Failed preparations preserve serving
data and the process-local high-water fence. There is no cross-replica atomic commit, consensus
or durable control journal. Restart uses an explicit current bootstrap configuration. See
[snapshot publication](snapshot_publication.md) for framing, fencing and recovery semantics.

## Structured service logs

The optional gRPC target owns a shared JSON Lines writer on stderr. Application handlers emit
one completion record with a process-local request ID, method, returned status, elapsed time,
index type, and observed live vector count. Atomic counters summarize handler outcomes after
shutdown drains requests. Lifecycle records cover effective startup configuration, snapshot
loading and reload, startup failure, and shutdown reason. Signal handlers only update lock-free
flags; normal threads perform logging. Payloads and raw exception messages are never logged.
See [logging](logging.md) for timing, privacy, collection, and failure semantics.

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
- No writable hot reload or concurrent HNSW construction.
- No transactional batch insertion.
- No TLS, authentication, authorization, or rate limiting.
- No writable replication. Read-only rolling publication provides observed eventual convergence,
  with no cross-replica atomicity or automatic restart recovery of the control fence.
- Direct calls to the low-level snapshot save API do not provide atomic replacement; the
  offline builder provides the durable Linux publication workflow.
- Resource limits are starting values, not production sizing recommendations.
