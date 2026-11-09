# Memory admission and resource limits

The service uses one ResourceLimits policy for startup, insertion, recovery, search and snapshot
publication. Two levels of limits work together: index shape/estimated size, and shared runtime
accounting for read-only indexes, loading workspaces and admitted searches. These are admission
units, not an allocator measurement or a hard RSS guarantee. Writable index storage is bounded
separately by maximum_index_bytes; it is not subtracted from the shared runtime counter.

## Configuration

| Server flag | Default | Meaning |
| --- | ---: | --- |
| `--search-queue-bytes` | 67108864 | Waiting tasks' owned query capacity, identity capacity and task object |
| `--memory-budget-bytes` | 2147483648 | Shared runtime accounting limit |
| `--max-index-bytes` | 1073741824 | Per-index estimated storage limit |
| `--max-index-vectors` | 1000000 | Physical nodes, including HNSW tombstones |
| `--max-dimension` | 65536 | Index/query dimension bound |
| `--max-top-k` | 10000 | Requested K bound, before clamping to available results |

`--snapshot-memory-bytes` is an alias for `--memory-budget-bytes`. Existing message size,
file size, batch count, queue count, WAL size and worker count limits remain in effect.
All service CLI limits must be positive. Exceeding a resource cap returns RESOURCE_EXHAUSTED;
startup violations fail before the service listens. Shape/protocol errors retain their existing
validation rules. There is no wait-for-memory queue or automatic eviction.

Example:

```bash
./build-clang-grpc/fast_vector_server --index flat --dimension 128 \
  --max-index-vectors 100000 --max-index-bytes 268435456 \
  --max-top-k 100 --search-workers 2 --search-queue-capacity 64 \
  --search-queue-bytes 1048576 --memory-budget-bytes 536870912
```

## Accounting model

Every arithmetic product/sum is checked for overflow before admission. The formulas intentionally
include metadata allowances; they can reject work whose actual allocation would fit.

- Flat shape cost: `N * (4*d + 512)` bytes.
- Durable Flat cost: `N * (4*d + 1024)`, including the index and retry metadata. Historical
  original request vectors are disk-backed. Per-request/record buffers remain dimension-sized.
- Writable HNSW uses physical N and a conservative 64-level envelope per node:
  `N * (4*d + 512 + 64*(48 + 32*M))`. This may be substantially larger than typical graphs.
- Read-only HNSW shared accounting uses `N*(4*d+512) + 16*edges + 48*N*(max_level+1)`.
- Loading reservation: `16*file_bytes + 512*physical_nodes`, derived from the actual fixed header
  and file size. HNSW additionally checks this payload-based allowance against the index limit
  before allocating graph storage. Both loaders check actual header dimension/count before row
  allocation; manifest live counts cannot hide tombstoned physical nodes.
- An admitted search reserves its waiting-input charge plus `64*K + 16*maximum_vectors` units.
  The latter is a configured scratch allowance, not a measurement of every algorithm allocation.
  Queue-byte admission independently limits only tasks still waiting.

Snapshot startup and SIGHUP copy to private staging before preflight/loading, avoiding pathname
replacement between preflight and loader opens. Managed publication already uses private staging.
Loading reservations overlap the serving index, old reader-held indexes and the candidate's final
reservation; a READY candidate retains its final reservation. Copying to staging uses disk space
and does not reserve the complete file as a RAM buffer.

Read-only accounting is attached to the index's shared ownership, not just the current root.
Even a retained index reference keeps its lease. Repeated activations cannot make reader-held
old indexes disappear from the counter. Index destruction precedes credit release. Exceptions
release reservations through RAII, including failed shared_ptr control-block allocation.

Query credit travels with the native SearchOutcome through the promise/future. Completing a worker
does not immediately release native result credit if a consumer still retains that outcome. Queue
credits, in contrast, are released on dequeue, waiting cancellation or shutdown. The communication
handler validates K/dimension before copying the query; protobuf decoding has already occurred.

WAL checks new-write limits after persistent retry lookup but before appending. A full index still
accepts an exact retry. Capacity rejection does not poison the writer or advance LSN. Recovery and
checkpoint loading apply the same configured shape constraints. Batch capacity is checked for the
entire requested batch before any element mutates the index; duplicate-ID batch behavior otherwise
remains nontransactional.

## Observability and boundaries

Startup emits `resource_limits`. GetStats reports current/peak queued bytes, queue byte limit,
shared runtime accounted bytes/limit, maximum K and maximum vector count. The historical protobuf
field names `snapshot_accounted_bytes` and `snapshot_budget_bytes` refer to the shared runtime
pool, which now also includes searches. Stats are observations and need not reflect one atomic
instant across scheduler and store counters.

The budget excludes gRPC transport/protobuf buffers, thread stacks, allocator arenas, fragmented
or retained capacity, process libraries, filesystem page cache and some algorithm/build scratch.
Writable storage is separately estimated/capped, so a rough configured envelope includes BOTH
the index cap and runtime cap, plus unaccounted resources. The service does not bound all handlers
waiting in synchronous gRPC. Do not equate accounted bytes with RSS or claim OOM is impossible.
Use an OS/container memory limit for a hard process boundary. No such deployment change is made
automatically in this stage. Low-level direct index add/search APIs still require caller resource
governance; service scheduling and VectorStore provide the admission boundary.

## Verification

Tests cover checked arithmetic, failed reservations, index/old-reader lease lifetime, byte-based
queue refusal and cancellation, shared-pool refusal, batch rejection before mutation, Top-K,
loader header bounds, physical tombstone counting and failed candidate loading preserving old data.
The real gRPC WAL test starts with one-vector/one-K/one-byte-queue limits: new writes and oversized
or inadmissible searches return RESOURCE_EXHAUSTED, the original retry succeeds, and LSN and writer
health are unchanged. Existing snapshot publication, semantic search and recovery suites remain
part of regression validation.

```bash
cmake --build build-clang-grpc -j 4
ctest --test-dir build-clang-grpc --output-on-failure -j 4
cmake --build build-grpc-sanitize -j 4
ctest --test-dir build-grpc-sanitize --output-on-failure -j 4
FAST_VECTOR_SERVER=build-clang-grpc/fast_vector_server \
  build/python-grpc-venv/bin/python -m unittest -q tests/wal_recovery_test.py
```

Local validation on 2026-11-08: Clang Release CTest 141/141 passed; GCC ASan/UBSan
140 passed with the single existing Ubuntu gRPC client deadline ABI test skipped. All 18 Python
gRPC/semantic-search/publication/WAL tests passed against Release executables; all three WAL
service tests also passed against sanitizer executables. clang-format, Python compilation and
git diff checks passed. No new performance or RSS benchmark, remote CI, container or cluster
run is claimed for this change.

The complete Release Python command was:

```bash
FAST_VECTOR_SERVER=build-clang-grpc/fast_vector_server \
FAST_VECTOR_BUILDER=build-clang-grpc/fast_vector_build_index \
FAST_VECTOR_HNSW_FIXTURE=build-clang-grpc/fast_vector_hnsw_snapshot_fixture \
build/python-grpc-venv/bin/python -m unittest -q \
  tests/grpc_python_test.py tests/semantic_search_test.py \
  tests/snapshot_publication_test.py tests/wal_recovery_test.py
```

Changed files:

- Policy: `include/fast_vector/resource_limits.h`.
- Store/scheduler: `include/fast_vector/vector_store.h`, `src/vector_store.cpp`,
  `include/fast_vector/search_executor.h`, `src/search_executor.cpp`.
- Persistence: `include/fast_vector/durable_flat_index.h`, `src/durable_flat_index.cpp`,
  `include/fast_vector/flat_index_io.h`, `src/flat_index_io.cpp`,
  `include/fast_vector/hnsw_index_io.h`, `src/hnsw_index_io.cpp`.
- Service: `apps/server.cpp`, `proto/fast_vector/v1/vector_search.proto`,
  `service/snapshot_publication.h`, `service/snapshot_publication.cpp`,
  `service/vector_search_service.cpp`.
- Tests/build: `CMakeLists.txt`, `tests/test_resource_limits.cpp`,
  `tests/test_search_executor.cpp`, `tests/test_snapshot_publication.cpp`,
  `tests/test_durable_flat_index.cpp`, `tests/wal_recovery_test.py`.
- Documentation: `README.md`, `docs/memory_admission.md`.
