# Phase 11C local verification and measurements

## Environment and method

Actual local runs on 2026-11-05 used WSL2 kernel `6.18.54.1-microsoft-standard-WSL2`, x86-64,
an Intel Core i9-13980HX, Clang 14.0.0, and CMake Release flags `-O3 -DNDEBUG`.
`findmnt` reported `/tmp` on ext4 (`/dev/sdd`) and the workspace on `/mnt/c` as 9p.
These are virtualized local storage paths, not production disk or power-loss tests.

Each run commits 1,000 inserts of dimension 128, seed 42, with uniform float input in [-1,1].
Every operation fdatasyncs; no group commit, RPC transport, competing clients, CPU isolation,
cache eviction, or repeated-run confidence interval is included. A checkpoint is taken at
LSN 500, followed by another 500 inserts. Recovery is measured first with the checkpoint and
then WAL-only against the same complete log. This is warm-cache measurement in a fixed order.
The benchmark temporarily renames its own recovery manifest for the WAL-only comparison;
it must only run in a newly created benchmark directory, never a live recovery directory.

```bash
./build-clang-grpc/fast_vector_wal_benchmark \
  /tmp/fast-vector-wal-11c-20261105 1000 128 > build/wal-benchmark-ext4.json
./build-clang-grpc/fast_vector_wal_benchmark \
  build/wal-benchmark-11c-drvfs 1000 128 > build/wal-benchmark-drvfs.json
```

Use new destination names for another run; the benchmark refuses existing directories.

## Observed results

| Metric | WSL ext4 | Workspace 9p |
| --- | ---: | ---: |
| Commit mean us | 4775.742 | 3417.054 |
| Commit P50 us | 4577.797 | 3385.979 |
| Commit P95 us | 5187.105 | 4152.069 |
| Commit P99 us | 11437.190 | 4440.188 |
| fdatasync mean us | 4754.397 | 2431.178 |
| Checkpoint wall time us | 20512.300 | 51483.946 |
| Checkpoint recovery wall time us | 4514.287 | 350644.043 |
| Suffix records applied | 500 | 500 |
| Suffix records / total recovery seconds | 110759.462 | 1425.948 |
| All validated records / total recovery seconds | 221518.924 | 2851.895 |
| WAL-only recovery wall time us | 9533.401 | 352670.633 |
| WAL-only records / recovery seconds | 104894.360 | 2835.507 |
| WAL bytes | 1084930 | 1084930 |

Commit includes validation, encoding, append, sync and in-memory application, but not input
generation. The sync submeasurement covers the fdatasync call, at integer-microsecond
resolution. Checkpoint wall time includes saving, reload verification and durable publication.
Recovery includes directory synchronization, checkpoint validation/loading, the entire WAL scan,
retry-table reconstruction, suffix application and synchronization of the accepted log prefix.
The suffix-rate column is not an isolated index-apply microbenchmark.

A single sample does not establish a filesystem ranking. In particular, the differing write
and recovery behavior cannot establish which storage stack offers stronger durability. Full
WAL retention means checkpointing does not remove log scanning or historical retry-table cost.

## Verification evidence

- Clang Release CTest: **133/133 passed**.
- GCC ASan/UBSan CTest: **132 passed, 1 existing skip**. The skip is
  `GrpcGrpcServiceTest.HonorsExpiredDeadline`, the already documented Ubuntu gRPC client ABI
  boundary under sanitizers; no new WAL test is skipped.
- Python gRPC, semantic-search, replica-publication and WAL integration suites: **17/17 passed**.
- The two WAL integration tests also pass against ASan/UBSan service executables.
- Actual post-sync `operator new` failure sweep covers **4 allocation failure points**, checks
  unchanged in-memory index contents after each failed apply, rejects later writes and verifies
  recovery of the durable operation. The boundary hook additionally models an EIO sync failure.
- Eight separate SIGKILL boundary cases cover before/partial/complete append, before/after sync,
  before/after apply and before response. Every previously acknowledged operation is recovered.
- The gRPC test confirms 100 writes before each of two real service kills, with and without a
  checkpoint, verifies exact ordered IDs and score bits after restart, retries all acknowledged
  operations, and tests 16 concurrent identical requests producing exactly one new insertion.
- Corrupt complete WAL records prevent the service from listening. Tail tests cover every cut
  position in a final record; illegal lengths and available invalid LSNs are rejected.
- Failed checkpoint publication preserves the prior manifest/WAL chain. Failure after manifest
  rename blocks writes until recovery instead of claiming the new name is durably published.

The real build/test commands were:

```bash
cmake --build build-clang-grpc -j 4
ctest --test-dir build-clang-grpc --output-on-failure -j 4
cmake --build build-grpc-sanitize -j 4
ctest --test-dir build-grpc-sanitize --output-on-failure -j 4
FAST_VECTOR_SERVER=build-clang-grpc/fast_vector_server \
FAST_VECTOR_BUILDER=build-clang-grpc/fast_vector_build_index \
FAST_VECTOR_HNSW_FIXTURE=build-clang-grpc/fast_vector_hnsw_snapshot_fixture \
build/python-grpc-venv/bin/python -m unittest -q \
  tests/grpc_python_test.py tests/semantic_search_test.py \
  tests/snapshot_publication_test.py tests/wal_recovery_test.py
FAST_VECTOR_SERVER=build-grpc-sanitize/fast_vector_server \
  build/python-grpc-venv/bin/python -m unittest -v tests/wal_recovery_test.py
```

Process termination leaves the kernel and its caches alive. No physical power-loss, kernel-crash,
device-cache-loss, remote CI, container or cluster durability test was run in this stage.

## Changed files

- Core: `include/fast_vector/durable_flat_index.h`, `src/durable_flat_index.cpp`,
  `include/fast_vector/flat_index.h`, `src/flat_index.cpp`,
  `include/fast_vector/vector_store.h`, `src/vector_store.cpp`.
- Service/client: `apps/server.cpp`, `proto/fast_vector/v1/vector_search.proto`,
  `service/vector_search_service.h`, `service/vector_search_service.cpp`, `scripts/grpc_client.py`.
- Tests: `tests/test_durable_flat_index.cpp`, `tests/test_hnsw_allocation_failure.cpp`,
  `tests/wal_recovery_test.py`.
- Build/measurement: `CMakeLists.txt`, `.github/workflows/ci.yml`, `benchmarks/benchmark_wal.cpp`.
- Documentation: `README.md`, `docs/architecture.md`, `docs/logging.md`,
  `docs/service_concurrency.md`, `docs/wal_recovery.md`, `docs/wal_recovery_measurements.md`.

Generated WAL/checkpoint files, test logs, benchmark JSON and protobuf outputs remain ignored
build or temporary artifacts.
