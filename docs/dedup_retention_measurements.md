# Finite dedup retention: local verification and measurements

Actual local Release runs on 2026-11-09 used Clang 14/WSL2 (`-O3 -DNDEBUG`), fresh `/tmp`
directories on the Linux filesystem, 1,000 vectors of dimension 128, seed 42, per-operation
fdatasync, 64 KiB segments, retention R=3 and the default 256 MiB per-epoch byte cap. The count
cap is varied below and is the binding epoch quota for these inputs. Builds had completed;
there is no CPU isolation, cache eviction or repeated-run confidence interval.

```bash
./build-clang-grpc/fast_vector_wal_benchmark \
  /tmp/fast-vector-epochs-20261109-q100 1000 128 65536 100 > build/epoch-q100.json
./build-clang-grpc/fast_vector_wal_benchmark \
  /tmp/fast-vector-epochs-20261109-q250 1000 128 65536 250 > build/epoch-q250.json
```

Use new destination names on another run. The benchmark explicitly checkpoints when the epoch
count fills and at halfway; automatic segment checkpointing keeps the epoch unchanged. It
records each operation's epoch and validates retained and expired retries after recovery. This
is orchestration in the benchmark, not automatic quota advancement by the production writer.

| Metric | 100 operations/epoch | 250 operations/epoch |
| --- | ---: | ---: |
| Index vectors preserved | 1000 | 1000 |
| Current epoch | 10 | 4 |
| Oldest retained epoch | 8 | 2 |
| Retained dedup entries | 300 | 750 |
| Successful retained retries | 300 | 750 |
| Expired requests rejected | 700 | 250 |
| Dedup metadata bytes | 25500 | 63750 |
| Current dedup checkpoint bytes | 158831 | 451232 |
| Current WAL bytes | 44885 | 15374 |
| Directory logical bytes | 702660 | 979590 |
| Recovery wall time ms | 9.100 | 10.684 |
| Suffix operations replayed | 41 | 14 |
| WAL record bytes scanned | 44813 | 15302 |
| Mean retained retry us | 1.503 | 1.436 |
| Mean commit ms | 5.064 | 5.576 |
| Commit P50 ms | 4.595 | 4.758 |
| Commit P95 ms | 5.341 | 5.404 |
| Commit P99 ms | 8.607 | 46.932 |
| Mean per-operation fdatasync ms | 4.635 | 4.870 |
| Halfway explicit checkpoint ms | 41.164 | 41.234 |
| Additional quota checkpoints | 8 | 2 |
| Total quota checkpoint ms | 355.451 | 85.455 |
| Timed ingestion loop seconds | 5.469 | 5.712 |
| Final WAL generation | 20 | 20 |
| Process RSS after recovery, KiB | 8984 | 9228 |

Metadata bytes include entry structures and encoded operation-key lengths, not map nodes,
buckets, string capacity, allocator arenas or transient buffers. In-memory keys now include
eight epoch bytes, including legacy epoch zero. RSS is the entire warm process and includes
the normalized index and prior allocation history; it is not an isolated before/after saving.
Directory sizes are logical file lengths, not allocated blocks or secure-erasure evidence.
The current dedup file holds checkpoint-covered retained entries, while newer retained requests
are still in the active WAL. It need not contain all 300/750 in-memory entries yet.

Commit timing excludes input/key generation and the benchmark's explicit quota/halfway
checkpoints, but includes automatic segment checkpoints. The ingestion-loop duration includes
input generation and all those checkpoints; initial empty-dataset migration is outside it.
Quota checkpoint costs therefore cannot be inferred from commit P99 alone. The observed
percentile difference is not a production latency guarantee or a claim that smaller windows
are universally faster. Recovery validates the complete normalized index, retained dedup file,
and suffix, then synchronizes/retries cleanup. The legacy JSON field
`validated_records_per_second` is N divided by total recovery time; in bounded mode it is an
index-scale observation, not evidence that expired request records were loaded.

Each expired refusal is checked without increasing index size or applied LSN. Exact retained
retries still read disk data after their SHA-256 filter. All 1,000 vectors remain indexed;
the bounded state is only the deduplication history. The theoretical old original-vector
payload in JSON is `1000*128*4=512000` bytes and is not a measured old RSS/full table size.
No WAL-only comparison is possible after prefix cleanup; those JSON fields are null.

## Verification evidence

- Clang Release CTest: **168/168 passed**.
- GCC ASan/UBSan CTest: **167 passed, one existing gRPC deadline ABI skip**.
- Release Python gRPC/semantic-search/publication/WAL suite: **20/20 passed**.
- Sanitizer real-service WAL suite: **5/5 passed**.
- Actual allocation-failure sweep covers **112 expiry-checkpoint failure points**, preserving the
  previous manifest, epoch window, retry sources, writer health and file descriptor count.
  A subsequent successful root expires those requests; restart cannot resurrect them.
- Four expiry-publication SIGKILL boundaries preserve either the old valid window or the new
  durable window as appropriate. Index data survives in both cases.
- Twenty successive epochs keep retained metadata/file sizes stable after warmup while all
  40 inserted vectors remain searchable.
- Tests cover exact boundaries, same ID string across epochs, quota-full retries, closed/future/
  expired requests, legacy migration and insufficient quotas, persisted policy mismatches,
  automatic rotation not advancing epochs, corrupt WAL/sidecar epochs and policy checksums,
  and an available invalid epoch in a truncated WAL tail.
- Checkpoint's expected-epoch comparison and result capture share the store lock. Native and
  real gRPC tests reject a repeated stale checkpoint without another epoch advance.
- The real service is killed and restarted without retention flags; its window persists,
  expired requests remain refused, retained retries succeed and query IDs/scores are unchanged.

Actual commands:

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
  build/python-grpc-venv/bin/python -m unittest -q tests/wal_recovery_test.py
```

No power-loss, remote CI, container or cluster run is claimed. There is no time-based TTL,
live policy change, anti-rollback backup fence, writable replication or HNSW recovery. Generated
WAL/checkpoint data, protobuf modules and reports remain ignored build/temporary artifacts.

## Changed files

- Core: `include/fast_vector/durable_flat_index.h`, `src/durable_flat_index.cpp`,
  `include/fast_vector/vector_store.h`, `src/vector_store.cpp`.
- Service/client: `apps/server.cpp`, `proto/fast_vector/v1/vector_search.proto`,
  `service/vector_search_service.cpp`, `scripts/grpc_client.py`.
- Tests/build: `CMakeLists.txt`, `tests/test_dedup_epochs.cpp`,
  `tests/test_hnsw_allocation_failure.cpp`, `tests/wal_recovery_test.py`.
- Benchmark: `benchmarks/benchmark_wal.cpp`.
- Documentation: `README.md`, `docs/architecture.md`, `docs/dedup_checkpoint.md`,
  `docs/wal_recovery.md`, `docs/wal_rotation.md`, `docs/dedup_retention.md`,
  `docs/dedup_retention_measurements.md`.
