# WAL rotation stage: local verification and measurements

Actual Release runs on 2026-11-09 used the existing Clang 14/WSL2 build (`-O3 -DNDEBUG`),
with fresh directories under `/tmp` on the Linux filesystem. Each run inserts 1,000 vectors
of dimension 128, seed 42, uniform input in [-1,1], fdatasync per operation, and an explicit
checkpoint at LSN 500. Automatic checkpoints occur when the next record would exceed the
configured active segment cap. There is no cache eviction, CPU isolation or confidence interval.
The final runs below occurred after concurrent build/test processes completed.

```bash
./build-clang-grpc/fast_vector_wal_benchmark \
  /tmp/fast-vector-rotation-final-20261109-64k 1000 128 65536 > build/rotation-final-64k.json
./build-clang-grpc/fast_vector_wal_benchmark \
  /tmp/fast-vector-rotation-final-20261109-256k 1000 128 262144 > build/rotation-final-256k.json
```

Use new directory names to repeat. The benchmark refuses an existing directory.

| Metric | 64 KiB segment cap | 256 KiB segment cap |
| --- | ---: | ---: |
| Mean new commit ms | 5.352 | 4.822 |
| Commit P50 ms | 4.598 | 4.492 |
| Commit P95 ms | 5.304 | 5.050 |
| Commit P99 ms | 43.437 | 11.444 |
| Mean operation fdatasync ms | 4.648 | 4.590 |
| Explicit halfway checkpoint ms | 44.104 | 39.534 |
| Final WAL generation | 17 | 5 |
| Final checkpoint LSN | 980 | 982 |
| Replayed suffix operations | 20 | 18 |
| Startup wall time ms | 8.444 | 9.466 |
| Suffix records / total startup seconds | 2368.515 | 1901.501 |
| Restored operations / total startup seconds | 118425.743 | 105638.921 |
| Mean exact retry us | 1.501 | 1.541 |
| Dedup metadata bytes | 76890 | 76890 |
| Current dedup file bytes | 592846 | 594056 |
| Startup WAL record bytes scanned | 21700 | 19530 |
| Final active WAL bytes | 21772 | 19602 |
| Successful file unlinks during construction | 49 | 13 |
| Final directory logical bytes | 1124442 | 1124522 |
| WAL bytes before halfway checkpoint | 21772 | 19602 |
| WAL bytes after halfway checkpoint | 72 | 72 |
| Directory bytes before halfway checkpoint | 561942 | 562022 |
| Directory bytes after halfway checkpoint | 562742 | 562742 |
| Warm process RSS after startup, KiB | 9068 | 9140 |

Commit timing excludes input generation and key construction, but includes any automatically
triggered checkpoint/cleanup. The fdatasync submeasurement is only the operation's WAL sync.
Checkpoint timing includes paired snapshot/dedup generation, validation, new segment creation,
durable root publication, writer adoption and cleanup. Startup includes selected-root restoration,
suffix replay/sync and cleanup. Retry timing includes both checkpoint-backed and current-segment
requests. No network is included in these microbenchmarks.

The smaller cap caused more stop-the-writer checkpoints in this sample, and its P99 includes
their pauses. Percentiles do not bound the slowest operation, and this single run is not a
production capacity claim. Index and dedup files continue growing with historical operations;
reclaiming old WAL does not make total directory size equal to the segment cap. Directory bytes
sum logical file lengths, not allocated blocks or a measured filesystem free-space increase.
RSS covers the whole warm process and allocator history, not isolated index or dedup memory.

WAL-only recovery metrics are null in rotation mode because obsolete prefix segments have been
removed. The benchmark does not remove the authoritative manifest and attempt an invalid fallback.
Full JSON reports remain ignored build artifacts. Earlier smoke measurements, made alongside
compilation, are not the figures in this table.

## Verification

- Clang Release CTest: **157/157 passed**.
- GCC ASan/UBSan CTest: **156 passed, one existing gRPC deadline ABI skip**.
- Release Python gRPC, semantic-search, replica-publication and WAL suites: **19/19 passed**.
- Sanitizer real-service WAL suite: **4/4 passed**.
- Actual rotating-checkpoint allocation sweep: **123 failure points**, retaining the previous
  manifest, retry sources, writer health and file descriptor count. Legacy paired checkpoint
  sweep still covers 100 failure points.
- Seven SIGKILL cases cover new segment creation, pre-publication, post-rename, both writer
  switch boundaries and both cleanup boundaries. All acknowledged operations recover, retry
  without duplication and permit the next global LSN. Startup removes obsolete/temp artifacts.
- Real gRPC service test confirms 100 inserts, checkpoints, appends LSN 101, SIGKILLs and restarts
  without the enable flag. Results/scores match, all old retries succeed, exactly one suffix
  operation replays and later checkpoints continue segmented mode.
- Explicit writer-switch exceptions fail closed; cleanup exceptions leave the valid new writer
  usable and are retried at startup. Unknown filenames remain untouched. Shared writable cleanup
  directories are rejected.
- Missing/corrupt active segments, wrong start/identity, partial active tails, paired-file caps,
  oversize records, repeated same-LSN checkpoints and configurable wide dimensions are tested.
- Missing segmented roots do not create an empty legacy replacement. An abandoned first
  publication with intact full legacy WAL remains recoverable, including subsequent legacy writes.
- clang-format, Python syntax and Git whitespace checks passed. Builds emitted no compiler warnings.

Commands actually used:

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

No power-loss, remote CI, container or cluster run is claimed. No epoch, operation expiry or
old-binary downgrade is implemented. Legacy mode remains available for directories not migrated.

## Files changed in this stage

- Core: `src/durable_flat_index.cpp`, `include/fast_vector/durable_flat_index.h`.
- Service: `apps/server.cpp`, `proto/fast_vector/v1/vector_search.proto`, `service/vector_search_service.cpp`.
- Tests: `tests/test_durable_flat_index.cpp`, `tests/test_hnsw_allocation_failure.cpp`,
  `tests/wal_recovery_test.py`.
- Benchmark: `benchmarks/benchmark_wal.cpp`.
- Documentation: `README.md`, `docs/architecture.md`, `docs/wal_recovery.md`,
  `docs/dedup_checkpoint.md`, `docs/wal_rotation.md`, `docs/wal_rotation_measurements.md`.

Verification covers the combined dedup-checkpoint and rotation implementation, including the
preceding stage's dependency/build changes. The preceding stage's measurements are recorded
separately in `dedup_checkpoint_measurements.md`.
