# Disk-backed deduplication stage 1: local verification

Actual local Release runs on 2026-11-09 used the existing Clang 14/WSL2 build (`-O3 -DNDEBUG`)
and `/tmp` on the Linux filesystem. Each invocation starts a fresh process and new directory.
Count is 1,000, seed 42, checkpoint LSN 500, with per-operation fdatasync. Input is uniform
float32 in [-1,1]. There is no CPU isolation, cache eviction or repeated-run confidence interval.
The checkpoint-backed recovery is followed by WAL-only recovery in the same process.

```bash
./build-clang-grpc/fast_vector_wal_benchmark \
  /tmp/fast-vector-dedup-20261109-d128 1000 128 > build/dedup-d128.json
./build-clang-grpc/fast_vector_wal_benchmark \
  /tmp/fast-vector-dedup-20261109-d768 1000 768 > build/dedup-d768.json
```

Use new names for another run: existing benchmark directories are refused. The benchmark
temporarily moves only its own fixture manifest for the WAL-only comparison. Never run this
procedure against a live recovery directory.

| Metric | Dimension 128 | Dimension 768 |
| --- | ---: | ---: |
| In-memory dedup metadata bytes, 1,000 entries | 76,890 | 76,890 |
| Removed original-vector payload baseline, calculated bytes | 512,000 | 3,072,000 |
| Dedup checkpoint bytes, 500 entries | 302,446 | 1,582,446 |
| WAL file bytes | 1,084,930 | 6,204,930 |
| WAL record bytes scanned with checkpoint | 542,500 | 3,102,500 |
| Checkpoint wall time ms | 36.306 | 48.748 |
| Checkpoint-backed recovery wall time ms | 4.240 | 23.198 |
| WAL-only recovery wall time ms | 8.376 | 22.681 |
| Suffix records applied / total recovery seconds | 117,918.660 | 21,553.962 |
| Restored operations / total recovery seconds | 235,837.320 | 43,107.924 |
| WAL-only records / total recovery seconds | 119,388.103 | 44,088.853 |
| Mean retry us | 1.544 | 5.710 |
| Current process RSS after checkpoint-backed recovery, KiB | 8,476 | 12,244 |
| Mean new commit us | 4,261.487 | 4,486.358 |
| New commit P99 us | 4,848.323 | 7,302.964 |

Metadata bytes count entry structures and operation-ID sizes, excluding string capacities,
map nodes/buckets, allocator overhead and temporary record buffers. The old vector payload is
a formula `N*d*4`, not a measured old process or a complete old retry table. Identical metadata
across dimensions demonstrates the new representation; it does not imply identical total RAM.
The index still stores N normalized vectors. RSS is for the warm entire process and includes
construction/checkpoint allocator history, not isolated metadata or a before/after delta.

Half the operations are restored from the dedup checkpoint and half from WAL; every one is
still validated through its authoritative source. Recovery includes index/dedup validation and
loading, metadata construction, suffix application and WAL sync. The benchmark's historical
`validated_records_per_second` JSON key now counts restored operations, not full WAL scans.
Retry timing excludes input regeneration and operation-key construction; it includes SHA-256,
disk pread, source fingerprint verification and exact byte comparison. Both prefix and suffix
operations are retried. Checkpoint timing includes paired file creation, readback verification,
next-table allocation and durable manifest publication.

The 768-dimensional checkpoint-backed run is slightly slower than WAL-only. This stage removes
retained original vectors from the table and old-prefix WAL scanning; it does not guarantee
faster recovery. The full dedup file remains validated and read. Disk usage grows by adding
dedup checkpoints while retaining WAL and older snapshots; rotation/retention are later stages.

## Verified behavior

- Clang Release CTest: **148/148 passed**.
- GCC ASan/UBSan CTest: **147 passed, 1 existing skip** (`GrpcGrpcServiceTest.HonorsExpiredDeadline`,
  the already documented Ubuntu gRPC client deadline ABI boundary).
- Release Python gRPC/semantic-search/publication/WAL suite: **18/18 passed**.
- Sanitizer service WAL suite: **3/3 passed**.
- Dedicated actual-allocation fault sweep: **100 paired-checkpoint failure points**, each checking
  unchanged previous manifest, usable old retry sources, unchanged logical index, writer health
  and unchanged `/proc/self/fd` count. A later successful checkpoint and restart restore both IDs.
- The two additional checkpoint SIGKILL cases recover all previously acknowledged writes, using
  the old paired checkpoint plus suffix before publication, and the new pair after rename.
- Legacy version-1 fixture recovery scans WAL, preserves retries and upgrades on checkpoint.
- A version-2 test corrupts checkpoint-covered WAL record bytes and confirms successful recovery
  and retries from the authoritative checkpoint, while only suffix bytes are scanned.
- Dedup structure fields are rejected even with recomputed record/file/manifest checksums.
  Post-load raw-data corruption is detected on retry and blocks subsequent writes.
- Existing post-sync allocation, signed-zero, corruption and deterministic search tests pass.

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

Formatting, Python syntax and Git whitespace checks are part of final verification. Build logs,
generated protobuf, WAL/checkpoint files and benchmark JSON remain ignored build/temporary artifacts.
No power-loss, remote CI, container or cluster run is claimed. Verification results were recorded
before committing. SHA-256 uses the already available OpenSSL installation; no package
was installed during this local task.

## Changed files

- Build/dependencies: `CMakeLists.txt`, `.github/workflows/ci.yml`.
- Core: `src/durable_flat_index.cpp`, `include/fast_vector/durable_flat_index.h`,
  `include/fast_vector/resource_limits.h`.
- Service contract/stats: `proto/fast_vector/v1/vector_search.proto`, `service/vector_search_service.cpp`.
- Tests: `tests/test_durable_flat_index.cpp`, `tests/test_hnsw_allocation_failure.cpp`,
  `tests/wal_recovery_test.py`.
- Measurement: `benchmarks/benchmark_wal.cpp`.
- Documentation: `README.md`, `docs/architecture.md`, `docs/memory_admission.md`,
  `docs/wal_recovery.md`, `docs/dedup_checkpoint.md`, `docs/dedup_checkpoint_measurements.md`.
