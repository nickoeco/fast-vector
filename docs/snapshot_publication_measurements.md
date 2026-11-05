# Phase 11B local acceptance measurements

These are actual local Release acceptance runs on 2026-11-04, not capacity benchmarks.
The build uses Clang Release in WSL2. Each index type starts three separate server processes
on individual localhost ports. An HTTP artifact server and continuously querying clients run
on the same host. There is no CPU isolation or repeated-run confidence interval.

Flat uses two 2-dimensional live vectors initially and one after forward publication; files
are 88 and 72 bytes. HNSW uses three 3-dimensional live vectors initially (four physical nodes,
one tombstone) and two after publication; files are 6662 and 6576 bytes. Both HNSW files use
format 3. These tiny fixtures verify protocol behavior and result identity, not index scalability.

## Successful release and rollback

Preparation is the mean of the three server-reported preparation times. Activation-store
times were all 0 at integer-microsecond resolution, meaning less than 1 us, not zero RPC cost.
The RPC column is the observed client wall-time range for the three activation calls.
RSS is the largest replica lifetime peak reported by Linux `getrusage(RUSAGE_SELF)` in KiB;
it is not live-only RSS, candidate memory, or a per-release increment.

| Index | Operation | Generation | Preparation mean ms | Atomic store us | Activation RPC range us | Observed convergence ms | Max peak RSS KiB |
| --- | --- | --- | --- | --- | --- | --- | --- |
| Flat | Forward | 2 | 2.183 | <1 | 426–554 | 15.691 | 44528 |
| Flat | Roll back to original bytes | 6 | 1.367 | <1 | 362–449 | 12.792 | 44528 |
| HNSW | Forward | 2 | 1.779 | <1 | 370–401 | 15.843 | 45296 |
| HNSW | Roll back to original bytes | 6 | 1.516 | <1 | 350–522 | 13.006 | 45296 |

Forward generation 2 points at artifact B. Rollback generation 6 points at the original
artifact A's unchanged snapshot ID. Final cleanup rollout uses A at generation 8. Rollback
does not decrease the fence or reuse an old deployment generation.

## Failure and continuity observations

- Corrupt and truncated streams preserve generation 2 on all replicas. Corruption is rejected
  by SHA-256; truncated/aborted transfers never become READY.
- A valid artifact/manifest for the other index type is rejected without changing serving data.
- During the generation 4 rollout, replica 1 receives an HTTP 404 while replicas 0 and 2 succeed.
  The report records current generations `[4, 2, 4]`, `converged: false`, and null convergence
  time. Its failed preparation time is recovered from status: 1352 us for Flat and 1222 us
  for HNSW in this run. This is not reported as a successful zero-cost preparation.
- Retrying the same generation with the same manifest updates the failed replica and skips
  the already-current replicas. A matching READY candidate is also reused without another
  download, even when that replica's otherwise-selected download URL is intentionally missing.
- A paused stream proves a second prepare is refused while old queries remain usable. Wrong
  candidate activation and delayed older commands are refused. A stale command is not called
  converged merely because its older data version is still currently serving.
- Expected-ID mismatch returns `FAILED_PRECONDITION`; matching queries return original data
  after rollback. Health stays SERVING while a valid old publication is retained.
- The final Release run observed 1988 background queries across the Flat replicas and 2200
  across HNSW replicas, with zero recorded errors or result/identity mismatches. Every response
  was checked against its own snapshot ID, generation, embedding revision and expected ID.
- All child service processes exit successfully, observer/HTTP threads finish, and replica
  staging directories are empty after preparation/activation/failure paths.

The complete records, including manifest, individual replica times, RPC wall times, errors,
current/candidate identity and high-water generation, are generated under:

- `build/publication-11b/flat-publication.json`
- `build/publication-11b/hnsw-publication.json`

Sanitizer process runs use a separate `build/publication-11b-asan/` output directory; their
memory and timing values are not mixed into this Release table. The reproduction command
is in [the protocol document](snapshot_publication.md#verification). Reports remain ignored
build artifacts; no snapshots or generated protobuf files are committed.

## Validation

Clang Release: 118/118 CTest cases passed. GCC ASan/UBSan: 117 passed, with the existing
Ubuntu gRPC client-deadline ABI test skipped. The three-process Flat/HNSW scenario also passed
against ASan/UBSan service executables. The Python publication, existing gRPC and semantic-search
suite passed 15/15 tests. A queued expected-ID/reload scenario passed 20 repetitions.

These checks confirm local behavior. They do not demonstrate cross-host networking, TLS/source
authentication, durable restart recovery, a cross-replica atomic transaction, or production
memory/latency scaling. Container and GitHub Actions configuration is updated; those remote
build/deployment jobs have not been executed as part of this local stage.

## Changed files

- Build/deployment: `.github/workflows/ci.yml`, `CMakeLists.txt`, `Dockerfile`.
- Server: `apps/server.cpp`, `service/vector_search_service.h`, `service/vector_search_service.cpp`,
  `service/snapshot_publication.h`, `service/snapshot_publication.cpp`.
- Core: `include/fast_vector/search_context.h`, `include/fast_vector/search_executor.h`,
  `include/fast_vector/vector_store.h`, `src/search_executor.cpp`, `src/vector_store.cpp`.
- Contract: `proto/fast_vector/v1/vector_search.proto`.
- Python: `scripts/create_snapshot_manifest.py`, `scripts/publish_snapshots.py`, `scripts/grpc_client.py`.
- Tests: `tests/test_search_executor.cpp`, `tests/test_snapshot_publication.cpp`,
  `tests/snapshot_publication_test.py`.
- Documentation: `README.md`, `docs/architecture.md`, `docs/logging.md`,
  `docs/service_concurrency.md`, `docs/snapshot_publication.md`,
  `docs/snapshot_publication_measurements.md`.
