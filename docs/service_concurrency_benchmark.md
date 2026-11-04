# Phase 11A measured development results

Actual local Release runs, not production capacity estimates. Source changes were uncommitted
at measurement time. Complete generated JSON and CSV reports are in ignored build directories.

Environment: Ubuntu/WSL2, Linux 6.18.54.1-microsoft-standard-WSL2, glibc 2.35, x86-64,
Intel Core i9-13980HX (32 visible logical CPUs), Clang 14.0.0, Release, Python 3.10.12.
No CPU pinning, host isolation or repeated-run confidence interval. Compilation and tests had
finished before this final pass. Each case owns a fresh server using insecure localhost gRPC.
JSON logs go to `/dev/null`; serialization/flush overhead is still present.

## Closed-loop service matrix

20,000 vectors, dimension 64, seed 20250908, exact Flat scalar search, Top-10, 300 timed
queries after 20 warmups, cancellation interval 64. Population is separately timed.

```bash
build/python-grpc-venv/bin/python scripts/benchmark_service_governance.py \
  --server build-clang-grpc/fast_vector_server --output-dir build/service-governance-11a \
  > build/service-governance-11a.csv
```

| Compute threads | Queue capacity | Clients | Success QPS | P50 ms | P95 ms | P99 ms | Refused % | Queue mean us | Execution mean us | RSS KiB |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 1 | 1 | 1 | 837.8 | 1.134 | 1.293 | 1.428 | 0 | 15.3 | 629.1 | 19284 |
| 1 | 1 | 16 | 1210.7 | 3.923 | 5.702 | 8.740 | 72.67 | 549.3 | 698.2 | 20280 |
| 1 | 32 | 16 | 1448.1 | 10.564 | 11.140 | 11.340 | 0 | 9080.2 | 651.9 | 20144 |
| 2 | 1 | 16 | 2288.1 | 3.371 | 6.046 | 7.976 | 46.33 | 194.5 | 665.1 | 19924 |
| 2 | 32 | 16 | 2772.0 | 5.232 | 5.531 | 5.752 | 0 | 3797.9 | 644.5 | 19784 |
| 4 | 32 | 32 | 3940.9 | 6.281 | 11.344 | 15.011 | 0 | 140.8 | 669.0 | 20248 |

Latency and queue/execution means describe successful responses. Refusals remain in rejection
denominators and overall wall time, not success percentiles. JSON also records all-request
P99, attempt QPS, errors, summed scheduler times, high-water queue/running counts, unfinished
tasks and process high-water RSS. RSS above is sampled after queries and includes the entire
server, not isolated index or queue memory.

With one worker and 16 clients, increasing capacity from 1 to 32 removes observed refusals but
grows mean queue time from 0.55 ms to 9.08 ms. Queue capacity admits more waiting work; it does
not add compute workers. Two workers with capacity 32 improve successful throughput and queue
time on this workload, without proving arbitrary linear scaling. Four workers/32 clients have
small application queue time yet higher client P99: client/transport costs are outside that
timer, and profiling is needed to attribute the remainder. Compute bounds do not bound every
communication resource. This short closed-loop run does not establish sustained open-loop
capacity, fairness, TLS/network cost, production settings or high-confidence tail distributions.

## Check-frequency experiment

```bash
./build-clang-grpc/fast_vector_cancellation_benchmark > build-clang-grpc/cancellation-11a.csv
```

Seed 42, dimension 64, Top-10, 20 timed queries after three warmups. Flat: 30,000 vectors.
HNSW: 3,000 vectors, M=8, efConstruction=32, deliberately broad efSearch=3,000. This is not
a recall/latency comparison between index types. Interval 0 disables the context; other
intervals enable cancellation and a far-future monotonic deadline.

| Index | Interval | Mean query us | Cancelled trials | Mean cancel-to-completion us |
| --- | --- | --- | --- | --- |
| Flat | 0 | 932.5 | Not attempted | Not attempted |
| Flat | 1 | 1449.4 | 10/10 | 79.6 |
| Flat | 16 | 984.2 | 10/10 | 48.0 |
| Flat | 64 | 941.7 | 10/10 | 38.4 |
| Flat | 256 | 943.9 | 10/10 | 30.8 |
| HNSW | 0 | 551.4 | Not attempted | Not attempted |
| HNSW | 1 | 1231.7 | 10/10 | 46.3 |
| HNSW | 16 | 611.6 | 10/10 | 24.4 |
| HNSW | 64 | 557.2 | 10/10 | 33.9 |
| HNSW | 256 | 532.5 | 10/10 | 36.0 |

Each cancellation trial signals worker entry, sleeps 100 us, requests cancellation, then awaits
the future. The entry signal precedes the index call and does not prove progress into an inner
loop. Completion includes exception handling, wakeup and scheduling. Checking every item costs
more here than batching. Near-baseline differences and nonmonotonic cancellation means are not
reliable speedup claims. Ten trials cannot establish a microsecond cancellation guarantee.

## Changed files

- Build/entry: `CMakeLists.txt`, `apps/server.cpp`.
- Headers in `include/fast_vector/`: `search_context.h`, `search_executor.h`, `vector_index.h`,
  `vector_store.h`, `flat_index.h`, `hnsw_index.h`.
- Sources in `src/`: `search_executor.cpp`, `vector_store.cpp`, `flat_index.cpp`, `hnsw_index.cpp`.
- Transport: `proto/fast_vector/v1/vector_search.proto`, `service/vector_search_service.h`,
  `service/vector_search_service.cpp`.
- Benchmarks: `benchmarks/benchmark_search_cancellation.cpp`, `scripts/benchmark_grpc.py`,
  `scripts/benchmark_service_governance.py`.
- Tests in `tests/`: `test_search_executor.cpp`, `test_grpc_service.cpp`,
  `test_hnsw_allocation_failure.cpp`, `grpc_python_test.py`.
- Docs: `README.md`, `docs/architecture.md`, `docs/logging.md`, `docs/service_concurrency.md`,
  `docs/service_concurrency_benchmark.md`.
