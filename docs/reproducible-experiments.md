# Reproducible experiments

## Record the environment

Performance results are meaningful only with their environment. Record at least:

```bash
git rev-parse HEAD
cmake --version
c++ --version
uname -a
lscpu
```

Also record the build type, compiler, vector count, dimension, query count, `K`, random seeds,
distance kernel, thread count, and HNSW parameters. Run performance measurements on an idle
machine and do not compare Debug results with Release results.

## Build and test

```bash
cmake -S . -B build-release \
  -DCMAKE_BUILD_TYPE=Release \
  -DFAST_VECTOR_BUILD_GRPC=ON
cmake --build build-release --parallel
ctest --test-dir build-release --output-on-failure
```

For memory and undefined-behavior checks, use a separate Debug directory as documented in the
README. Sanitizer timings must not be reported as Release performance.

## Exact correctness

```bash
python3 -m venv .venv
. .venv/bin/activate
python -m pip install -r scripts/requirements.txt
python scripts/validate_recall.py \
  --runner build-release/fast_vector_validate \
  --output-dir validation-output/flat \
  --kernel scalar
```

The exact Flat result must match the NumPy ranked ID sequence, achieve Recall@K `1.0`, and stay
within the configured score tolerance. Keep `summary.json` with the commit and environment
record when publishing an experiment; generated validation data is intentionally not committed.

## In-process performance

```bash
./build-release/fast_vector_benchmark \
  --vectors 10000 --dimension 128 --queries 1000 --k 10 \
  --warmup 20 --threads 4 --kernel auto --seed 20250908

./build-release/fast_vector_hnsw_benchmark \
  --vectors 10000 --dimension 128 --queries 1000 --k 10 \
  --m 16 --ef-construction 200 --ef-search 10,50,100,200 \
  --neighbor-selection heuristic --seed 20250908 --hnsw-seed 42
```

Report construction time separately from query latency. Report recall beside every HNSW
latency/QPS result. RSS includes the process and allocator, whereas payload counters cover only
selected index data; neither should be described as the exact total index memory.

## Service performance

Start a fresh server for each independent run, ingest exactly once, and then run:

```bash
python scripts/benchmark_grpc.py \
  --address 127.0.0.1:50051 \
  --vectors 10000 --dimension 128 --queries 1000 --k 10 \
  --workers 1,2,4,8 --seed 20250908
```

Service QPS includes protobuf serialization, networking, gRPC scheduling, locking, and search.
Do not compare it directly with the in-process algorithm benchmark. Preserve failures grouped
by gRPC status; a run with request failures is not a valid throughput result.

## Reporting checklist

- Commit SHA and dirty-worktree status.
- OS, CPU, memory, compiler, and CMake versions.
- Exact command and all random seeds.
- Release build confirmation and enabled distance kernel.
- Dataset size, dimension, query count, `K`, and warm-up count.
- Mean plus P50/P95/P99 latency and QPS.
- HNSW Recall@K and all graph parameters.
- Construction, save, and load times kept separate from query time.
- Any failures, skipped tests, thermal throttling, or environment limitations.

The repository intentionally contains no pre-filled performance claims. Add numbers to a
release report only after running these commands on a named machine.
