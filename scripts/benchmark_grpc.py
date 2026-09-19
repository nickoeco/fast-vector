#!/usr/bin/env python3
"""Run a reproducible in-process-client load test against fast-vector gRPC."""

from __future__ import annotations

import argparse
import atexit
import concurrent.futures
import json
import random
import statistics
import subprocess
import time
from pathlib import Path
from typing import Sequence

import grpc

from grpc_client import VectorSearchClient


def percentile(sorted_values: Sequence[float], percentile_value: float) -> float:
    """Return a linearly interpolated percentile from sorted values."""
    if not sorted_values:
        raise ValueError("cannot compute a percentile of an empty sequence")
    position = (len(sorted_values) - 1) * percentile_value
    lower = int(position)
    upper = min(lower + 1, len(sorted_values) - 1)
    fraction = position - lower
    return sorted_values[lower] + (sorted_values[upper] - sorted_values[lower]) * fraction


def random_vector(generator: random.Random, dimension: int) -> list[float]:
    return [generator.uniform(-1.0, 1.0) for _ in range(dimension)]


def parse_arguments(arguments: Sequence[str] | None = None) -> argparse.Namespace:
    repository_root = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--address", default="127.0.0.1:50051")
    parser.add_argument(
        "--server-executable",
        type=Path,
        help="optionally start and stop this fast_vector_server executable",
    )
    parser.add_argument("--index", choices=("flat", "hnsw"), default="flat")
    parser.add_argument(
        "--generated-dir", type=Path, default=repository_root / "build" / "python-grpc"
    )
    parser.add_argument("--vectors", type=int, default=10_000)
    parser.add_argument("--dimension", type=int, default=128)
    parser.add_argument("--queries", type=int, default=1_000)
    parser.add_argument("--k", type=int, default=10)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--batch-size", type=int, default=500)
    parser.add_argument("--warmup", type=int, default=20)
    parser.add_argument("--timeout", type=float, default=10.0)
    parser.add_argument("--seed", type=int, default=20250908)
    options = parser.parse_args(arguments)
    for name in ("vectors", "dimension", "queries", "k", "workers", "batch_size"):
        if getattr(options, name) <= 0:
            parser.error(f"--{name.replace('_', '-')} must be positive")
    if options.warmup < 0:
        parser.error("--warmup must not be negative")
    return options


def stop_server(server: subprocess.Popen[bytes]) -> None:
    if server.poll() is not None:
        return
    server.terminate()
    try:
        server.wait(timeout=5)
    except subprocess.TimeoutExpired:
        server.kill()
        server.wait(timeout=5)


def main(arguments: Sequence[str] | None = None) -> int:
    options = parse_arguments(arguments)
    server = None
    if options.server_executable is not None:
        server = subprocess.Popen(
            [
                str(options.server_executable),
                "--address",
                options.address,
                "--index",
                options.index,
                "--dimension",
                str(options.dimension),
                "--max-batch-size",
                str(options.batch_size),
            ],
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        )
        atexit.register(stop_server, server)
    generator = random.Random(options.seed)
    database = [random_vector(generator, options.dimension) for _ in range(options.vectors)]
    queries = [random_vector(generator, options.dimension) for _ in range(options.queries)]

    with VectorSearchClient(
        options.address, options.generated_dir, options.timeout
    ) as client:
        build_start = time.perf_counter()
        for offset in range(0, options.vectors, options.batch_size):
            batch = [
                (index + 1, database[index])
                for index in range(offset, min(offset + options.batch_size, options.vectors))
            ]
            client.add_batch(batch)
        build_seconds = time.perf_counter() - build_start

        for query in queries[: min(options.warmup, len(queries))]:
            client.search(query, options.k)

        def execute(query: Sequence[float]) -> tuple[float, str | None]:
            start = time.perf_counter()
            try:
                client.search(query, options.k)
                return (time.perf_counter() - start) * 1_000.0, None
            except grpc.RpcError as error:
                return (time.perf_counter() - start) * 1_000.0, error.code().name

        wall_start = time.perf_counter()
        with concurrent.futures.ThreadPoolExecutor(max_workers=options.workers) as executor:
            measurements = list(executor.map(execute, queries))
        wall_seconds = time.perf_counter() - wall_start

    latencies = sorted(latency for latency, error in measurements if error is None)
    errors: dict[str, int] = {}
    for _, error in measurements:
        if error is not None:
            errors[error] = errors.get(error, 0) + 1
    successful = len(latencies)
    report = {
        "address": options.address,
        "seed": options.seed,
        "vectors": options.vectors,
        "dimension": options.dimension,
        "queries": options.queries,
        "top_k": options.k,
        "workers": options.workers,
        "batch_size": options.batch_size,
        "warmup_queries": min(options.warmup, len(queries)),
        "index_build_seconds": build_seconds,
        "query_wall_seconds": wall_seconds,
        "successful_queries": successful,
        "failed_queries": len(measurements) - successful,
        "errors_by_status": errors,
        "qps": successful / wall_seconds if wall_seconds > 0 else 0.0,
    }
    if latencies:
        report.update(
            {
                "average_latency_ms": statistics.fmean(latencies),
                "p50_latency_ms": percentile(latencies, 0.50),
                "p95_latency_ms": percentile(latencies, 0.95),
                "p99_latency_ms": percentile(latencies, 0.99),
            }
        )
    print(json.dumps(report, indent=2, sort_keys=True))
    if server is not None:
        stop_server(server)
        atexit.unregister(stop_server)
    return 0 if not errors else 1


if __name__ == "__main__":
    raise SystemExit(main())
