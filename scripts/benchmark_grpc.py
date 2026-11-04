#!/usr/bin/env python3
"""Run a reproducible in-process-client load test against fast-vector gRPC."""

from __future__ import annotations

import argparse
import atexit
import concurrent.futures
import json
import os
import platform
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
    parser.add_argument("--search-workers", type=int, default=2)
    parser.add_argument("--search-queue-capacity", type=int, default=64)
    parser.add_argument("--search-check-interval", type=int, default=64)
    parser.add_argument("--batch-size", type=int, default=500)
    parser.add_argument("--warmup", type=int, default=20)
    parser.add_argument("--timeout", type=float, default=10.0)
    parser.add_argument("--seed", type=int, default=20250908)
    options = parser.parse_args(arguments)
    for name in ("vectors", "dimension", "queries", "k", "workers", "batch_size",
                 "search_workers", "search_queue_capacity", "search_check_interval"):
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
                "--search-workers", str(options.search_workers),
                "--search-queue-capacity", str(options.search_queue_capacity),
                "--search-check-interval", str(options.search_check_interval),
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
        before = client.stats().search_executor

        def execute(query: Sequence[float]):
            start = time.perf_counter()
            try:
                response = client.search(query, options.k)
                return (time.perf_counter() - start) * 1_000.0, None, response.queue_us, response.execution_us
            except grpc.RpcError as error:
                return (time.perf_counter() - start) * 1_000.0, error.code().name, None, None

        wall_start = time.perf_counter()
        with concurrent.futures.ThreadPoolExecutor(max_workers=options.workers) as executor:
            measurements = list(executor.map(execute, queries))
        wall_seconds = time.perf_counter() - wall_start
        after = client.stats().search_executor

    latencies = sorted(item[0] for item in measurements if item[1] is None)
    errors: dict[str, int] = {}
    for _, error, _, _ in measurements:
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
        "search_workers": after.workers,
        "search_queue_capacity": after.queue_capacity,
        "search_check_interval": options.search_check_interval if server is not None else None,
        "batch_size": options.batch_size,
        "warmup_queries": min(options.warmup, len(queries)),
        "index_build_seconds": build_seconds,
        "query_wall_seconds": wall_seconds,
        "successful_queries": successful,
        "failed_queries": len(measurements) - successful,
        "errors_by_status": errors,
        "qps": successful / wall_seconds if wall_seconds > 0 else 0.0,
        "attempt_qps": len(measurements) / wall_seconds if wall_seconds > 0 else 0.0,
        "rejection_rate": errors.get("RESOURCE_EXHAUSTED", 0) / len(measurements),
        "peak_queued": after.peak_queued,
        "peak_running": after.peak_running,
        "scheduler_completed": after.completed - before.completed,
        "scheduler_rejected": after.rejected - before.rejected,
        "scheduler_cancelled": after.cancelled - before.cancelled,
        "scheduler_expired": after.expired - before.expired,
        "scheduler_queued_after_queries": after.queued,
        "scheduler_running_after_queries": after.running,
        "queue_us_total": after.queue_us_total - before.queue_us_total,
        "execution_us_total": after.execution_us_total - before.execution_us_total,
        "platform": platform.platform(),
        "cpu_count": os.cpu_count(),
        "python_version": platform.python_version(),
        "logging_sink": "devnull" if server is not None else "external",
    }
    for name, position in (("queue", 2), ("execution", 3)):
        values = sorted(item[position] for item in measurements if item[position] is not None)
        if values:
            report[f"{name}_average_us"] = statistics.fmean(values)
            for label, fraction in (("p50", .50), ("p95", .95), ("p99", .99)):
                report[f"{name}_{label}_us"] = percentile(values, fraction)
    report["all_request_p99_ms"] = percentile(sorted(item[0] for item in measurements), .99)
    if server is not None:
        status_path = Path(f"/proc/{server.pid}/status")
        if status_path.is_file():
            memory = dict(line.split(":", 1) for line in status_path.read_text().splitlines() if ":" in line)
            report["server_rss_kib_after_queries"] = int(memory["VmRSS"].split()[0])
            report["server_peak_rss_kib"] = int(memory["VmHWM"].split()[0])
        cache = options.server_executable.parent / "CMakeCache.txt"
        if cache.is_file():
            for line in cache.read_text().splitlines():
                for key in ("CMAKE_BUILD_TYPE", "CMAKE_CXX_COMPILER"):
                    if line.startswith(key + ":"):
                        report[key.lower()] = line.split("=", 1)[1]
            compiler = report.get("cmake_cxx_compiler")
            if compiler:
                report["compiler_version"] = subprocess.check_output(
                    [compiler, "--version"], text=True
                ).splitlines()[0]
    cpuinfo = Path("/proc/cpuinfo")
    if cpuinfo.is_file():
        for line in cpuinfo.read_text().splitlines():
            if line.startswith("model name"):
                report["cpu_model"] = line.split(":", 1)[1].strip()
                break
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
