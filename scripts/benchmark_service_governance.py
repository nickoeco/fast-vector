#!/usr/bin/env python3
"""Run a closed-loop Search admission matrix; preserve complete per-run JSON reports."""
from __future__ import annotations

import argparse
import json
import socket
import subprocess
import sys
from pathlib import Path


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--server", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, default=Path("build/service-governance"))
    parser.add_argument("--vectors", type=int, default=20_000)
    parser.add_argument("--dimension", type=int, default=64)
    parser.add_argument("--queries", type=int, default=300)
    options = parser.parse_args()
    options.output_dir.mkdir(parents=True, exist_ok=True)
    root = Path(__file__).resolve().parents[1]
    generated = options.output_dir / "generated"
    subprocess.run([sys.executable, str(root / "scripts/generate_grpc_python.py"),
                    "--output-dir", str(generated)], check=True, stdout=subprocess.DEVNULL)
    print("compute_threads,queue_capacity,clients,qps,p50_ms,p95_ms,p99_ms,rejection_rate,queue_avg_us,execution_avg_us,rss_kib")
    for threads, capacity, clients in ((1, 1, 1), (1, 1, 16), (1, 32, 16),
                                       (2, 1, 16), (2, 32, 16), (4, 32, 32)):
        with socket.socket() as listener:
            listener.bind(("127.0.0.1", 0))
            port = listener.getsockname()[1]
        command = [sys.executable, str(root / "scripts/benchmark_grpc.py"),
                   "--server-executable", str(options.server), "--generated-dir", str(generated),
                   "--address", f"127.0.0.1:{port}", "--index", "flat",
                   "--vectors", str(options.vectors), "--dimension", str(options.dimension),
                   "--queries", str(options.queries), "--workers", str(clients),
                   "--search-workers", str(threads), "--search-queue-capacity", str(capacity),
                   "--search-check-interval", "64", "--seed", "20250908"]
        result = subprocess.run(command, capture_output=True, text=True, timeout=120)
        if result.returncode not in (0, 1):
            raise RuntimeError(result.stderr)
        try:
            report = json.loads(result.stdout)
        except json.JSONDecodeError as error:
            raise RuntimeError(result.stderr) from error
        report["reproduction_command"] = command
        name = f"threads-{threads}-queue-{capacity}-clients-{clients}.json"
        (options.output_dir / name).write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
        print(",".join(str(report.get(key, "NA")) for key in (
            "search_workers", "search_queue_capacity", "workers", "qps", "p50_latency_ms",
            "p95_latency_ms", "p99_latency_ms", "rejection_rate", "queue_average_us",
            "execution_average_us", "server_rss_kib_after_queries")), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
