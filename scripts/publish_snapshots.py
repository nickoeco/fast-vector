#!/usr/bin/env python3
"""Roll a read-only snapshot over an explicit replica list and report observed convergence."""
from __future__ import annotations

import argparse
from contextlib import contextmanager
from datetime import datetime, timezone
import json
import math
from pathlib import Path
from queue import Empty, SimpleQueue
import re
import time
from urllib.parse import urlparse
from urllib.request import urlopen

import grpc

from grpc_client import load_generated_modules


@contextmanager
def open_source(source: str, timeout: float):
    if urlparse(source).scheme in ("http", "https", "file"):
        with urlopen(source, timeout=timeout) as response:
            yield response
    else:
        with Path(source).open("rb") as response:
            yield response


def read_manifest(source: str, timeout: float) -> dict:
    with open_source(source, timeout) as response:
        raw = response.read(16385)
    if len(raw) > 16384:
        raise ValueError("manifest too large")
    manifest = json.loads(raw)
    required = {"format_version", "snapshot_id", "index_type", "dimension", "vector_count", "file_size", "sha256"}
    if not isinstance(manifest, dict) or not required <= manifest.keys() or manifest.keys() - required - {"embedding_revision"}:
        raise ValueError("invalid manifest fields")
    for field in ("dimension", "vector_count", "file_size", "format_version"):
        if type(manifest[field]) is not int or not 0 <= manifest[field] < 2**64:
            raise ValueError("invalid manifest integer")
    if (manifest["dimension"] == 0 or manifest["file_size"] < 56 or
        not isinstance(manifest["sha256"], str) or not re.fullmatch("[0-9a-f]{64}", manifest["sha256"]) or
        manifest["snapshot_id"] != "sha256:" + manifest["sha256"] or
        manifest["index_type"] not in ("flat", "hnsw") or
        manifest["format_version"] not in ((1,) if manifest["index_type"] == "flat" else (1, 2, 3)) or
        not isinstance(manifest.get("embedding_revision", ""), str) or
        len(manifest.get("embedding_revision", "").encode()) > 256):
        raise ValueError("invalid snapshot manifest")
    return manifest


def identity_dict(identity) -> dict:
    return {"snapshot_id": identity.snapshot_id, "generation": identity.generation,
            "embedding_revision": identity.embedding_revision}


def observe_status(row: dict, status, target: dict) -> None:
    row.update(current=identity_dict(status.current), candidate=identity_dict(status.candidate),
               state=status.state, last_result=status.last_result, peak_rss_kib=status.peak_rss_kib,
               highest_generation=status.highest_generation, publication_enabled=status.enabled)
    # Failed RPCs carry no response body: recover related server timing from status.
    if (row["phase"] == "prepare" and identity_dict(status.last_attempt) == target and
        status.last_result in ("PREPARE_FAILED", "READY_TO_ACTIVATE")):
        row["prepare_us"] = status.prepare_us
    if (row["phase"] == "activate" and identity_dict(status.current) == target and
        status.last_result in ("ACTIVATED", "ALREADY_ACTIVE")):
        row["activation_us"] = status.activation_us


def publish(manifest: dict, source: str, replicas: list[dict], generation: int,
            generated_directory: Path, timeout: float = 30.0) -> dict:
    if type(generation) is not int or not 0 < generation < 2**64 or not math.isfinite(timeout) or timeout <= 0 or not replicas:
        raise ValueError("positive generation, timeout and nonempty replicas required")
    names, addresses = set(), set()
    for replica in replicas:
        if (not isinstance(replica, dict) or not isinstance(replica.get("name"), str) or
            not replica["name"] or not isinstance(replica.get("address"), str) or not replica["address"] or
            replica["name"] in names or replica["address"] in addresses):
            raise ValueError("replica names and addresses must be nonempty and unique")
        if "snapshot_url" in replica and (not isinstance(replica["snapshot_url"], str) or not replica["snapshot_url"]):
            raise ValueError("snapshot_url must be a nonempty string")
        names.add(replica["name"])
        addresses.add(replica["address"])
    messages, service = load_generated_modules(generated_directory)
    expected = {"snapshot_id": manifest["snapshot_id"], "generation": generation,
                "embedding_revision": manifest.get("embedding_revision", "")}
    report = {"target": expected, "manifest": dict(manifest), "started_at": datetime.now(timezone.utc).isoformat(),
              "converged": False, "convergence_us": None, "replicas": []}
    started = time.monotonic_ns()
    for replica in replicas:
        row = {"name": replica["name"], "address": replica["address"], "result": "FAILED",
               "prepare_us": 0, "activation_us": 0, "phase": "observe"}
        download_errors = SimpleQueue()
        with grpc.insecure_channel(replica["address"]) as channel:
            stub = service.VectorSearchServiceStub(channel)
            try:
                before = stub.GetStats(messages.GetStatsRequest(), timeout=timeout, wait_for_ready=True).publication
                row["before"] = identity_dict(before.current)
                if not before.enabled:
                    row["error"] = "PUBLICATION_DISABLED"
                    raise ValueError("publication disabled")
                if before.highest_generation > generation:
                    row["error"] = "STALE_GENERATION"
                    raise ValueError("stale generation")
                if identity_dict(before.current) != expected:
                    if identity_dict(before.candidate) != expected or before.state != "READY_TO_ACTIVATE":
                        def frames(descriptor=dict(manifest), release_generation=generation,
                                   source_url=replica.get("snapshot_url", source), errors=download_errors):
                            yield messages.PrepareSnapshotRequest(
                                manifest=messages.SnapshotManifest(**descriptor), generation=release_generation)
                            # Each replica gets an independent download/transfer attempt.
                            try:
                                with open_source(source_url, timeout) as response:
                                    while chunk := response.read(256 * 1024):
                                        yield messages.PrepareSnapshotRequest(data=chunk)
                            except Exception as error:
                                errors.put(type(error).__name__)
                                yield messages.PrepareSnapshotRequest(abort=True)
                                return
                        row["phase"] = "prepare"
                        prepare_start = time.monotonic_ns()
                        prepared = stub.PrepareSnapshot(frames(), timeout=timeout)
                        row["prepare_wall_us"] = (time.monotonic_ns() - prepare_start) // 1000
                        row["prepare_us"] = prepared.prepare_us
                        if identity_dict(prepared.candidate) != expected or prepared.state != "READY_TO_ACTIVATE":
                            raise ValueError("candidate changed before activation")
                    row["phase"] = "activate"
                    activation_start = time.monotonic_ns()
                    activated = stub.ActivateSnapshot(messages.ActivateSnapshotRequest(
                        snapshot_id=expected["snapshot_id"], generation=generation), timeout=timeout)
                    row["activation_wall_us"] = (time.monotonic_ns() - activation_start) // 1000
                    row["activation_us"] = activated.activation_us
                row["result"] = "OBSERVED_TARGET"
            except grpc.RpcError as error:
                row["error"] = error.code().name
            except Exception as error:
                row.setdefault("error", type(error).__name__)
            try:
                row["download_error"] = download_errors.get_nowait()
            except Empty:
                pass
            try:
                status = stub.GetStats(messages.GetStatsRequest(), timeout=timeout).publication
                observe_status(row, status, expected)
            except grpc.RpcError as error:
                row["observation_error"] = error.code().name
        report["replicas"].append(row)
    # Re-observe every replica after all rolling activations, rather than trust mutation replies.
    for row in report["replicas"]:
        with grpc.insecure_channel(row["address"]) as channel:
            stub = service.VectorSearchServiceStub(channel)
            try:
                status = stub.GetStats(messages.GetStatsRequest(), timeout=timeout).publication
                observe_status(row, status, expected)
            except grpc.RpcError as error:
                row["observation_error"] = error.code().name
    def matches(row):
        return (row.get("current") == expected and "observation_error" not in row and
                row.get("highest_generation") == generation and row.get("publication_enabled"))
    report["converged"] = all(matches(row) for row in report["replicas"])
    for row in report["replicas"]:
        row["result"] = "CONVERGED" if matches(row) else "NOT_CONVERGED"
    report["elapsed_us"] = (time.monotonic_ns() - started) // 1000
    if report["converged"]:
        report["convergence_us"] = report["elapsed_us"]
    return report


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", required=True, help="local path or HTTP(S)/file URL")
    parser.add_argument("--snapshot", required=True, help="local path or HTTP(S)/file URL")
    parser.add_argument("--generation", type=int, required=True)
    parser.add_argument("--replicas", type=Path, required=True)
    parser.add_argument("--generated-dir", type=Path, default=Path("build/python-grpc"))
    parser.add_argument("--timeout", type=float, default=30.0)
    parser.add_argument("--report", type=Path)
    options = parser.parse_args()
    try:
        manifest = read_manifest(options.manifest, options.timeout)
        report = publish(manifest, options.snapshot, json.loads(options.replicas.read_text()),
                         options.generation, options.generated_dir, options.timeout)
    except Exception as error:
        report = {"converged": False, "convergence_us": None, "replicas": [],
                  "error": "RELEASE_INPUT_FAILED", "error_type": type(error).__name__}
    output = json.dumps(report, indent=2, sort_keys=True) + "\n"
    if options.report:
        options.report.write_text(output)
    print(output, end="")
    return 0 if report["converged"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
