"""Three real replica processes: rolling activation, failures, fencing and rollback."""
from __future__ import annotations

import concurrent.futures
from functools import partial
from http.server import SimpleHTTPRequestHandler, ThreadingHTTPServer
import json
import os
from pathlib import Path
import socket
import subprocess
import sys
import tempfile
import threading
import time
import unittest

import grpc
from grpc_health.v1 import health_pb2, health_pb2_grpc

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
from create_snapshot_manifest import create_manifest
from generate_grpc_python import generate
from grpc_client import VectorSearchClient
from publish_snapshots import publish, read_manifest


def port() -> int:
    with socket.socket() as connection:
        connection.bind(("127.0.0.1", 0))
        return connection.getsockname()[1]


class QuietHandler(SimpleHTTPRequestHandler):
    def log_message(self, *_):
        pass


class SnapshotPublicationIntegrationTest(unittest.TestCase):
    def test_three_replicas_roll_fail_recover_and_rollback(self):
        server = Path(os.environ.get("FAST_VECTOR_SERVER", ROOT / "build-grpc/fast_vector_server"))
        builder = Path(os.environ.get("FAST_VECTOR_BUILDER", ROOT / "build-grpc/fast_vector_build_index"))
        fixture = Path(os.environ.get("FAST_VECTOR_HNSW_FIXTURE", ROOT / "build-grpc/fast_vector_hnsw_snapshot_fixture"))
        if not all(path.is_file() for path in (server, builder, fixture)):
            self.skipTest("publication integration executables missing")
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            generated = root / "generated"
            generate(ROOT / "proto", generated)
            for index_type in ("flat", "hnsw"):
                with self.subTest(index_type=index_type):
                    self.exercise_type(root, generated, server, builder, fixture, index_type)

    def exercise_type(self, root, generated, server, builder, fixture, index_type):
        directory = root / index_type
        directory.mkdir()
        a, b = directory / "a.bin", directory / "b.bin"
        if index_type == "flat":
            for path, records in ((a, [{"id": 101, "values": [1, 0]}, {"id": 102, "values": [0, 1]}]),
                                  (b, [{"id": 999, "values": [0, 1]}])):
                data = path.with_suffix(".jsonl")
                data.write_text("".join(json.dumps(record) + "\n" for record in records))
                subprocess.run([str(builder), "--input", str(data), "--output", str(path), "--dimension", "2"],
                               capture_output=True, text=True, check=True, timeout=10)
            dimension, count_a, count_b, query, old_id = 2, 2, 1, [0, 1], 102
            wrong = directory / "wrong.bin"
            subprocess.run([str(fixture), str(wrong)], capture_output=True, check=True, timeout=10)
            wrong_manifest = create_manifest(wrong, "hnsw", 3, 3)
        else:
            subprocess.run([str(fixture), str(a)], capture_output=True, check=True, timeout=10)
            subprocess.run([str(fixture), str(b), "--replacement"], capture_output=True, check=True, timeout=10)
            dimension, count_a, count_b, query, old_id = 3, 3, 2, [0, 0, 1], 101
            wrong = directory / "wrong.bin"
            data = directory / "wrong.jsonl"
            data.write_text(json.dumps({"id": 1, "values": [1, 0]}) + "\n")
            subprocess.run([str(builder), "--input", str(data), "--output", str(wrong), "--dimension", "2"],
                           capture_output=True, check=True, timeout=10)
            wrong_manifest = create_manifest(wrong, "flat", 2, 1)
        manifests = {"a": create_manifest(a, index_type, dimension, count_a, "embed-r1"),
                     "b": create_manifest(b, index_type, dimension, count_b, "embed-r1")}
        for name, manifest in manifests.items():
            (directory / f"{name}.manifest.json").write_text(json.dumps(manifest))
        corrupt = bytearray(b.read_bytes())
        corrupt[-1] ^= 1
        (directory / "corrupt.bin").write_bytes(corrupt)
        (directory / "truncated.bin").write_bytes(b.read_bytes()[:-1])
        http = ThreadingHTTPServer(("127.0.0.1", 0), partial(QuietHandler, directory=str(directory)))
        http_thread = threading.Thread(target=http.serve_forever)
        http_thread.start()
        base = f"http://127.0.0.1:{http.server_port}"
        processes, clients, logs, replicas = [], [], [], []
        stop = threading.Event()
        readers = []
        errors = []
        observations = [0, 0, 0]
        allowed = {(manifests["a"]["snapshot_id"], generation): (old_id, "embed-r1")
                   for generation in (1, 4, 6, 8)}
        allowed.update({(manifests["b"]["snapshot_id"], generation): (999, "embed-r1")
                        for generation in (2, 5, 7)})
        reports = []
        try:
            for number in range(3):
                staging = directory / f"staging-{number}"
                staging.mkdir()
                address = f"127.0.0.1:{port()}"
                output = (directory / f"replica-{number}.jsonl").open("w")
                logs.append(output)
                process = subprocess.Popen([str(server), "--address", address, "--index", index_type,
                    "--load-index", str(a), "--snapshot-manifest", str(directory / "a.manifest.json"),
                    "--generation", "1", "--read-only", "true", "--enable-snapshot-publication", "true",
                    "--snapshot-staging-dir", str(staging), "--max-snapshot-bytes", "1048576"],
                    stdout=subprocess.DEVNULL, stderr=output)
                processes.append(process)
                client = VectorSearchClient(address, generated, timeout_seconds=5)
                clients.append(client)
                replicas.append({"name": f"replica-{number}", "address": address})
                initial = client.stats().publication
                self.assertEqual(initial.current.snapshot_id, manifests["a"]["snapshot_id"])
                self.assertEqual(initial.current.generation, 1)
                self.assertFalse(list(staging.iterdir()))

            def observe(number):
                while not stop.is_set():
                    try:
                        result = clients[number].search(query, 1)
                        identity = (result.snapshot_id, result.generation)
                        expected = allowed.get(identity)
                        if expected != (result.neighbors[0].id, result.embedding_revision):
                            errors.append((number, identity, [neighbor.id for neighbor in result.neighbors]))
                        observations[number] += 1
                    except Exception as error:
                        errors.append((number, type(error).__name__))
                    time.sleep(0.001)
            readers = [threading.Thread(target=observe, args=(number,)) for number in range(3)]
            for reader in readers:
                reader.start()

            def release(name, generation, descriptor, source, targets=None):
                report = publish(descriptor, source, targets or replicas, generation, generated, timeout=5)
                report["scenario"] = name
                reports.append(report)
                return report

            descriptor_b = read_manifest(base + "/b.manifest.json", 5)
            rolled = release("rolling", 2, descriptor_b, base + "/b.bin")
            self.assertTrue(rolled["converged"], rolled)
            self.assertGreater(rolled["convergence_us"], 0)
            self.assertTrue(all("prepare_us" in row and "activation_us" in row for row in rolled["replicas"]))

            broken = release("corrupt", 3, descriptor_b, base + "/corrupt.bin")
            self.assertFalse(broken["converged"])
            self.assertTrue(all(row["current"]["generation"] == 2 for row in broken["replicas"]))
            truncated = release("truncated", 3, descriptor_b, base + "/truncated.bin")
            self.assertFalse(truncated["converged"])
            self.assertTrue(all(row["current"]["generation"] == 2 for row in truncated["replicas"]))
            stale_same = release("stale-same-serving-version", 2, descriptor_b, base + "/b.bin")
            self.assertFalse(stale_same["converged"])
            self.assertTrue(all(row["error"] == "STALE_GENERATION" for row in stale_same["replicas"]))
            wrong_report = release("wrong-format", 4, wrong_manifest, base + "/wrong.bin")
            self.assertFalse(wrong_report["converged"])
            self.assertTrue(all(row["current"]["generation"] == 2 for row in wrong_report["replicas"]))

            failure_targets = [dict(replica) for replica in replicas]
            failure_targets[1]["snapshot_url"] = base + "/does-not-exist.bin"
            failed = release("partial-download-failure", 4, manifests["a"], base + "/a.bin", failure_targets)
            self.assertFalse(failed["converged"])
            self.assertEqual([row["current"]["generation"] for row in failed["replicas"]], [4, 2, 4])
            self.assertIn("download_error", failed["replicas"][1])
            self.assertGreater(failed["replicas"][1]["prepare_us"], 0)
            recovered = release("retry-same-generation", 4, manifests["a"], base + "/a.bin")
            self.assertTrue(recovered["converged"], recovered)
            self.assertTrue(release("forward", 5, descriptor_b, base + "/b.bin")["converged"])
            rollback = release("rollback", 6, manifests["a"], base + "/a.bin")
            self.assertTrue(rollback["converged"], rollback)
            for client in clients:
                with self.assertRaises(grpc.RpcError) as stale:
                    client._stub.ActivateSnapshot(client._messages.ActivateSnapshotRequest(
                        snapshot_id=descriptor_b["snapshot_id"], generation=5), timeout=5)
                self.assertEqual(stale.exception.code(), grpc.StatusCode.FAILED_PRECONDITION)
                with self.assertRaises(grpc.RpcError) as mismatch:
                    client.search(query, 1, expected_snapshot_id=descriptor_b["snapshot_id"])
                self.assertEqual(mismatch.exception.code(), grpc.StatusCode.FAILED_PRECONDITION)
                matched = client.search(query, 1, expected_snapshot_id=manifests["a"]["snapshot_id"])
                self.assertEqual((matched.neighbors[0].id, matched.generation), (old_id, 6))

            # Pause one real stream after its manifest: one candidate slot, old queries still work.
            client = clients[0]
            resume = threading.Event()
            def paused_frames():
                yield client._messages.PrepareSnapshotRequest(
                    manifest=client._messages.SnapshotManifest(**descriptor_b), generation=7)
                resume.wait(5)
                content = b.read_bytes()
                for offset in range(0, len(content), 37):
                    yield client._messages.PrepareSnapshotRequest(data=content[offset:offset + 37])
            with concurrent.futures.ThreadPoolExecutor(max_workers=1) as executor:
                pending = executor.submit(client._stub.PrepareSnapshot, paused_frames(), timeout=10)
                try:
                    deadline = time.monotonic() + 5
                    while client.stats().publication.state != "PREPARING" and time.monotonic() < deadline:
                        time.sleep(0.01)
                    self.assertEqual(client.stats().publication.state, "PREPARING")
                    with self.assertRaises(grpc.RpcError) as busy:
                        client._stub.PrepareSnapshot(iter([client._messages.PrepareSnapshotRequest(
                            manifest=client._messages.SnapshotManifest(**descriptor_b), generation=8)]), timeout=5)
                    self.assertEqual(busy.exception.code(), grpc.StatusCode.RESOURCE_EXHAUSTED)
                    self.assertEqual(client.search(query, 1).generation, 6)
                finally:
                    resume.set()
                ready = pending.result(timeout=10)
            self.assertEqual(ready.state, "READY_TO_ACTIVATE")
            self.assertEqual(client.search(query, 1).generation, 6)
            with self.assertRaises(grpc.RpcError) as wrong_candidate:
                client._stub.ActivateSnapshot(client._messages.ActivateSnapshotRequest(
                    snapshot_id=manifests["a"]["snapshot_id"], generation=7), timeout=5)
            self.assertEqual(wrong_candidate.exception.code(), grpc.StatusCode.FAILED_PRECONDITION)
            ready_targets = [dict(replica) for replica in replicas]
            ready_targets[0]["snapshot_url"] = base + "/unneeded-download.bin"
            resumed = release("resume-ready", 7, descriptor_b, base + "/b.bin", ready_targets)
            self.assertTrue(resumed["converged"], resumed)
            self.assertEqual(resumed["replicas"][0]["prepare_us"], 0)
            self.assertNotIn("download_error", resumed["replicas"][0])
            self.assertTrue(release("final-rollback", 8, manifests["a"], base + "/a.bin")["converged"])
            before = [client.stats().snapshot_version for client in clients]
            self.assertTrue(release("idempotent-observation", 8, manifests["a"], base + "/a.bin")["converged"])
            self.assertEqual([client.stats().snapshot_version for client in clients], before)
            configuration = directory / "replicas.json"
            configuration.write_text(json.dumps(replicas))
            cli_report = directory / "cli-report.json"
            command = [sys.executable, str(ROOT / "scripts/publish_snapshots.py"),
                       "--manifest", str(directory / "a.manifest.json"), "--snapshot", base + "/a.bin",
                       "--generation", "8", "--replicas", str(configuration),
                       "--generated-dir", str(generated), "--timeout", "5", "--report", str(cli_report)]
            completed = subprocess.run(command, capture_output=True, text=True, timeout=30)
            self.assertEqual(completed.returncode, 0, completed.stderr + completed.stdout)
            self.assertTrue(json.loads(cli_report.read_text())["converged"])
            command[command.index("--manifest") + 1] = base + "/missing-manifest.json"
            failed_cli = subprocess.run(command, capture_output=True, text=True, timeout=30)
            self.assertNotEqual(failed_cli.returncode, 0)
            self.assertFalse(json.loads(failed_cli.stdout)["converged"])
            self.assertEqual([client.stats().snapshot_version for client in clients], before)
            for client in clients:
                health = health_pb2_grpc.HealthStub(client._channel)
                self.assertEqual(health.Check(health_pb2.HealthCheckRequest(), timeout=5).status,
                                 health_pb2.HealthCheckResponse.SERVING)
            self.assertFalse(errors, errors[:10])
            self.assertTrue(all(count > 0 for count in observations), observations)
            for number in range(3):
                self.assertFalse(list((directory / f"staging-{number}").iterdir()))
            destination = os.environ.get("FAST_VECTOR_PUBLICATION_REPORT_DIR")
            if destination:
                output = Path(destination)
                output.mkdir(parents=True, exist_ok=True)
                (output / f"{index_type}-publication.json").write_text(json.dumps(
                    {"index_type": index_type, "observed_queries": observations, "reports": reports}, indent=2))
        finally:
            stop.set()
            for reader in readers:
                reader.join(timeout=10)
            for client in clients:
                client.close()
            for process in processes:
                if process.poll() is None:
                    process.terminate()
                try:
                    process.wait(timeout=10)
                except subprocess.TimeoutExpired:
                    process.kill()
                    process.wait(timeout=5)
            for output in logs:
                output.close()
            http.shutdown()
            http.server_close()
            http_thread.join(timeout=5)
        self.assertTrue(all(process.returncode == 0 for process in processes))
        self.assertTrue(all(not reader.is_alive() for reader in readers))
        self.assertFalse(http_thread.is_alive())


if __name__ == "__main__":
    unittest.main()
