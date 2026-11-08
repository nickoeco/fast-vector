"""Real-process SIGKILL recovery tests; these are not power-loss tests."""
from __future__ import annotations

import concurrent.futures
import json
import os
from pathlib import Path
import signal
import struct
import subprocess
import sys
import tempfile
import time
import unittest

import grpc
from grpc_health.v1 import health_pb2, health_pb2_grpc

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
from generate_grpc_python import generate
from grpc_client import VectorSearchClient
from grpc_client import response_to_dictionary


class WalRecoveryIntegrationTest(unittest.TestCase):
    def setUp(self):
        self.server = Path(os.environ.get("FAST_VECTOR_SERVER", ROOT / "build-grpc/fast_vector_server"))
        if not self.server.is_file():
            self.skipTest("gRPC server executable missing")
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name)
        self.generated = self.root / "generated"
        generate(ROOT / "proto", self.generated)
        self.processes = []
        self.addCleanup(self.stop_processes)

    def stop_processes(self):
        for process in self.processes:
            if process.poll() is None:
                process.terminate()
                try:
                    process.wait(timeout=10)
                except subprocess.TimeoutExpired:
                    process.kill()
                    process.wait(timeout=5)

    def start(self, directory):
        log = self.root / f"server-{len(self.processes)}.jsonl"
        with log.open("w") as output:
            process = subprocess.Popen([
                str(self.server), "--address", "127.0.0.1:0", "--index", "flat",
                "--dimension", "3", "--wal-dir", str(directory),
            ], stdout=subprocess.DEVNULL, stderr=output)
        self.processes.append(process)
        limit = time.monotonic() + 10
        while time.monotonic() < limit:
            if process.poll() is not None:
                self.fail(f"server exited during recovery: {log.read_text()}")
            for line in log.read_text().splitlines(keepends=True):
                if not line.endswith("\n"):
                    continue
                event = json.loads(line)
                if event.get("event") == "server_started":
                    client = VectorSearchClient(f"127.0.0.1:{event['selected_port']}", self.generated)
                    self.addCleanup(client.close)
                    health = health_pb2_grpc.HealthStub(client._channel)
                    self.assertEqual(health.Check(health_pb2.HealthCheckRequest(), timeout=5).status,
                                     health_pb2.HealthCheckResponse.SERVING)
                    return process, client
            time.sleep(.01)
        self.fail("server readiness timed out")

    def test_all_acknowledged_writes_survive_sigkill_with_and_without_checkpoint(self):
        for checkpoint in (False, True):
            with self.subTest(checkpoint=checkpoint):
                directory = self.root / str(checkpoint)
                directory.mkdir()
                process, client = self.start(directory)
                values = [(float(i + 1), 2.3, -3.7) for i in range(100)]
                for i, vector in enumerate(values):
                    result = client.add(1000 + i, vector, f"operation-{i}")
                    self.assertEqual(result.lsn, i + 1)
                    self.assertFalse(result.replayed)
                    if checkpoint and i == 49:
                        self.assertEqual(client.checkpoint().lsn, 50)
                before = client.search([1, 2, 3], 100)
                process.kill()
                self.assertEqual(process.wait(timeout=5), -signal.SIGKILL)
                client.close()
                process, client = self.start(directory)
                stats = client.stats()
                self.assertEqual(response_to_dictionary(stats)["durability"]["applied_lsn"], 100)
                self.assertTrue(stats.durability.enabled)
                self.assertFalse(stats.durability.recovery_required)
                self.assertEqual(stats.vector_count, 100)
                self.assertEqual(stats.inserted_vectors, 0)
                self.assertEqual(stats.durability.applied_lsn, 100)
                self.assertEqual(stats.durability.checkpoint_lsn, 50 if checkpoint else 0)
                self.assertEqual(stats.durability.recovered_records, 50 if checkpoint else 100)
                after = client.search([1, 2, 3], 100)
                def exact(response):
                    return [(n.id, struct.pack("<f", n.score)) for n in response.neighbors]
                self.assertEqual(exact(before), exact(after))
                self.assertEqual({n.id for n in after.neighbors}, set(range(1000, 1100)))
                for i, vector in enumerate(values):
                    retry = client.add(1000 + i, vector, f"operation-{i}")
                    self.assertTrue(retry.replayed)
                    self.assertEqual(retry.lsn, i + 1)
                    self.assertEqual(retry.index_size, i + 1)
                with concurrent.futures.ThreadPoolExecutor(max_workers=8) as pool:
                    responses = list(pool.map(lambda _: client.add(9999, [1, 0, 0], "concurrent"),
                                              range(16)))
                self.assertEqual(sum(not r.replayed for r in responses), 1)
                self.assertEqual({r.lsn for r in responses}, {101})
                self.assertEqual(client.stats().vector_count, 101)
                with self.assertRaises(grpc.RpcError) as error:
                    client.add(9999, [0, 1, 0], "concurrent")
                self.assertEqual(error.exception.code(), grpc.StatusCode.INVALID_ARGUMENT)
                with self.assertRaises(grpc.RpcError) as error:
                    client.add_batch([(12345, [1, 0, 0])])
                self.assertEqual(error.exception.code(), grpc.StatusCode.FAILED_PRECONDITION)
                self.assertEqual(client.stats().vector_count, 101)
                self.assertEqual(client.checkpoint().lsn, 101)
                process.terminate()
                self.assertEqual(process.wait(timeout=10), 0)
                client.close()

    def test_corrupt_complete_record_prevents_server_listening(self):
        directory = self.root / "corrupt"
        directory.mkdir()
        process, client = self.start(directory)
        client.add(1, [1, 2, 3], "first")
        process.kill()
        process.wait(timeout=5)
        client.close()
        wal = directory / "operations.wal"
        bytes_ = bytearray(wal.read_bytes())
        bytes_[-1] ^= 1
        wal.write_bytes(bytes_)
        result = subprocess.run([str(self.server), "--index", "flat", "--dimension", "3",
                                 "--address", "127.0.0.1:0", "--wal-dir", str(directory)],
                                capture_output=True, text=True, timeout=10)
        self.assertNotEqual(result.returncode, 0)
        self.assertNotIn('"event":"server_started"', result.stderr)


if __name__ == "__main__":
    unittest.main()
