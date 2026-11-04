"""End-to-end tests for the Python gRPC client."""

from __future__ import annotations

import os
import json
import concurrent.futures
from collections import Counter
import signal
import socket
import subprocess
import sys
import tempfile
import time
import unittest
from pathlib import Path

import grpc
from grpc_health.v1 import health_pb2, health_pb2_grpc


REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
SCRIPTS_DIRECTORY = REPOSITORY_ROOT / "scripts"
sys.path.insert(0, str(SCRIPTS_DIRECTORY))

from benchmark_grpc import percentile  # noqa: E402
from generate_grpc_python import generate  # noqa: E402
from grpc_client import VectorSearchClient  # noqa: E402


def reserve_local_port() -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as listener:
        listener.bind(("127.0.0.1", 0))
        return listener.getsockname()[1]


def stop_server(server: subprocess.Popen[str]) -> tuple[str, str]:
    if server.poll() is None:
        server.terminate()
    try:
        stdout, stderr = server.communicate(timeout=5)
    except subprocess.TimeoutExpired:
        server.kill()
        stdout, stderr = server.communicate(timeout=5)
    return stdout or "", stderr or ""


def read_log_records(path: Path) -> list[dict]:
    # A live writer may not have completed the final line yet.
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines(keepends=True)
            if line.endswith("\n")]


class StatisticsTest(unittest.TestCase):
    def test_percentile_uses_linear_interpolation(self) -> None:
        values = [1.0, 2.0, 3.0, 4.0]
        self.assertEqual(percentile(values, 0.0), 1.0)
        self.assertEqual(percentile(values, 1.0), 4.0)
        self.assertAlmostEqual(percentile(values, 0.5), 2.5)


class GrpcPythonClientTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        server_path = Path(
            os.environ.get("FAST_VECTOR_SERVER", REPOSITORY_ROOT / "build-grpc" / "fast_vector_server")
        )
        if not server_path.is_file():
            raise unittest.SkipTest(f"gRPC server executable not found: {server_path}")

        cls._temporary_directory = tempfile.TemporaryDirectory()
        generated_directory = Path(cls._temporary_directory.name) / "generated"
        generate(REPOSITORY_ROOT / "proto", generated_directory)
        port = reserve_local_port()
        cls._address = f"127.0.0.1:{port}"
        cls._server = subprocess.Popen(
            [
                str(server_path),
                "--address",
                cls._address,
                "--index",
                "flat",
                "--dimension",
                "3",
                "--max-batch-size",
                "10",
            ],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
        )
        try:
            cls._client = VectorSearchClient(
                cls._address, generated_directory, timeout_seconds=5.0
            )
        except (grpc.RpcError, grpc.FutureTimeoutError):
            stdout, stderr = stop_server(cls._server)
            raise RuntimeError(f"server did not become ready\nstdout: {stdout}\nstderr: {stderr}")

    @classmethod
    def tearDownClass(cls) -> None:
        if hasattr(cls, "_client"):
            cls._client.close()
        if hasattr(cls, "_server"):
            stop_server(cls._server)
        if hasattr(cls, "_temporary_directory"):
            cls._temporary_directory.cleanup()

    def test_add_search_and_stats(self) -> None:
        first = self._client.add(101, [1.0, 0.0, 0.0])
        self.assertEqual(first.index_size, 1)
        batch = self._client.add_batch(
            [(102, [0.0, 1.0, 0.0]), (103, [-1.0, 0.0, 0.0])]
        )
        self.assertEqual(batch.added_count, 2)
        self.assertEqual(batch.index_size, 3)

        result = self._client.search([1.0, 0.0, 0.0], 2)
        self.assertEqual([neighbor.id for neighbor in result.neighbors], [101, 102])
        self.assertAlmostEqual(result.neighbors[0].score, 1.0, places=5)

        stats = self._client.stats()
        self.assertEqual(stats.index_type, "flat")
        self.assertEqual(stats.vector_count, 3)
        self.assertEqual(stats.dimension, 3)
        self.assertEqual(stats.successful_queries, 1)
        self.assertEqual(stats.inserted_vectors, 3)
        self.assertFalse(stats.read_only)

    def test_invalid_vector_surfaces_grpc_status(self) -> None:
        with self.assertRaises(grpc.RpcError) as context:
            self._client.add(200, [1.0, 0.0])
        self.assertEqual(context.exception.code(), grpc.StatusCode.INVALID_ARGUMENT)


class SnapshotServerIntegrationTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        server_path = Path(
            os.environ.get("FAST_VECTOR_SERVER", REPOSITORY_ROOT / "build-grpc" / "fast_vector_server")
        )
        builder_path = Path(
            os.environ.get(
                "FAST_VECTOR_BUILDER",
                REPOSITORY_ROOT / "build-grpc" / "fast_vector_build_index",
            )
        )
        if not server_path.is_file():
            raise unittest.SkipTest(f"gRPC server executable not found: {server_path}")
        if not builder_path.is_file():
            raise unittest.SkipTest(f"index builder executable not found: {builder_path}")

        cls._temporary_directory = tempfile.TemporaryDirectory()
        temporary_path = Path(cls._temporary_directory.name)
        generated_directory = temporary_path / "generated"
        snapshot_path = temporary_path / "index.fv"
        cls._builder_path = builder_path
        cls._snapshot_path = snapshot_path
        cls._temporary_path = temporary_path
        cls._log_path = temporary_path / "server.jsonl"
        cls._log_output = cls._log_path.open("w", encoding="utf-8")
        generate(REPOSITORY_ROOT / "proto", generated_directory)

        build_result = subprocess.run(
            [
                str(builder_path),
                "--input",
                str(REPOSITORY_ROOT / "tests" / "data" / "vectors.jsonl"),
                "--output",
                str(snapshot_path),
                "--dimension",
                "3",
                "--expected-count",
                "3",
            ],
            capture_output=True,
            text=True,
            check=False,
        )
        if build_result.returncode != 0:
            cls._log_output.close()
            cls._temporary_directory.cleanup()
            raise RuntimeError(
                "index builder failed\n"
                f"stdout: {build_result.stdout}\nstderr: {build_result.stderr}"
            )

        port = reserve_local_port()
        cls._address = f"127.0.0.1:{port}"
        cls._server = subprocess.Popen(
            [
                str(server_path),
                "--address",
                cls._address,
                "--index",
                "flat",
                "--load-index",
                str(snapshot_path),
                "--read-only",
                "true",
                "--reload-on-sighup",
                "true",
                "--max-batch-size",
                "10",
                "--kernel",
                "scalar",
            ],
            stdout=subprocess.PIPE,
            stderr=cls._log_output,
            text=True,
        )
        try:
            cls._client = VectorSearchClient(
                cls._address, generated_directory, timeout_seconds=5.0
            )
        except (grpc.RpcError, grpc.FutureTimeoutError):
            stdout, stderr = stop_server(cls._server)
            cls._log_output.close()
            cls._temporary_directory.cleanup()
            raise RuntimeError(f"server did not become ready\nstdout: {stdout}\nstderr: {stderr}")

    @classmethod
    def tearDownClass(cls) -> None:
        if hasattr(cls, "_client"):
            cls._client.close()
        if hasattr(cls, "_server"):
            stop_server(cls._server)
        if hasattr(cls, "_log_output"):
            cls._log_output.close()
        if hasattr(cls, "_temporary_directory"):
            cls._temporary_directory.cleanup()

    def test_snapshot_server_lifecycle(self) -> None:
        with grpc.insecure_channel(self._address) as channel:
            health = health_pb2_grpc.HealthStub(channel)
            response = health.Check(
                health_pb2.HealthCheckRequest(service=""), timeout=5.0
            )
            self.assertEqual(response.status, health_pb2.HealthCheckResponse.SERVING)

        startup_records = read_log_records(self._log_path)
        loaded = next(record for record in startup_records if record["event"] == "index_load_completed")
        self.assertEqual(loaded["vector_count"], 3)
        self.assertEqual(loaded["operation"], "startup")
        self.assertGreaterEqual(loaded["latency_us"], 0)
        configuration = next(record for record in startup_records if record["event"] == "startup_config")
        self.assertEqual(configuration["config_source"], "snapshot")
        self.assertEqual(configuration["dimension"], 3)
        self.assertEqual(self._client.stats().snapshot_version, 1)

        result = self._client.search([1.0, 0.0, 0.0], 3)
        self.assertEqual([neighbor.id for neighbor in result.neighbors], [101, 102, 103])
        self.assertAlmostEqual(result.neighbors[0].score, 1.0, places=5)
        self.assertAlmostEqual(result.neighbors[1].score, 0.0, places=5)
        self.assertAlmostEqual(result.neighbors[2].score, -1.0, places=5)

        stats = self._client.stats()
        self.assertEqual(stats.index_type, "flat")
        self.assertEqual(stats.vector_count, 3)
        self.assertEqual(stats.dimension, 3)
        self.assertTrue(stats.read_only)
        self.assertEqual(stats.inserted_vectors, 0)

        with self.assertRaises(grpc.RpcError) as single_context:
            self._client.add(200, [1.0, 0.0, 0.0])
        self.assertEqual(single_context.exception.code(), grpc.StatusCode.FAILED_PRECONDITION)

        with self.assertRaises(grpc.RpcError) as batch_context:
            self._client.add_batch([(201, [0.0, 1.0, 0.0])])
        self.assertEqual(batch_context.exception.code(), grpc.StatusCode.FAILED_PRECONDITION)

        final_stats = self._client.stats()
        self.assertEqual(final_stats.vector_count, 3)
        self.assertEqual(final_stats.inserted_vectors, 0)

        replacement_input = self._temporary_path / "replacement.jsonl"
        replacement_input.write_text(
            '{"id": 999, "values": [0.0, 0.0, 1.0]}\n', encoding="utf-8"
        )
        replacement = subprocess.run(
            [
                str(self._builder_path),
                "--input",
                str(replacement_input),
                "--output",
                str(self._snapshot_path),
                "--dimension",
                "3",
                "--expected-count",
                "1",
                "--overwrite",
                "true",
            ],
            capture_output=True,
            text=True,
            check=False,
        )
        self.assertEqual(replacement.returncode, 0, replacement.stderr)
        os.kill(self._server.pid, signal.SIGHUP)

        deadline = time.monotonic() + 5.0
        while time.monotonic() < deadline:
            reloaded = self._client.search([0.0, 0.0, 1.0], 1)
            if reloaded.neighbors and reloaded.neighbors[0].id == 999:
                break
            time.sleep(0.05)
        else:
            self.fail("server did not expose the replacement snapshot before the timeout")

        reloaded_stats = self._client.stats()
        self.assertEqual(reloaded_stats.vector_count, 1)
        self.assertEqual(reloaded_stats.dimension, 3)
        self.assertTrue(reloaded_stats.read_only)
        self.assertEqual(reloaded_stats.inserted_vectors, 0)
        self.assertEqual(reloaded_stats.snapshot_version, 2)
        self.assertEqual(reloaded.snapshot_version, 2)

        self._snapshot_path.write_bytes(b"corrupt snapshot")
        os.kill(self._server.pid, signal.SIGHUP)
        deadline = time.monotonic() + 5.0
        while time.monotonic() < deadline:
            if any(record["event"] == "index_load_failed" and record.get("operation") == "reload"
                   for record in read_log_records(self._log_path)):
                break
            time.sleep(0.05)
        else:
            self.fail("server did not log a structured reload failure")

        preserved = self._client.search([0.0, 0.0, 1.0], 1)
        self.assertEqual([neighbor.id for neighbor in preserved.neighbors], [999])
        self.assertEqual(self._client.stats().vector_count, 1)


class HnswSnapshotServerIntegrationTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls._server_path = Path(
            os.environ.get("FAST_VECTOR_SERVER", REPOSITORY_ROOT / "build-grpc" / "fast_vector_server")
        )
        fixture_path = Path(
            os.environ.get(
                "FAST_VECTOR_HNSW_FIXTURE",
                REPOSITORY_ROOT / "build-grpc" / "fast_vector_hnsw_snapshot_fixture",
            )
        )
        if not cls._server_path.is_file() or not fixture_path.is_file():
            raise unittest.SkipTest("gRPC server or HNSW snapshot fixture executable not found")

        cls._temporary_directory = tempfile.TemporaryDirectory()
        temporary_path = Path(cls._temporary_directory.name)
        cls._snapshot_path = temporary_path / "index.fvh"
        cls._generated_directory = temporary_path / "generated"
        generate(REPOSITORY_ROOT / "proto", cls._generated_directory)
        fixture = subprocess.run(
            [str(fixture_path), str(cls._snapshot_path)],
            capture_output=True,
            text=True,
            check=False,
        )
        if fixture.returncode != 0:
            cls._temporary_directory.cleanup()
            raise RuntimeError(f"HNSW fixture failed: {fixture.stderr}")

        cls._address = f"127.0.0.1:{reserve_local_port()}"
        cls._server = subprocess.Popen(
            [
                str(cls._server_path),
                "--address",
                cls._address,
                "--index",
                "hnsw",
                "--load-index",
                str(cls._snapshot_path),
                "--dimension",
                "99",
                "--m",
                "99",
                "--read-only",
                "true",
            ],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
        )
        try:
            cls._client = VectorSearchClient(
                cls._address, cls._generated_directory, timeout_seconds=5.0
            )
        except (grpc.RpcError, grpc.FutureTimeoutError):
            stdout, stderr = stop_server(cls._server)
            cls._temporary_directory.cleanup()
            raise RuntimeError(f"HNSW server did not become ready\nstdout: {stdout}\nstderr: {stderr}")

    @classmethod
    def tearDownClass(cls) -> None:
        if hasattr(cls, "_client"):
            cls._client.close()
        if hasattr(cls, "_server"):
            stop_server(cls._server)
        if hasattr(cls, "_temporary_directory"):
            cls._temporary_directory.cleanup()

    def test_loaded_snapshot_serves_search_stats_and_rejects_writes(self) -> None:
        with grpc.insecure_channel(self._address) as channel:
            health = health_pb2_grpc.HealthStub(channel)
            response = health.Check(health_pb2.HealthCheckRequest(service=""), timeout=5.0)
            self.assertEqual(response.status, health_pb2.HealthCheckResponse.SERVING)

        result = self._client.search([1.0, 0.0, 0.0], 3)
        self.assertEqual([neighbor.id for neighbor in result.neighbors], [101, 102, 103])
        stats = self._client.stats()
        self.assertEqual(stats.index_type, "hnsw")
        self.assertEqual(stats.vector_count, 3)
        self.assertEqual(stats.dimension, 3)
        self.assertTrue(stats.read_only)
        self.assertEqual(stats.inserted_vectors, 0)
        # ID 999 is a tombstone in the fixture, even though its vector matches this query.
        surviving = self._client.search([0.0, 0.0, 1.0], 10)
        self.assertEqual([neighbor.id for neighbor in surviving.neighbors], [101, 102, 103])
        with self.assertRaises(grpc.RpcError) as context:
            self._client.add(104, [0.0, 0.0, 1.0])
        self.assertEqual(context.exception.code(), grpc.StatusCode.FAILED_PRECONDITION)

    def test_corrupt_snapshot_fails_before_listening(self) -> None:
        corrupt_path = Path(self._temporary_directory.name) / "corrupt.fvh"
        corrupt_path.write_bytes(b"corrupt snapshot")
        result = subprocess.run(
            [
                str(self._server_path),
                "--address",
                f"127.0.0.1:{reserve_local_port()}",
                "--index",
                "hnsw",
                "--load-index",
                str(corrupt_path),
            ],
            capture_output=True,
            text=True,
            timeout=5,
            check=False,
        )
        self.assertNotEqual(result.returncode, 0)
        records = [json.loads(line) for line in result.stderr.splitlines()]
        self.assertTrue(any(record["event"] == "index_load_failed" and
                            record["status"] == "FAILED_PRECONDITION" for record in records))
        self.assertNotIn(str(corrupt_path), result.stderr)

    def test_writable_hnsw_sighup_reload_is_rejected(self) -> None:
        result = subprocess.run(
            [
                str(self._server_path),
                "--index",
                "hnsw",
                "--load-index",
                str(self._snapshot_path),
                "--read-only",
                "false",
                "--reload-on-sighup",
                "true",
            ],
            capture_output=True,
            text=True,
            timeout=5,
            check=False,
        )
        self.assertNotEqual(result.returncode, 0)
        records = [json.loads(line) for line in result.stderr.splitlines()]
        self.assertEqual(records[-1]["event"], "startup_failed")
        self.assertEqual(records[-1]["status"], "INVALID_ARGUMENT")

    def test_loaded_snapshot_can_accept_runtime_writes(self) -> None:
        address = f"127.0.0.1:{reserve_local_port()}"
        server = subprocess.Popen(
            [
                str(self._server_path),
                "--address",
                address,
                "--index",
                "hnsw",
                "--load-index",
                str(self._snapshot_path),
            ],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
        )
        try:
            with VectorSearchClient(address, self._generated_directory, timeout_seconds=5.0) as client:
                self.assertEqual(client.add(104, [0.0, 0.0, 1.0]).index_size, 4)
                self.assertEqual(client.search([0.0, 0.0, 1.0], 1).neighbors[0].id, 104)
                stats = client.stats()
                self.assertFalse(stats.read_only)
                self.assertEqual(stats.inserted_vectors, 1)
        finally:
            stop_server(server)


class SnapshotReloadIntegrationTest(unittest.TestCase):
    def test_reload_requires_read_only_mode_and_snapshot_path(self) -> None:
        server = Path(os.environ.get("FAST_VECTOR_SERVER", REPOSITORY_ROOT / "build-grpc/fast_vector_server"))
        if not server.is_file():
            self.skipTest("gRPC server executable not found")
        for index_type in ("flat", "hnsw"):
            for arguments in (
                ["--read-only", "true"],
                ["--load-index", "unused-snapshot-path", "--read-only", "false"],
            ):
                with self.subTest(index_type=index_type, arguments=arguments):
                    result = subprocess.run(
                        [str(server), "--index", index_type, "--reload-on-sighup", "true", *arguments],
                        capture_output=True, text=True, timeout=5, check=False,
                    )
                    self.assertNotEqual(result.returncode, 0)
                    records = [json.loads(line) for line in result.stderr.splitlines()]
                    self.assertEqual(records[-1]["event"], "startup_failed")
                    self.assertEqual(records[-1]["status"], "INVALID_ARGUMENT")

    def test_flat_and_hnsw_reload_with_concurrent_queries_and_failed_replacements(self) -> None:
        server = Path(os.environ.get("FAST_VECTOR_SERVER", REPOSITORY_ROOT / "build-grpc/fast_vector_server"))
        builder = Path(os.environ.get("FAST_VECTOR_BUILDER", REPOSITORY_ROOT / "build-grpc/fast_vector_build_index"))
        fixture = Path(os.environ.get("FAST_VECTOR_HNSW_FIXTURE", REPOSITORY_ROOT / "build-grpc/fast_vector_hnsw_snapshot_fixture"))
        if not all(path.is_file() for path in (server, builder, fixture)):
            self.skipTest("snapshot reload executables not found")
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            generated = root / "generated"
            generate(REPOSITORY_ROOT / "proto", generated)
            flat = root / "original.fv"
            hnsw = root / "original.fvh"
            flat_replacement = root / "replacement.fv"
            hnsw_replacement = root / "replacement.fvh"
            data = root / "replacement.jsonl"
            data.write_text(json.dumps({"id": 999, "values": [0.0, 0.0, 1.0]}) + "\n", encoding="utf-8")
            commands = (
                [str(builder), "--input", str(REPOSITORY_ROOT / "tests/data/vectors.jsonl"),
                 "--output", str(flat), "--dimension", "3"],
                [str(builder), "--input", str(data), "--output", str(flat_replacement), "--dimension", "3"],
                [str(fixture), str(hnsw)],
                [str(fixture), str(hnsw_replacement), "--replacement"],
            )
            for command in commands:
                subprocess.run(command, capture_output=True, text=True, check=True, timeout=10)

            for index_type, original, replacement, other_format in (
                ("flat", flat, flat_replacement, hnsw),
                ("hnsw", hnsw, hnsw_replacement, flat),
            ):
                with self.subTest(index_type=index_type):
                    active = root / f"active-{index_type}"
                    active.write_bytes(original.read_bytes())
                    log_path = root / f"{index_type}.jsonl"
                    address = f"127.0.0.1:{reserve_local_port()}"
                    expected = (999,) if index_type == "flat" else (999, 1000)
                    with log_path.open("w", encoding="utf-8") as output:
                        process = subprocess.Popen(
                            [str(server), "--address", address, "--index", index_type,
                             "--load-index", str(active), "--read-only", "true",
                             "--reload-on-sighup", "true"],
                            stdout=subprocess.DEVNULL, stderr=output,
                        )
                        try:
                            with VectorSearchClient(address, generated, timeout_seconds=5.0) as client:
                                def query_many():
                                    for _ in range(30):
                                        result = client.search([0.0, 0.0, 1.0], 10)
                                        ids = tuple(neighbor.id for neighbor in result.neighbors)
                                        if ids not in ((101, 102, 103), expected):
                                            raise AssertionError(f"mixed snapshot result: {ids}")

                                self.assertEqual(client.stats().vector_count, 3)
                                with concurrent.futures.ThreadPoolExecutor(max_workers=4) as pool:
                                    readers = [pool.submit(query_many) for _ in range(4)]
                                    published = root / f"publish-{index_type}"
                                    published.write_bytes(replacement.read_bytes())
                                    published.replace(active)
                                    process.send_signal(signal.SIGHUP)
                                    deadline = time.monotonic() + 5.0
                                    while time.monotonic() < deadline:
                                        if client.search([0.0, 0.0, 1.0], 1).neighbors[0].id == 999:
                                            break
                                        time.sleep(0.02)
                                    else:
                                        self.fail("replacement snapshot was not published")
                                    for reader in readers:
                                        reader.result()
                                stats = client.stats()
                                self.assertEqual(stats.vector_count, len(expected))
                                self.assertEqual(stats.dimension, 3)
                                self.assertEqual(stats.index_type, index_type)
                                self.assertEqual(stats.inserted_vectors, 0)
                                for payload in (b"corrupt snapshot", other_format.read_bytes()):
                                    failures = sum(record["event"] == "index_load_failed"
                                                   for record in read_log_records(log_path))
                                    published.write_bytes(payload)
                                    published.replace(active)
                                    process.send_signal(signal.SIGHUP)
                                    deadline = time.monotonic() + 5.0
                                    while time.monotonic() < deadline:
                                        records = read_log_records(log_path)
                                        if sum(record["event"] == "index_load_failed" for record in records) > failures:
                                            break
                                        time.sleep(0.02)
                                    else:
                                        self.fail("failed replacement was not reported")
                                    result = client.search([0.0, 0.0, 1.0], 10)
                                    self.assertEqual(tuple(neighbor.id for neighbor in result.neighbors), expected)
                                    with grpc.insecure_channel(address) as channel:
                                        health = health_pb2_grpc.HealthStub(channel)
                                        response = health.Check(health_pb2.HealthCheckRequest(service=""), timeout=5.0)
                                        self.assertEqual(response.status, health_pb2.HealthCheckResponse.SERVING)
                                with self.assertRaises(grpc.RpcError) as rejected:
                                    client.add(5000, [1.0, 0.0, 0.0])
                                self.assertEqual(rejected.exception.code(), grpc.StatusCode.FAILED_PRECONDITION)
                        finally:
                            stop_server(process)
                    self.assertEqual(process.returncode, 0)
                    records = read_log_records(log_path)
                    self.assertTrue(any(record["event"] == "index_replaced" and
                                        record["index_type"] == index_type for record in records))
                    failed = [record for record in records if record["event"] == "index_load_failed"]
                    self.assertEqual(len(failed), 2)
                    self.assertTrue(all(record["kept_previous_index"] for record in failed))


class StructuredLoggingIntegrationTest(unittest.TestCase):
    def test_sigint_shutdown_reason(self) -> None:
        server_path = Path(os.environ.get(
            "FAST_VECTOR_SERVER", REPOSITORY_ROOT / "build-grpc" / "fast_vector_server"
        ))
        if not server_path.is_file():
            self.skipTest(f"gRPC server executable not found: {server_path}")
        with tempfile.TemporaryDirectory() as directory:
            log_path = Path(directory) / "server.jsonl"
            address = f"127.0.0.1:{reserve_local_port()}"
            with log_path.open("w", encoding="utf-8") as output:
                server = subprocess.Popen(
                    [str(server_path), "--address", address, "--index", "flat", "--dimension", "3"],
                    stdout=subprocess.DEVNULL, stderr=output,
                )
                try:
                    with grpc.insecure_channel(address) as channel:
                        grpc.channel_ready_future(channel).result(timeout=5.0)
                    server.send_signal(signal.SIGINT)
                    server.wait(timeout=5)
                    self.assertEqual(server.returncode, 0)
                finally:
                    if server.poll() is None:
                        server.kill()
                        server.wait(timeout=5)
            completed = [record for record in read_log_records(log_path)
                         if record["event"] == "shutdown_completed"]
            self.assertEqual(len(completed), 1)
            self.assertEqual(completed[0]["reason"], "SIGINT")

    def test_concurrent_rpc_logs_privacy_summary_and_shutdown(self) -> None:
        server_path = Path(os.environ.get(
            "FAST_VECTOR_SERVER", REPOSITORY_ROOT / "build-grpc" / "fast_vector_server"
        ))
        if not server_path.is_file():
            self.skipTest(f"gRPC server executable not found: {server_path}")
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            generated = root / "generated"
            generate(REPOSITORY_ROOT / "proto", generated)
            address = f"127.0.0.1:{reserve_local_port()}"
            log_path = root / "server.jsonl"
            with log_path.open("w", encoding="utf-8") as output:
                server = subprocess.Popen(
                    [str(server_path), "--address", address, "--index", "flat", "--dimension", "3",
                     "--max-batch-size", "2"],
                    stdout=subprocess.PIPE, stderr=output, text=True,
                )
                try:
                    with VectorSearchClient(address, generated, timeout_seconds=5.0) as client:
                        client.add(101, [98765.125, 0.0, 0.0])
                        client.add_batch([(102, [0.0, 1.0, 0.0])])
                        client.search([12345.625, 0.0, 0.0], 1)
                        with self.assertRaises(grpc.RpcError) as invalid:
                            client._stub.Search(
                                client._messages.SearchRequest(query=[123456.789], k=1),
                                metadata=(("authorization", "PRIVATE_AUTH_VALUE"),), timeout=5.0,
                            )
                        self.assertEqual(invalid.exception.code(), grpc.StatusCode.INVALID_ARGUMENT)
                        with self.assertRaises(grpc.RpcError) as oversized:
                            client.add_batch([(201 + i, [1.0, 0.0, 0.0]) for i in range(3)])
                        self.assertEqual(oversized.exception.code(), grpc.StatusCode.RESOURCE_EXHAUSTED)
                        client.stats()
                        with concurrent.futures.ThreadPoolExecutor(max_workers=4) as pool:
                            futures = [pool.submit(client.search, [1.0, 0.0, 0.0], 1)
                                       for _ in range(16)]
                            for future in futures:
                                self.assertEqual(future.result().neighbors[0].id, 101)
                finally:
                    stdout, _ = stop_server(server)
                self.assertEqual(server.returncode, 0)
                self.assertEqual(stdout, "")
            raw = log_path.read_text(encoding="utf-8")
            records = read_log_records(log_path)
            self.assertEqual(len(records), len(raw.splitlines()))
            for record in records:
                self.assertRegex(record["timestamp"], r"^\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}\.\d{3}Z$")
                self.assertIn(record["severity"], ("INFO", "WARN", "ERROR"))
                for field in ("event", "request_id", "method", "status", "latency_us",
                              "index_type", "vector_count"):
                    self.assertIn(field, record)
            completed = [record for record in records if record["event"] == "rpc_completed"]
            self.assertEqual(len(completed), 22)
            self.assertEqual(len({record["request_id"] for record in completed}), 22)
            self.assertEqual({record["method"] for record in completed},
                             {"AddVector", "BatchAdd", "Search", "GetStats"})
            for record in completed:
                self.assertTrue(record["request_id"].isdigit())
                self.assertGreaterEqual(record["latency_us"], 0)
                self.assertEqual(record["index_type"], "flat")
                self.assertIsInstance(record["vector_count"], int)
            statuses = Counter(record["status"] for record in completed)
            self.assertEqual(statuses, {"OK": 20, "INVALID_ARGUMENT": 1, "RESOURCE_EXHAUSTED": 1})
            error_summaries = {record["status"]: record["count"] for record in records
                               if record["event"] == "rpc_error_summary"}
            self.assertEqual(error_summaries, {"INVALID_ARGUMENT": 1, "RESOURCE_EXHAUSTED": 1})
            totals = [record for record in records if record["event"] == "rpc_summary"]
            self.assertEqual(len(totals), 1)
            self.assertEqual(totals[0]["completed_rpcs"], 22)
            self.assertEqual(totals[0]["successful_rpcs"], 20)
            self.assertEqual(totals[0]["failed_rpcs"], 2)
            self.assertEqual(totals[0]["dropped_log_records"], 0)
            shutdown = [record for record in records if record["event"] == "shutdown_completed"]
            self.assertEqual(len(shutdown), 1)
            self.assertEqual(shutdown[0]["reason"], "SIGTERM")
            config = next(record for record in records if record["event"] == "startup_config")
            self.assertEqual(config["dimension"], 3)
            self.assertEqual(config["max_batch_size"], 2)
            for private in ("98765.125", "12345.625", "123456.789", "PRIVATE_AUTH_VALUE"):
                self.assertNotIn(private, raw)
            for record in records:
                self.assertFalse({"query", "values", "text", "metadata", "error_message"} & record.keys())


if __name__ == "__main__":
    unittest.main()
