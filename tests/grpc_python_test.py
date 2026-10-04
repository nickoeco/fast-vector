"""End-to-end tests for the Python gRPC client."""

from __future__ import annotations

import os
import socket
import subprocess
import sys
import tempfile
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
    server.terminate()
    try:
        return server.communicate(timeout=5)
    except subprocess.TimeoutExpired:
        server.kill()
        return server.communicate(timeout=5)


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
                "--max-batch-size",
                "10",
                "--kernel",
                "scalar",
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
            cls._temporary_directory.cleanup()
            raise RuntimeError(f"server did not become ready\nstdout: {stdout}\nstderr: {stderr}")

    @classmethod
    def tearDownClass(cls) -> None:
        if hasattr(cls, "_client"):
            cls._client.close()
        if hasattr(cls, "_server"):
            stop_server(cls._server)
        if hasattr(cls, "_temporary_directory"):
            cls._temporary_directory.cleanup()

    def test_snapshot_server_lifecycle(self) -> None:
        with grpc.insecure_channel(self._address) as channel:
            health = health_pb2_grpc.HealthStub(channel)
            response = health.Check(
                health_pb2.HealthCheckRequest(service=""), timeout=5.0
            )
            self.assertEqual(response.status, health_pb2.HealthCheckResponse.SERVING)

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


if __name__ == "__main__":
    unittest.main()
