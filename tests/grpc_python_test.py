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
            cls._server.terminate()
            stdout, stderr = cls._server.communicate(timeout=5)
            raise RuntimeError(f"server did not become ready\nstdout: {stdout}\nstderr: {stderr}")

    @classmethod
    def tearDownClass(cls) -> None:
        if hasattr(cls, "_client"):
            cls._client.close()
        if hasattr(cls, "_server"):
            cls._server.terminate()
            try:
                cls._server.communicate(timeout=5)
            except subprocess.TimeoutExpired:
                cls._server.kill()
                cls._server.communicate(timeout=5)
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

    def test_invalid_vector_surfaces_grpc_status(self) -> None:
        with self.assertRaises(grpc.RpcError) as context:
            self._client.add(200, [1.0, 0.0])
        self.assertEqual(context.exception.code(), grpc.StatusCode.INVALID_ARGUMENT)


if __name__ == "__main__":
    unittest.main()
