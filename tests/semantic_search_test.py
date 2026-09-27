"""Validation and localhost integration tests for semantic search."""

from __future__ import annotations

import json
import os
import socket
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

import grpc


REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPOSITORY_ROOT / "scripts"))

from generate_grpc_python import generate  # noqa: E402
from grpc_client import VectorSearchClient  # noqa: E402
from semantic_search import ingest_dataset, load_dataset, search_chunks  # noqa: E402


def write_dataset(directory: Path, chunk_ids: tuple[int, ...] = (11, 12)) -> None:
    vectors = ((11, [1.0, 0.0, 0.0]), (12, [0.0, 1.0, 0.0]))
    (directory / "manifest.json").write_text(
        json.dumps(
            {
                "format_version": 1,
                "model": "fake-model",
                "dimension": 3,
                "chunk_count": 2,
                "normalized_embeddings": True,
                "vectors_file": "vectors.jsonl",
                "chunks_file": "chunks.jsonl",
            }
        ),
        encoding="utf-8",
    )
    (directory / "vectors.jsonl").write_text(
        "".join(json.dumps({"id": vector_id, "values": values}) + "\n" for vector_id, values in vectors),
        encoding="utf-8",
    )
    (directory / "chunks.jsonl").write_text(
        "".join(
            json.dumps(
                {
                    "vector_id": vector_id,
                    "document_id": f"doc-{vector_id}",
                    "chunk_id": 0,
                    "text": f"chunk {vector_id}",
                    "source": "test",
                }
            )
            + "\n"
            for vector_id in chunk_ids
        ),
        encoding="utf-8",
    )


def reserve_local_port() -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as listener:
        listener.bind(("127.0.0.1", 0))
        return listener.getsockname()[1]


class DatasetValidationTest(unittest.TestCase):
    def test_rejects_vector_and_chunk_id_mismatch(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            directory = Path(temporary_directory)
            write_dataset(directory, (11, 99))
            with self.assertRaisesRegex(ValueError, "different IDs"):
                load_dataset(directory)


class SemanticSearchIntegrationTest(unittest.TestCase):
    def test_ingests_searches_and_maps_chunk(self) -> None:
        server_path = Path(
            os.environ.get("FAST_VECTOR_SERVER", REPOSITORY_ROOT / "build-grpc" / "fast_vector_server")
        )
        if not server_path.is_file():
            self.skipTest(f"gRPC server executable not found: {server_path}")
        with tempfile.TemporaryDirectory() as temporary_directory:
            directory = Path(temporary_directory)
            generated_directory = directory / "generated"
            write_dataset(directory)
            generate(REPOSITORY_ROOT / "proto", generated_directory)
            dataset = load_dataset(directory)
            address = f"127.0.0.1:{reserve_local_port()}"
            server = subprocess.Popen(
                [
                    str(server_path),
                    "--address",
                    address,
                    "--index",
                    "flat",
                    "--dimension",
                    "3",
                    "--max-batch-size",
                    "1",
                ],
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
            )
            try:
                with VectorSearchClient(address, generated_directory, 5.0) as client:
                    self.assertEqual(ingest_dataset(client, dataset, 1), 2)
                    results = search_chunks(client, dataset, [1.0, 0.0, 0.0], 1)
                self.assertEqual(results[0]["vector_id"], 11)
                self.assertEqual(results[0]["text"], "chunk 11")
                self.assertAlmostEqual(results[0]["score"], 1.0, places=5)
            except (grpc.RpcError, grpc.FutureTimeoutError) as error:
                self.fail(f"localhost gRPC integration failed: {error}")
            finally:
                server.terminate()
                try:
                    server.wait(timeout=5)
                except subprocess.TimeoutExpired:
                    server.kill()
                    server.wait(timeout=5)


if __name__ == "__main__":
    unittest.main()
