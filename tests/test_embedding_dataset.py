"""Tests for the deterministic embedding data pipeline."""

from __future__ import annotations

import json
import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np


REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPOSITORY_ROOT / "scripts"))

from build_embedding_dataset import (  # noqa: E402
    Document,
    chunk_text,
    make_chunks,
    stable_vector_id,
    write_dataset,
)


class EmbeddingDatasetTest(unittest.TestCase):
    def test_chunking_is_deterministic_and_overlapping(self) -> None:
        chunks = chunk_text("one two three four five six", 4, 2)
        self.assertEqual(chunks, ["one two three four", "three four five six"])
        self.assertEqual(chunks, chunk_text("one  two\nthree four five six", 4, 2))

    def test_chunking_rejects_invalid_configuration(self) -> None:
        with self.assertRaises(ValueError):
            chunk_text("text", 0, 0)
        with self.assertRaises(ValueError):
            chunk_text("text", 4, 4)

    def test_vector_ids_are_stable_and_include_chunk_identity(self) -> None:
        self.assertEqual(stable_vector_id("doc", 0), stable_vector_id("doc", 0))
        self.assertNotEqual(stable_vector_id("doc", 0), stable_vector_id("doc", 1))
        self.assertNotEqual(stable_vector_id("doc-a", 0), stable_vector_id("doc-b", 0))

    def test_writes_separate_vectors_chunks_and_manifest(self) -> None:
        chunks = make_chunks(
            [Document("doc-1", "alpha beta gamma delta", "memory://one")], 3, 1
        )

        def fake_encode(texts: list[str]) -> np.ndarray:
            basis = ([1.0, 0.0], [0.0, 1.0])
            return np.asarray([basis[index % len(basis)] for index, _ in enumerate(texts)])

        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory)
            manifest = write_dataset(chunks, fake_encode, output, "fake-model", 3, 1)
            vectors = [
                json.loads(line)
                for line in (output / "vectors.jsonl").read_text(encoding="utf-8").splitlines()
            ]
            chunk_records = [
                json.loads(line)
                for line in (output / "chunks.jsonl").read_text(encoding="utf-8").splitlines()
            ]

        self.assertEqual(manifest["dimension"], 2)
        self.assertEqual(manifest["chunk_count"], 2)
        self.assertEqual([record["id"] for record in vectors], [chunk.vector_id for chunk in chunks])
        self.assertEqual(manifest["chunks_file"], "chunks.jsonl")
        self.assertEqual(
            [record["text"] for record in chunk_records], [chunk.text for chunk in chunks]
        )


if __name__ == "__main__":
    unittest.main()
