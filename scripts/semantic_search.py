#!/usr/bin/env python3
"""Load an embedding dataset, populate fast-vector, and run semantic search."""

from __future__ import annotations

import argparse
import atexit
import json
import math
import subprocess
from dataclasses import dataclass
from pathlib import Path
from typing import Sequence

from grpc_client import VectorSearchClient


@dataclass(frozen=True)
class EmbeddingDataset:
    model: str
    dimension: int
    vectors: list[tuple[int, list[float]]]
    chunks: dict[int, dict[str, object]]


def read_jsonl(path: Path) -> list[dict[str, object]]:
    records: list[dict[str, object]] = []
    with path.open(encoding="utf-8") as input_file:
        for line_number, line in enumerate(input_file, start=1):
            if not line.strip():
                continue
            try:
                records.append(json.loads(line))
            except json.JSONDecodeError as error:
                raise ValueError(f"{path}:{line_number}: invalid JSON") from error
    return records


def load_dataset(directory: Path) -> EmbeddingDataset:
    manifest_path = directory / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if manifest.get("format_version") != 1:
        raise ValueError("unsupported embedding dataset format version")
    if manifest.get("normalized_embeddings") is not True:
        raise ValueError("embedding dataset must contain normalized vectors")
    dimension = int(manifest["dimension"])
    expected_count = int(manifest["chunk_count"])
    if dimension <= 0 or expected_count < 0:
        raise ValueError("manifest dimension and chunk count are invalid")

    vector_records = read_jsonl(directory / str(manifest["vectors_file"]))
    chunk_records = read_jsonl(directory / str(manifest["chunks_file"]))
    vectors: list[tuple[int, list[float]]] = []
    vector_ids: set[int] = set()
    for record in vector_records:
        vector_id = int(record["id"])
        values = [float(value) for value in record["values"]]
        if vector_id in vector_ids:
            raise ValueError(f"duplicate vector ID {vector_id}")
        if len(values) != dimension or not all(math.isfinite(value) for value in values):
            raise ValueError(f"vector {vector_id} has invalid values")
        squared_norm = sum(value * value for value in values)
        if not math.isclose(squared_norm, 1.0, rel_tol=1.0e-3, abs_tol=1.0e-3):
            raise ValueError(f"vector {vector_id} is not L2-normalized")
        vector_ids.add(vector_id)
        vectors.append((vector_id, values))

    chunks: dict[int, dict[str, object]] = {}
    for record in chunk_records:
        vector_id = int(record["vector_id"])
        if vector_id in chunks:
            raise ValueError(f"duplicate chunk vector ID {vector_id}")
        chunks[vector_id] = record
    if len(vectors) != expected_count or len(chunks) != expected_count:
        raise ValueError("manifest chunk count does not match dataset files")
    if vector_ids != chunks.keys():
        raise ValueError("vectors and chunks contain different IDs")
    return EmbeddingDataset(str(manifest["model"]), dimension, vectors, chunks)


def ingest_dataset(
    client: VectorSearchClient, dataset: EmbeddingDataset, batch_size: int
) -> int:
    if batch_size <= 0:
        raise ValueError("batch_size must be positive")
    inserted = 0
    for offset in range(0, len(dataset.vectors), batch_size):
        batch = dataset.vectors[offset : offset + batch_size]
        response = client.add_batch(batch)
        if response.added_count != len(batch):
            raise RuntimeError("server reported an unexpected inserted count")
        inserted += response.added_count
    return inserted


def search_chunks(
    client: VectorSearchClient,
    dataset: EmbeddingDataset,
    query_vector: Sequence[float],
    k: int,
) -> list[dict[str, object]]:
    if len(query_vector) != dataset.dimension:
        raise ValueError("query embedding dimension does not match the dataset")
    response = client.search(query_vector, k)
    results: list[dict[str, object]] = []
    for neighbor in response.neighbors:
        chunk = dataset.chunks.get(neighbor.id)
        if chunk is None:
            raise RuntimeError(f"search returned unknown vector ID {neighbor.id}")
        results.append({"vector_id": neighbor.id, "score": neighbor.score, **chunk})
    return results


def stop_server(server: subprocess.Popen[bytes]) -> None:
    if server.poll() is not None:
        return
    server.terminate()
    try:
        server.wait(timeout=5)
    except subprocess.TimeoutExpired:
        server.kill()
        server.wait(timeout=5)


def parse_arguments(arguments: Sequence[str] | None = None) -> argparse.Namespace:
    repository_root = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-dir", type=Path, required=True)
    parser.add_argument("--query", required=True)
    parser.add_argument("--k", type=int, default=5)
    parser.add_argument("--batch-size", type=int, default=500)
    parser.add_argument("--address", default="127.0.0.1:50051")
    parser.add_argument("--timeout", type=float, default=10.0)
    parser.add_argument("--device", default=None)
    parser.add_argument("--skip-ingest", action="store_true")
    parser.add_argument("--server-executable", type=Path)
    parser.add_argument("--index", choices=("flat", "hnsw"), default="flat")
    parser.add_argument(
        "--generated-dir", type=Path, default=repository_root / "build" / "python-grpc"
    )
    options = parser.parse_args(arguments)
    if options.k <= 0:
        parser.error("--k must be positive")
    if options.batch_size <= 0:
        parser.error("--batch-size must be positive")
    return options


def main(arguments: Sequence[str] | None = None) -> int:
    options = parse_arguments(arguments)
    dataset = load_dataset(options.dataset_dir)
    server = None
    if options.server_executable is not None:
        if options.skip_ingest:
            raise ValueError("--skip-ingest cannot be used with a newly started server")
        server = subprocess.Popen(
            [
                str(options.server_executable),
                "--address",
                options.address,
                "--index",
                options.index,
                "--dimension",
                str(dataset.dimension),
                "--max-batch-size",
                str(options.batch_size),
            ],
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        )
        atexit.register(stop_server, server)

    from sentence_transformers import SentenceTransformer

    model = SentenceTransformer(dataset.model, device=options.device)
    method = getattr(model, "encode_query", model.encode)
    query_vector = method(
        options.query, convert_to_numpy=True, normalize_embeddings=True
    ).tolist()
    with VectorSearchClient(options.address, options.generated_dir, options.timeout) as client:
        inserted = 0 if options.skip_ingest else ingest_dataset(client, dataset, options.batch_size)
        results = search_chunks(client, dataset, query_vector, options.k)
    print(
        json.dumps(
            {"query": options.query, "inserted_vectors": inserted, "results": results},
            ensure_ascii=False,
            indent=2,
        )
    )
    if server is not None:
        stop_server(server)
        atexit.unregister(stop_server)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
