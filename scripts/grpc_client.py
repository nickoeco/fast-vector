#!/usr/bin/env python3
"""Command-line client and reusable Python API for fast-vector gRPC."""

from __future__ import annotations

import argparse
import importlib
import json
import sys
from pathlib import Path
from types import ModuleType
from typing import Sequence

import grpc


def load_generated_modules(generated_directory: Path) -> tuple[ModuleType, ModuleType]:
    """Load generated modules without committing them to the repository."""
    directory = str(generated_directory.resolve())
    if directory not in sys.path:
        sys.path.insert(0, directory)
    try:
        messages = importlib.import_module("fast_vector.v1.vector_search_pb2")
        service = importlib.import_module("fast_vector.v1.vector_search_pb2_grpc")
    except ModuleNotFoundError as error:
        raise RuntimeError(
            "Python gRPC modules are missing; run scripts/generate_grpc_python.py first"
        ) from error
    return messages, service


class VectorSearchClient:
    """Small synchronous client for the versioned vector search API."""

    def __init__(
        self,
        address: str,
        generated_directory: Path,
        timeout_seconds: float = 5.0,
        wait_for_ready: bool = True,
    ) -> None:
        if not address:
            raise ValueError("address must not be empty")
        if timeout_seconds <= 0:
            raise ValueError("timeout_seconds must be positive")
        self._messages, service = load_generated_modules(generated_directory)
        self._channel = grpc.insecure_channel(address)
        self._stub = service.VectorSearchServiceStub(self._channel)
        self._timeout_seconds = timeout_seconds
        if wait_for_ready:
            grpc.channel_ready_future(self._channel).result(timeout=timeout_seconds)

    def close(self) -> None:
        self._channel.close()

    def __enter__(self) -> "VectorSearchClient":
        return self

    def __exit__(self, *_: object) -> None:
        self.close()

    def add(self, vector_id: int, values: Sequence[float]):
        vector = self._messages.Vector(id=vector_id, values=values)
        request = self._messages.AddVectorRequest(vector=vector)
        return self._stub.AddVector(request, timeout=self._timeout_seconds)

    def add_batch(self, vectors: Sequence[tuple[int, Sequence[float]]]):
        request = self._messages.BatchAddRequest(
            vectors=[self._messages.Vector(id=vector_id, values=values) for vector_id, values in vectors]
        )
        return self._stub.BatchAdd(request, timeout=self._timeout_seconds)

    def search(self, query: Sequence[float], k: int):
        request = self._messages.SearchRequest(query=query, k=k)
        return self._stub.Search(request, timeout=self._timeout_seconds)

    def stats(self):
        return self._stub.GetStats(
            self._messages.GetStatsRequest(), timeout=self._timeout_seconds
        )


def parse_vector(value: str) -> list[float]:
    try:
        values = [float(component) for component in value.split(",")]
    except ValueError as error:
        raise argparse.ArgumentTypeError("vectors must contain comma-separated floats") from error
    if not values:
        raise argparse.ArgumentTypeError("vectors must not be empty")
    return values


def response_to_dictionary(response: object) -> dict[str, object]:
    if hasattr(response, "neighbors"):
        return {
            "neighbors": [
                {"id": neighbor.id, "score": neighbor.score}
                for neighbor in response.neighbors
            ]
        }
    fields = response.DESCRIPTOR.fields
    return {field.name: getattr(response, field.name) for field in fields}


def parse_arguments(arguments: Sequence[str] | None = None) -> argparse.Namespace:
    repository_root = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--address", default="127.0.0.1:50051")
    parser.add_argument(
        "--generated-dir", type=Path, default=repository_root / "build" / "python-grpc"
    )
    parser.add_argument("--timeout", type=float, default=5.0)
    subparsers = parser.add_subparsers(dest="command", required=True)

    add_parser = subparsers.add_parser("add")
    add_parser.add_argument("--id", type=int, required=True)
    add_parser.add_argument("--vector", type=parse_vector, required=True)

    batch_parser = subparsers.add_parser("batch-add")
    batch_parser.add_argument(
        "--input", type=Path, required=True, help="JSON array of objects with id and values"
    )

    search_parser = subparsers.add_parser("search")
    search_parser.add_argument("--query", type=parse_vector, required=True)
    search_parser.add_argument("--k", type=int, required=True)

    subparsers.add_parser("stats")
    return parser.parse_args(arguments)


def main(arguments: Sequence[str] | None = None) -> int:
    options = parse_arguments(arguments)
    with VectorSearchClient(options.address, options.generated_dir, options.timeout) as client:
        if options.command == "add":
            response = client.add(options.id, options.vector)
        elif options.command == "batch-add":
            records = json.loads(options.input.read_text(encoding="utf-8"))
            vectors = [(record["id"], record["values"]) for record in records]
            response = client.add_batch(vectors)
        elif options.command == "search":
            response = client.search(options.query, options.k)
        else:
            response = client.stats()
    print(json.dumps(response_to_dictionary(response), indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
