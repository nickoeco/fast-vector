#!/usr/bin/env python3
"""Chunk JSONL documents and export embeddings with separate chunk records."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Callable, Iterable, Sequence


@dataclass(frozen=True)
class Document:
    document_id: str
    text: str
    source: str = ""


@dataclass(frozen=True)
class Chunk:
    vector_id: int
    document_id: str
    chunk_id: int
    text: str
    source: str


def chunk_text(text: str, maximum_words: int, overlap_words: int) -> list[str]:
    """Split normalized whitespace into deterministic overlapping word windows."""
    if maximum_words <= 0:
        raise ValueError("maximum_words must be positive")
    if overlap_words < 0 or overlap_words >= maximum_words:
        raise ValueError("overlap_words must be in [0, maximum_words)")
    words = text.split()
    if not words:
        return []
    step = maximum_words - overlap_words
    chunks: list[str] = []
    for start in range(0, len(words), step):
        chunks.append(" ".join(words[start : start + maximum_words]))
        if start + maximum_words >= len(words):
            break
    return chunks


def stable_vector_id(document_id: str, chunk_id: int) -> int:
    """Create a stable unsigned 64-bit ID independent of Python's hash seed."""
    identity = f"{document_id}\0{chunk_id}".encode("utf-8")
    return int.from_bytes(hashlib.blake2b(identity, digest_size=8).digest(), "little")


def read_documents(path: Path) -> list[Document]:
    documents: list[Document] = []
    seen_ids: set[str] = set()
    with path.open(encoding="utf-8") as input_file:
        for line_number, line in enumerate(input_file, start=1):
            if not line.strip():
                continue
            record = json.loads(line)
            document = Document(
                document_id=str(record["document_id"]),
                text=str(record["text"]),
                source=str(record.get("source", "")),
            )
            if not document.document_id:
                raise ValueError(f"line {line_number}: document_id must not be empty")
            if document.document_id in seen_ids:
                raise ValueError(f"line {line_number}: duplicate document_id {document.document_id}")
            seen_ids.add(document.document_id)
            documents.append(document)
    return documents


def make_chunks(
    documents: Iterable[Document], maximum_words: int, overlap_words: int
) -> list[Chunk]:
    chunks: list[Chunk] = []
    seen_vector_ids: set[int] = set()
    for document in documents:
        for chunk_id, text in enumerate(chunk_text(document.text, maximum_words, overlap_words)):
            vector_id = stable_vector_id(document.document_id, chunk_id)
            if vector_id in seen_vector_ids:
                raise RuntimeError("stable vector ID collision")
            seen_vector_ids.add(vector_id)
            chunks.append(
                Chunk(vector_id, document.document_id, chunk_id, text, document.source)
            )
    return chunks


def write_dataset(
    chunks: Sequence[Chunk],
    encode: Callable[[Sequence[str]], Sequence[Sequence[float]]],
    output_directory: Path,
    model_name: str,
    maximum_words: int,
    overlap_words: int,
) -> dict[str, object]:
    embeddings = encode([chunk.text for chunk in chunks]) if chunks else []
    if len(embeddings) != len(chunks):
        raise ValueError("embedding count does not match chunk count")
    dimension = len(embeddings[0]) if len(embeddings) > 0 else 0
    output_directory.mkdir(parents=True, exist_ok=True)
    vectors_path = output_directory / "vectors.jsonl"
    chunks_path = output_directory / "chunks.jsonl"
    with vectors_path.open("w", encoding="utf-8") as vectors_file, chunks_path.open(
        "w", encoding="utf-8"
    ) as chunks_file:
        for chunk, embedding in zip(chunks, embeddings):
            values = [float(value) for value in embedding]
            if len(values) != dimension or not all(math.isfinite(value) for value in values):
                raise ValueError("embeddings must have one finite fixed dimension")
            squared_norm = sum(value * value for value in values)
            if not math.isclose(squared_norm, 1.0, rel_tol=1.0e-3, abs_tol=1.0e-3):
                raise ValueError("embeddings must be L2-normalized")
            vectors_file.write(json.dumps({"id": chunk.vector_id, "values": values}) + "\n")
            chunks_file.write(json.dumps(asdict(chunk), ensure_ascii=False) + "\n")
    manifest = {
        "format_version": 1,
        "model": model_name,
        "dimension": dimension,
        "chunk_count": len(chunks),
        "maximum_words": maximum_words,
        "overlap_words": overlap_words,
        "normalized_embeddings": True,
        "vectors_file": vectors_path.name,
        "chunks_file": chunks_path.name,
    }
    (output_directory / "manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    return manifest


def parse_arguments(arguments: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--model", default="sentence-transformers/all-MiniLM-L6-v2")
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--maximum-words", type=int, default=120)
    parser.add_argument("--overlap-words", type=int, default=20)
    parser.add_argument("--device", default=None)
    return parser.parse_args(arguments)


def main(arguments: Sequence[str] | None = None) -> int:
    options = parse_arguments(arguments)
    if options.batch_size <= 0:
        raise ValueError("batch_size must be positive")
    documents = read_documents(options.input)
    chunks = make_chunks(documents, options.maximum_words, options.overlap_words)

    from sentence_transformers import SentenceTransformer

    model = SentenceTransformer(options.model, device=options.device)

    def encode(texts: Sequence[str]) -> Sequence[Sequence[float]]:
        method = getattr(model, "encode_document", model.encode)
        return method(
            list(texts),
            batch_size=options.batch_size,
            convert_to_numpy=True,
            normalize_embeddings=True,
            show_progress_bar=True,
        )

    manifest = write_dataset(
        chunks,
        encode,
        options.output_dir,
        options.model,
        options.maximum_words,
        options.overlap_words,
    )
    print(json.dumps(manifest, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
