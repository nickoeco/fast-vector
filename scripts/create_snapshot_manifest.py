#!/usr/bin/env python3
"""Describe immutable snapshot bytes; replicas independently validate the supplied live shape."""
from __future__ import annotations

import argparse
import hashlib
import json
import struct
from pathlib import Path


def create_manifest(path: Path, index_type: str, dimension: int, vector_count: int,
                    embedding_revision: str = "") -> dict:
    if index_type not in ("flat", "hnsw") or not 0 < dimension < 2**64 or not 0 <= vector_count < 2**64:
        raise ValueError("invalid snapshot shape")
    if len(embedding_revision.encode("utf-8")) > 256:
        raise ValueError("embedding revision too long")
    digest = hashlib.sha256()
    with path.open("rb") as source:
        header = source.read(12)
        if len(header) != 12 or header[:8] != (b"FVINDEX\0" if index_type == "flat" else b"FVHNSW\0\0"):
            raise ValueError("snapshot magic mismatch")
        version = struct.unpack_from("<I", header, 8)[0]
        if version not in ((1,) if index_type == "flat" else (1, 2, 3)):
            raise ValueError("unsupported snapshot format")
        digest.update(header)
        size = 12
        while chunk := source.read(1024 * 1024):
            size += len(chunk)
            digest.update(chunk)
    checksum = digest.hexdigest()
    return {"format_version": version, "snapshot_id": f"sha256:{checksum}",
            "index_type": index_type, "dimension": dimension, "vector_count": vector_count,
            "file_size": size, "sha256": checksum, "embedding_revision": embedding_revision}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--index", type=Path, required=True)
    parser.add_argument("--index-type", choices=("flat", "hnsw"), required=True)
    parser.add_argument("--dimension", type=int, required=True)
    parser.add_argument("--vector-count", type=int, required=True, help="live vectors, excluding tombstones")
    parser.add_argument("--embedding-revision", default="")
    parser.add_argument("--output", type=Path, required=True)
    options = parser.parse_args()
    manifest = create_manifest(options.index, options.index_type, options.dimension,
                               options.vector_count, options.embedding_revision)
    # Exclusive creation prevents accidentally overwriting an immutable release descriptor.
    with options.output.open("x", encoding="utf-8") as output:
        json.dump(manifest, output, indent=2, sort_keys=True)
        output.write("\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
