#!/usr/bin/env python3
"""Generate Python protobuf and gRPC modules into a build directory."""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Sequence

from grpc_tools import protoc


def generate(proto_root: Path, output_directory: Path) -> None:
    """Generate Python modules for the vector search service contract."""
    proto_file = proto_root / "fast_vector" / "v1" / "vector_search.proto"
    if not proto_file.is_file():
        raise FileNotFoundError(f"protobuf contract not found: {proto_file}")

    output_directory.mkdir(parents=True, exist_ok=True)
    exit_code = protoc.main(
        [
            "grpc_tools.protoc",
            f"--proto_path={proto_root}",
            f"--python_out={output_directory}",
            f"--grpc_python_out={output_directory}",
            str(proto_file),
        ]
    )
    if exit_code != 0:
        raise RuntimeError(f"grpc_tools.protoc failed with exit code {exit_code}")


def parse_arguments(arguments: Sequence[str] | None = None) -> argparse.Namespace:
    repository_root = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--proto-root", type=Path, default=repository_root / "proto")
    parser.add_argument(
        "--output-dir", type=Path, default=repository_root / "build" / "python-grpc"
    )
    return parser.parse_args(arguments)


def main(arguments: Sequence[str] | None = None) -> int:
    options = parse_arguments(arguments)
    generate(options.proto_root.resolve(), options.output_dir.resolve())
    print(f"Generated Python gRPC modules in {options.output_dir.resolve()}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
