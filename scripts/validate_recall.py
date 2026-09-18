#!/usr/bin/env python3
"""Generate exact cosine ground truth and measure a C++ FlatIndex or HNSW index."""

from __future__ import annotations

import argparse
import csv
import json
import subprocess
from pathlib import Path

import numpy as np


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--runner", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--vector-count", type=int, default=1_000)
    parser.add_argument("--query-count", type=int, default=100)
    parser.add_argument("--dimension", type=int, default=64)
    parser.add_argument("--k", type=int, default=10)
    parser.add_argument("--seed", type=int, default=20250908)
    parser.add_argument("--kernel", choices=("scalar", "auto", "avx2"), default="scalar")
    parser.add_argument("--index", choices=("flat", "hnsw"), default="flat")
    parser.add_argument("--m", type=int, default=16)
    parser.add_argument("--ef-construction", type=int, default=200)
    parser.add_argument("--ef-search", type=int, default=50)
    parser.add_argument("--hnsw-seed", type=int, default=42)
    parser.add_argument(
        "--neighbor-selection", choices=("simple", "heuristic"), default="simple"
    )
    parser.add_argument("--minimum-recall", type=float)
    return parser.parse_args()


def require_positive(name: str, value: int) -> None:
    if value <= 0:
        raise ValueError(f"{name} must be positive")


def normalize_like_cpp(values: np.ndarray) -> np.ndarray:
    norms = np.sqrt(np.sum(values.astype(np.float64) ** 2, axis=1))
    if np.any(norms == 0.0):
        raise ValueError("generated data unexpectedly contains a zero vector")
    return (values.astype(np.float64) / norms[:, None]).astype(np.float32)


def write_vectors(path: Path, ids: np.ndarray, values: np.ndarray, id_name: str) -> None:
    with path.open("w", newline="", encoding="utf-8") as output:
        writer = csv.writer(output, lineterminator="\n")
        writer.writerow([id_name, *(f"x{i}" for i in range(values.shape[1]))])
        for item_id, row in zip(ids, values, strict=True):
            writer.writerow([int(item_id), *(format(float(value), ".9g") for value in row)])


def exact_top_k(
    vector_ids: np.ndarray, vectors: np.ndarray, queries: np.ndarray, k: int
) -> tuple[np.ndarray, np.ndarray]:
    normalized_vectors = normalize_like_cpp(vectors)
    normalized_queries = normalize_like_cpp(queries)
    scores = normalized_queries.astype(np.float64) @ normalized_vectors.astype(np.float64).T
    result_count = min(k, len(vector_ids))
    top_ids = np.empty((len(queries), result_count), dtype=np.uint64)
    top_scores = np.empty((len(queries), result_count), dtype=np.float64)
    for query_index, query_scores in enumerate(scores):
        order = np.lexsort((vector_ids, -query_scores))[:result_count]
        top_ids[query_index] = vector_ids[order]
        top_scores[query_index] = query_scores[order]
    return top_ids, top_scores


def write_results(
    path: Path, query_ids: np.ndarray, result_ids: np.ndarray, scores: np.ndarray
) -> None:
    with path.open("w", newline="", encoding="utf-8") as output:
        writer = csv.writer(output, lineterminator="\n")
        writer.writerow(["query_id", "rank", "vector_id", "score"])
        for query_index, query_id in enumerate(query_ids):
            for rank in range(result_ids.shape[1]):
                writer.writerow(
                    [
                        int(query_id),
                        rank,
                        int(result_ids[query_index, rank]),
                        format(float(scores[query_index, rank]), ".17g"),
                    ]
                )


def read_cpp_results(
    path: Path, query_ids: np.ndarray, result_count: int
) -> tuple[np.ndarray, np.ndarray]:
    ids = np.empty((len(query_ids), result_count), dtype=np.uint64)
    scores = np.empty((len(query_ids), result_count), dtype=np.float64)
    ranks_seen = np.zeros((len(query_ids), result_count), dtype=np.bool_)
    query_positions = {int(query_id): index for index, query_id in enumerate(query_ids)}

    with path.open(newline="", encoding="utf-8") as input_file:
        for row in csv.DictReader(input_file):
            query_id = int(row["query_id"])
            rank = int(row["rank"])
            if query_id not in query_positions or not 0 <= rank < result_count:
                raise ValueError("C++ result contains an unexpected query ID or rank")
            position = query_positions[query_id]
            if ranks_seen[position, rank]:
                raise ValueError("C++ result contains a duplicate query rank")
            ids[position, rank] = int(row["vector_id"])
            scores[position, rank] = float(row["score"])
            ranks_seen[position, rank] = True

    if not np.all(ranks_seen):
        raise ValueError("C++ result does not contain exactly K rows for every query")
    return ids, scores


def recall_at_k(expected_ids: np.ndarray, actual_ids: np.ndarray) -> np.ndarray:
    if expected_ids.shape != actual_ids.shape or expected_ids.shape[1] == 0:
        raise ValueError("expected and actual IDs must have the same non-empty K dimension")
    return np.array(
        [
            len(set(expected.tolist()) & set(actual.tolist())) / len(expected)
            for expected, actual in zip(expected_ids, actual_ids, strict=True)
        ]
    )


def scores_for_result_ids(
    vector_ids: np.ndarray,
    vectors: np.ndarray,
    queries: np.ndarray,
    result_ids: np.ndarray,
) -> np.ndarray:
    normalized_vectors = normalize_like_cpp(vectors)
    normalized_queries = normalize_like_cpp(queries)
    positions = {int(vector_id): position for position, vector_id in enumerate(vector_ids)}
    scores = np.empty(result_ids.shape, dtype=np.float64)
    for query_index, ids in enumerate(result_ids):
        for rank, vector_id in enumerate(ids):
            if int(vector_id) not in positions:
                raise ValueError("C++ result contains an unknown vector ID")
            vector = normalized_vectors[positions[int(vector_id)]]
            scores[query_index, rank] = np.dot(
                normalized_queries[query_index].astype(np.float64),
                vector.astype(np.float64),
            )
    return scores


def main() -> int:
    args = parse_args()
    for name in ("vector_count", "query_count", "dimension", "k", "m", "ef_construction", "ef_search"):
        require_positive(name, getattr(args, name))
    if args.hnsw_seed < 0:
        raise ValueError("hnsw_seed must be non-negative")
    if args.minimum_recall is not None and not 0.0 <= args.minimum_recall <= 1.0:
        raise ValueError("minimum_recall must be between zero and one")
    if not args.runner.is_file():
        raise FileNotFoundError(f"validation runner not found: {args.runner}")

    args.output_dir.mkdir(parents=True, exist_ok=True)
    vectors_path = args.output_dir / "vectors.csv"
    queries_path = args.output_dir / "queries.csv"
    truth_path = args.output_dir / "ground_truth.csv"
    cpp_path = args.output_dir / "cpp_results.csv"

    generator = np.random.default_rng(args.seed)
    vectors = generator.uniform(-1.0, 1.0, (args.vector_count, args.dimension)).astype(
        np.float32
    )
    queries = generator.uniform(-1.0, 1.0, (args.query_count, args.dimension)).astype(
        np.float32
    )
    vector_ids = generator.permutation(args.vector_count).astype(np.uint64) + 10_000
    query_ids = np.arange(args.query_count, dtype=np.uint64)

    write_vectors(vectors_path, vector_ids, vectors, "vector_id")
    write_vectors(queries_path, query_ids, queries, "query_id")
    truth_ids, truth_scores = exact_top_k(vector_ids, vectors, queries, args.k)
    write_results(truth_path, query_ids, truth_ids, truth_scores)

    runner_command = [
        str(args.runner),
        str(vectors_path),
        str(queries_path),
        str(args.k),
        str(cpp_path),
        args.kernel,
    ]
    if args.index == "hnsw":
        runner_command.extend(
            [
                "hnsw",
                str(args.m),
                str(args.ef_construction),
                str(args.ef_search),
                str(args.hnsw_seed),
                args.neighbor_selection,
            ]
        )
    subprocess.run(runner_command, check=True)
    cpp_ids, cpp_scores = read_cpp_results(cpp_path, query_ids, truth_ids.shape[1])

    per_query_recall = recall_at_k(truth_ids, cpp_ids)
    mean_recall = float(np.mean(per_query_recall))
    exact_id_match = bool(np.array_equal(truth_ids, cpp_ids))
    expected_cpp_scores = scores_for_result_ids(vector_ids, vectors, queries, cpp_ids)
    max_score_error = float(np.max(np.abs(expected_cpp_scores - cpp_scores)))
    summary = {
        "seed": args.seed,
        "vector_count": args.vector_count,
        "query_count": args.query_count,
        "dimension": args.dimension,
        "k": args.k,
        "kernel": args.kernel,
        "index": args.index,
        "m": args.m if args.index == "hnsw" else None,
        "ef_construction": args.ef_construction if args.index == "hnsw" else None,
        "ef_search": args.ef_search if args.index == "hnsw" else None,
        "hnsw_seed": args.hnsw_seed if args.index == "hnsw" else None,
        "neighbor_selection": args.neighbor_selection if args.index == "hnsw" else None,
        "mean_recall_at_k": mean_recall,
        "minimum_recall_at_k": float(np.min(per_query_recall)),
        "exact_ranked_id_match": exact_id_match,
        "max_absolute_score_error": max_score_error,
    }
    (args.output_dir / "summary.json").write_text(
        json.dumps(summary, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(summary, indent=2))

    if args.index == "flat":
        if mean_recall != 1.0 or np.min(per_query_recall) != 1.0:
            raise RuntimeError("FlatIndex did not reproduce the exact NumPy Top-K IDs")
        if not exact_id_match:
            raise RuntimeError("FlatIndex Top-K order differs from the NumPy ground truth")
        if max_score_error > 1.0e-5:
            raise RuntimeError("C++ scores differ from NumPy by more than 1e-5")
    if args.minimum_recall is not None and mean_recall < args.minimum_recall:
        raise RuntimeError(
            f"mean Recall@K {mean_recall:.6f} is below {args.minimum_recall:.6f}"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
