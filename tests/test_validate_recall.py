import csv
import tempfile
import unittest
from pathlib import Path

import numpy as np

from scripts.validate_recall import exact_top_k, read_cpp_results


class GroundTruthTest(unittest.TestCase):
    def test_exact_top_k_uses_ascending_id_to_break_ties(self) -> None:
        vector_ids = np.array([20, 10, 30], dtype=np.uint64)
        vectors = np.array([[1.0, -1.0], [1.0, 1.0], [0.0, 1.0]], dtype=np.float32)
        queries = np.array([[1.0, 0.0]], dtype=np.float32)

        result_ids, _ = exact_top_k(vector_ids, vectors, queries, 3)

        np.testing.assert_array_equal(result_ids[0], np.array([10, 20, 30]))

    def test_cpp_result_reader_rejects_duplicate_rank(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "results.csv"
            with path.open("w", newline="", encoding="utf-8") as output:
                writer = csv.writer(output)
                writer.writerow(["query_id", "rank", "vector_id", "score"])
                writer.writerow([0, 0, 10, 1.0])
                writer.writerow([0, 0, 20, 0.5])

            with self.assertRaisesRegex(ValueError, "duplicate query rank"):
                read_cpp_results(path, np.array([0], dtype=np.uint64), 2)


if __name__ == "__main__":
    unittest.main()
