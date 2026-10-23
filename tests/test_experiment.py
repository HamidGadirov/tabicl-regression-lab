"""CPU checks for splitting, calibration, query batching, and the complete workflow."""
import json
from pathlib import Path
import tempfile
import unittest

import numpy as np

from tabicl_lab.experiment import calibration_offset, load_data, parse_args, predict_blocks, run, split_indices


class WorkflowTests(unittest.TestCase):
    def test_disjoint_reproducible_splits(self):
        first = split_indices(101, 42)
        second = split_indices(101, 42)
        self.assertEqual([len(x) for x in first], [61, 20, 20])
        np.testing.assert_array_equal(np.sort(np.concatenate(first)), np.arange(101))
        for left, right in zip(first, second):
            np.testing.assert_array_equal(left, right)

    def test_conformal_finite_sample_rank(self):
        # n=9, coverage=.8 => ceil(10*.8)=8; residual order statistics are 1,...,9.
        offset = calibration_offset(np.arange(1, 10), np.zeros((9, 2)), 0.8)
        self.assertEqual(offset, 8.0)

    def test_calibration_never_shrinks_and_rejects_too_few_rows(self):
        bounds = np.tile([-10.0, 10.0], (9, 1))
        self.assertEqual(calibration_offset(np.zeros(9), bounds, 0.8), 0.0)
        with self.assertRaisesRegex(ValueError, "Too few"):
            calibration_offset(np.zeros(9), bounds, 0.99)

    def test_query_chunks_preserve_order_and_public_api_contract(self):
        class Predictor:
            def __init__(self):
                self.sizes = []

            def predict(self, X, output_type, alphas):
                self.sizes.append(len(X))
                self_output = {"median": X[:, 0], "quantiles": np.column_stack([X[:, 0] - 1, X[:, 0] + 1])}
                np.testing.assert_allclose(alphas, [0.1, 0.9])
                assert output_type == ["median", "quantiles"]
                return self_output

        model = Predictor()
        point, bounds = predict_blocks(model, np.arange(11)[:, None], "tabicl", 4, 0.8, "median")
        self.assertEqual(model.sizes, [4, 4, 3])
        np.testing.assert_array_equal(point, np.arange(11))
        np.testing.assert_array_equal(bounds[:, 0], np.arange(11) - 1)

    def test_invalid_model_output_is_rejected(self):
        class Predictor:
            def predict(self, X, **kwargs):
                return {"median": np.zeros(len(X)), "quantiles": np.tile([2.0, 1.0], (len(X), 1))}

        with self.assertRaisesRegex(ValueError, "crossing"):
            predict_blocks(Predictor(), np.ones((2, 1)), "tabicl", 4, 0.8, "median")

    def test_csv_validation(self):
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / "input.csv"
            path.write_text("feature,target\n" + "1,2\n" * 100)
            X, y = load_data(path, "target", 100, 42)
            self.assertEqual(X.shape, (100, 1))
            self.assertEqual(y.shape, (100,))
            path.write_text("feature,target,target\n" + "1,2,2\n" * 100)
            with self.assertRaisesRegex(ValueError, "unique"):
                load_data(path, "target", 100, 42)
            path.write_text("feature,target\n" + "1,\n" * 100)
            with self.assertRaisesRegex(ValueError, "finite"):
                load_data(path, "target", 100, 42)

    def test_full_cpu_workflow_and_saved_outputs(self):
        with tempfile.TemporaryDirectory() as folder:
            args = parse_args(["--backend", "extra-trees", "--rows", "150", "--repeats", "1", "--output", folder])
            report = run(args)
            self.assertEqual(report["split_rows"], {"train": 90, "calibration": 30, "test": 30})
            self.assertTrue(np.isfinite(report["metrics"]["mae"]))
            self.assertGreaterEqual(report["metrics"]["calibration_expansion"], 0)
            self.assertGreaterEqual(report["metrics"]["calibrated_interval_coverage"], report["metrics"]["raw_interval_coverage"])
            saved = json.loads((Path(folder) / "report.json").read_text())
            self.assertEqual(saved, report)
            self.assertEqual(len((Path(folder) / "predictions.csv").read_text().splitlines()), 31)


if __name__ == "__main__":
    unittest.main()
