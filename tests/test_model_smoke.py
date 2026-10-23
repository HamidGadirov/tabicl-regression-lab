"""Optional real-model check; downloads the upstream checkpoint on first use."""
import os
import unittest

import numpy as np
from sklearn.datasets import make_friedman1

from tabicl_lab.experiment import predict_blocks


@unittest.skipUnless(os.environ.get("RUN_TABICL_SMOKE") == "1", "Set RUN_TABICL_SMOKE=1 to download and run TabICL.")
class RealModelTests(unittest.TestCase):
    def test_cpu_regression_with_representation_cache(self):
        from tabicl import TabICLRegressor

        X, y = make_friedman1(n_samples=90, n_features=5, random_state=42)
        model = TabICLRegressor(n_estimators=2, batch_size=2, kv_cache="repr", device="cpu", n_jobs=2)
        model.fit(X[:80], y[:80])
        point, bounds = predict_blocks(model, X[80:], "tabicl", 5, 0.8, "median")
        self.assertEqual(point.shape, (10,))
        self.assertEqual(bounds.shape, (10, 2))
        self.assertTrue(np.isfinite(point).all())
        self.assertTrue(np.all(bounds[:, 0] <= bounds[:, 1]))


if __name__ == "__main__":
    unittest.main()
