# Validation

Checked on 23 October 2026, Python 3.12, Linux CPU.

- Seven workflow tests passed, covering disjoint splits, finite-sample calibration,
  small calibration sets, CSV input, query ordering, invalid output, and the full
  baseline workflow.
- The optional eighth test also passed using the actual TabICL regressor,
  the public regression checkpoint, representation caching, and chunked queries.
- The default 600-row synthetic experiment completed with both backends,
  including writing reports and predictions.
- A wheel was built and installed successfully.
- The release archive was checked for private names, internal network addresses,
  local machine paths, common credential patterns, and unwanted binary/data files.
  It contains newly written application code and documentation, with no repository
  history, bundled datasets, model parameters, or cached context.

Observed point and interval metrics on synthetic Friedman data, seed 42:

| Backend | MAE | RMSE | R² | Raw coverage | Calibrated coverage |
| --- | ---: | ---: | ---: | ---: | ---: |
| TabICL, 2 estimators, representation cache | 0.910 | 1.168 | 0.946 | 79.2% | 86.7% |
| Extra Trees, 128 trees | 1.902 | 2.387 | 0.776 | 81.7% | 81.7% |

Both runs used 360 fitting rows, 120 calibration rows, and 120 test rows, with an
80% coverage target and median point output. These numbers demonstrate a working
workflow on one synthetic dataset. They are not evidence of general superiority,
expected real-data performance, or exact coverage on other datasets.

Environment: NumPy 2.3.5, pandas 2.2.3, scikit-learn 1.8.0,
TabICL 2.2.0, PyTorch 2.14.1+cpu. GPU execution was not tested for this release.
Cache modes other than representation caching are exposed through the verified
public API but were not exercised in the actual-model checks for this release.
