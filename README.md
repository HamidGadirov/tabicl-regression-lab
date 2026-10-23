# TabICL Regression Lab

A compact research and education project for predicting continuous targets from
tabular data with **TabICLv2**. Application code by **Hamid Gadirov**.

The example combines a reproducible evaluation split, repeated inference with
cached context, and held-out interval calibration. It runs on synthetic data by
default, with an optional numeric CSV input and a lightweight Extra Trees baseline.

The foundation model is provided by the
[TabICL authors](https://github.com/soda-inria/tabicl). This repository implements
an application workflow around its public Python API; it does not introduce a new
foundation model or claim a new calibration method.

## Quick start

Use Python 3.10 or newer; Python 3.12 is used for automated checks.

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -e '.[model]'
python -m tabicl_lab --device cpu
```

The first model run downloads the upstream regression checkpoint. CUDA users
should install the appropriate [PyTorch build](https://pytorch.org/get-started/locally/)
for their hardware before installing the model extra, then run:

```bash
python -m tabicl_lab --device cuda --cache repr
```

For a quick CPU-only workflow without downloading a foundation model:

```bash
python -m pip install -e .
python -m tabicl_lab --backend extra-trees
```

Outputs are written to `outputs/demo/report.json` and `outputs/demo/predictions.csv`.
Generated outputs and local datasets are ignored by Git. Runs using the same
output directory overwrite earlier results; use separate directories for comparisons.

## What the workflow does

1. Generate a noisy nonlinear Friedman regression dataset, or read a local CSV.
2. Randomly split rows into training/context (60%), calibration (20%), and test
   (20%) sets using a fixed seed.
3. Fit median imputation on training rows only. Fit the selected estimator on
   those training rows and labels.
4. Predict interval endpoints on calibration rows and compute a conformal
   expansion without using test labels.
5. Predict test rows in chunks and evaluate point accuracy, interval coverage,
   interval width, and repeated prediction latency.

For TabICL, `fit()` prepares transformations and context and optionally builds a
cache; the pretrained neural weights are not optimized on this dataset. Extra
Trees uses ordinary supervised training. Calibration rows are not added to either
model's fitting data.

## Inference controls

| Option | Meaning |
| --- | --- |
| `--point median` / `mean` | Select the upstream point output; both are supported |
| `--cache none` | Reprocess context during predictions |
| `--cache repr` | Cache context representations with a lower memory cost than full KV caching |
| `--cache kv` | Cache full key/value projections for repeated inference |
| `--estimators 2` | Number of TabICL ensemble members; increase and evaluate the tradeoff |
| `--ensemble-batch-size 2` | Ensemble members processed together inside TabICL |
| `--chunk-size 128` | Query rows sent to each `predict()` call |
| `--coverage 0.8` | Target marginal interval coverage; requests raw 0.1/0.9 endpoints |
| `--repeats 3` | Repeated predictions of the whole test set for timing |

Cache, estimator, device, thread, and ensemble batch settings apply to TabICL. The
baseline uses 128 trees on one CPU thread. Chunking applies to both backends.
Larger caches or batches can improve throughput while increasing memory use.
Changing the context requires fitting again and rebuilding any cache.

Compare cache modes on identical data splits:

```bash
for cache in none repr kv; do
  python -m tabicl_lab --device cpu --cache "$cache" --repeats 10 \
    --output "outputs/cache-$cache"
done
```

Setup timing includes model loading and context preparation, and can include a
first checkpoint download. Test timing covers all query chunks and both point
and interval outputs, after calibration inference has warmed the model. It excludes
CSV reading, imputation, and output writing. GPU work is synchronized before and
after each timed pass. P50/P95 values summarize repeated whole-test-set runs;
they are not per-row latency or a concurrent-serving benchmark. Three repeats are
only a demonstration; use more for a meaningful timing comparison.

## Raw and calibrated intervals

TabICL supplies estimated conditional quantiles. Their nominal probability levels
alone do not guarantee empirical coverage. For the baseline, raw endpoints are
quantiles of individual tree predictions: they measure ensemble spread and need
not represent the distribution of target noise.

This example applies an expansion-only variant of split-conformal quantile
calibration. For each calibration label `y` with endpoints `L` and `U`:

```text
score = max(L - y, y - U, 0)
k = ceil((n_calibration + 1) * coverage)
offset = kth smallest calibration score
final interval = [L - offset, U + offset]
```

It rejects coverage settings for which there are too few calibration rows to
select a finite order statistic. The zero clips negative scores, so calibration
expands or preserves the raw intervals. Under exchangeability of calibration
and future observations, split conformal provides a marginal coverage guarantee;
it does not guarantee coverage for each subgroup or each particular input.
Measured coverage on a finite test set can differ from its target.

The included random split is appropriate for the synthetic independent samples.
For time-dependent or grouped data, replace it with suitable disjoint splits and
reconsider the exchangeability assumption. Keep tuning separate from the final
calibration and test sets. See
[Conformalized Quantile Regression](https://arxiv.org/abs/1905.03222) for the method
underlying this calibration approach.

## Local CSVs

```bash
python -m tabicl_lab --csv data/table.csv --target target \
  --device cpu --output outputs/local-experiment
```

Use at least 100 rows, unique column names, numeric features, and one finite
numeric target. Missing feature values are imputed from training medians; a
feature that is entirely missing in training is filled with zero. Remove row
identifiers, outcome-derived fields, and other sources of target leakage before
running. Categorical encoding and time/group splits are deliberately left to
dataset-specific preprocessing.

The report records versions, settings, split sizes, metrics, and timings. The
prediction CSV contains test row positions, labels, and predictions. Keep these
local when working with non-public datasets. No raw data, checkpoint, fitted
model, or cache is bundled in this repository.

## Checks

```bash
python -m unittest discover -s tests -v
```

CPU CI checks splitting, the finite-sample calibration rank, CSV validation,
query ordering, output validation, and a complete baseline run. It uses a small
test double to check the TabICL adapter without installing PyTorch.

After installing the model extra, run the optional actual-model check:

```bash
RUN_TABICL_SMOKE=1 python -m unittest discover -s tests -v
```

`VALIDATION.md` records the checks performed for this release.

## Files and attribution

| File | Purpose |
| --- | --- |
| `tabicl_lab/experiment.py` | Dataset loading, splits, inference, calibration, evaluation, CLI |
| `tests/` | Workflow checks and optional actual-model smoke test |
| `.github/workflows/checks.yml` | CPU checks on pushes and pull requests |

The application code is MIT licensed. TabICL is installed separately and retains
its own [BSD 3-Clause license](https://github.com/soda-inria/tabicl/blob/main/LICENSE)
and authorship. No upstream implementation or weights are vendored here.
The model dependency is pinned to `tabicl==2.2.0`; other dependencies have bounded
version ranges rather than a universal hardware-specific lock file. Record the
reported versions when reproducing an experiment.

References: [TabICL repository](https://github.com/soda-inria/tabicl),
[public API documentation](https://tabicl.readthedocs.io/en/latest/api.html), and
[Romano, Patterson & Candès (2019)](https://arxiv.org/abs/1905.03222).
