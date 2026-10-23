"""Reproducible regression, context caching, and held-out interval calibration."""
from __future__ import annotations

import argparse
import csv as csv_module
from importlib.metadata import PackageNotFoundError, version
import json
import math
from pathlib import Path
import sys
from time import perf_counter

import numpy as np
import pandas as pd
from sklearn.datasets import make_friedman1
from sklearn.ensemble import ExtraTreesRegressor
from sklearn.impute import SimpleImputer
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score


def load_data(csv: Path | None, target: str, rows: int, seed: int):
    """Accept numeric features and missing feature values; reject missing labels."""
    if csv is None:
        X, y = make_friedman1(n_samples=rows, n_features=10, noise=1.0, random_state=seed)
    else:
        with csv.open(newline="") as stream:
            header = next(csv_module.reader(stream), [])
        if len(header) != len(set(header)):
            raise ValueError("The CSV must have unique column names.")
        frame = pd.read_csv(csv)
        if target not in frame or frame.columns.has_duplicates:
            raise ValueError("The CSV must contain one target column and unique column names.")
        y = pd.to_numeric(frame.pop(target), errors="raise").to_numpy(dtype=float)
        X = frame.apply(pd.to_numeric, errors="raise").to_numpy(dtype=float)
    if X.ndim != 2 or X.shape[1] == 0 or len(y) < 100:
        raise ValueError("Use at least 100 rows and one numeric feature.")
    if not np.isfinite(y).all() or np.isinf(X).any():
        raise ValueError("Labels must be finite; features may contain NaN, but not infinity.")
    return X, y


def split_indices(n: int, seed: int):
    """Disjoint random 60/20/20 splits; labels do not affect the split."""
    order = np.random.default_rng(seed).permutation(n)
    held_out = n // 5
    return order[: n - 2 * held_out], order[n - 2 * held_out : n - held_out], order[n - held_out :]


def calibration_offset(y, bounds, coverage: float) -> float:
    """Finite-sample split-conformal rank, with nonnegative interval expansion."""
    y, bounds = np.asarray(y), np.asarray(bounds)
    if not 0 < coverage < 1 or bounds.shape != (len(y), 2):
        raise ValueError("Invalid coverage or calibration interval shape.")
    if not np.isfinite(y).all() or not np.isfinite(bounds).all():
        raise ValueError("Calibration inputs must be finite.")
    if np.any(bounds[:, 0] > bounds[:, 1]):
        raise ValueError("Calibration lower bounds exceed upper bounds.")
    rank = math.ceil((len(y) + 1) * coverage)
    if rank > len(y) or rank < 1:
        raise ValueError("Too few calibration rows for this coverage; use more data or lower coverage.")
    scores = np.maximum.reduce([bounds[:, 0] - y, y - bounds[:, 1], np.zeros(len(y))])
    return float(np.partition(scores, rank - 1)[rank - 1])


def make_model(args):
    if args.backend == "extra-trees":
        return ExtraTreesRegressor(n_estimators=128, min_samples_leaf=3, random_state=args.seed, n_jobs=1)
    try:
        from tabicl import TabICLRegressor
    except ImportError as error:
        raise RuntimeError("Install the model extra: python -m pip install -e '.[model]'") from error
    return TabICLRegressor(
        n_estimators=args.estimators, batch_size=args.ensemble_batch_size,
        kv_cache=False if args.cache == "none" else args.cache,
        device=args.device, random_state=args.seed, n_jobs=args.threads,
    )


def predict_blocks(model, X, backend: str, chunk_size: int, coverage: float, point: str):
    """Chunk query rows independently from TabICL's ensemble batch size."""
    if len(X) == 0 or chunk_size < 1:
        raise ValueError("Prediction needs nonempty rows and a positive chunk size.")
    tail = (1 - coverage) / 2
    points, intervals = [], []
    for start in range(0, len(X), chunk_size):
        block = X[start : start + chunk_size]
        if backend == "tabicl":
            output = model.predict(block, output_type=[point, "quantiles"], alphas=[tail, 1 - tail])
            prediction, bounds = np.asarray(output[point]), np.asarray(output["quantiles"])
        else:
            tree_predictions = np.stack([tree.predict(block) for tree in model.estimators_])
            prediction = (np.mean if point == "mean" else np.median)(tree_predictions, axis=0)
            bounds = np.quantile(tree_predictions, [tail, 1 - tail], axis=0).T
        if prediction.shape != (len(block),) or bounds.shape != (len(block), 2):
            raise ValueError("Unexpected prediction output shape.")
        if not np.isfinite(prediction).all() or not np.isfinite(bounds).all():
            raise ValueError("Model returned nonfinite predictions.")
        if np.any(bounds[:, 0] > bounds[:, 1]):
            raise ValueError("Model returned crossing interval bounds.")
        points.append(prediction)
        intervals.append(bounds)
    return np.concatenate(points), np.concatenate(intervals)


def synchronize(model):
    """Include asynchronous CUDA work in reported wall-clock durations."""
    torch = sys.modules.get("torch")
    device = getattr(model, "device_", None)
    if torch is not None and str(device).startswith("cuda"):
        torch.cuda.synchronize(device)
    elif torch is not None and str(device).startswith("mps"):
        torch.mps.synchronize()


def installed_version(name: str):
    try:
        return version(name)
    except PackageNotFoundError:
        return None


def run(args):
    X, y = load_data(args.csv, args.target, args.rows, args.seed)
    train, calibration, test = split_indices(len(y), args.seed)
    # Learn imputation statistics from training rows only.
    imputer = SimpleImputer(strategy="median", keep_empty_features=True)
    X_train = imputer.fit_transform(X[train])
    X_cal, X_test = imputer.transform(X[calibration]), imputer.transform(X[test])
    model = make_model(args)
    started = perf_counter()
    model.fit(X_train, y[train])
    synchronize(model)
    setup_seconds = perf_counter() - started
    _, cal_bounds = predict_blocks(model, X_cal, args.backend, args.chunk_size, args.coverage, args.point)
    offset = calibration_offset(y[calibration], cal_bounds, args.coverage)
    durations = []
    for _ in range(args.repeats):
        synchronize(model)
        started = perf_counter()
        prediction, raw = predict_blocks(model, X_test, args.backend, args.chunk_size, args.coverage, args.point)
        synchronize(model)
        durations.append(perf_counter() - started)
    lower, upper = raw[:, 0] - offset, raw[:, 1] + offset
    p50 = float(np.quantile(durations, 0.5))
    report = {
        "dataset": "synthetic_friedman1" if args.csv is None else "user_csv",
        "backend": args.backend,
        "configuration": {
            "seed": args.seed, "features": int(X.shape[1]), "point": args.point,
            "coverage_target": args.coverage, "query_chunk_size": args.chunk_size,
            "cache": args.cache if args.backend == "tabicl" else None,
            "estimators": args.estimators if args.backend == "tabicl" else 128,
            "ensemble_batch_size": args.ensemble_batch_size if args.backend == "tabicl" else None,
            "device": str(getattr(model, "device_", "cpu")),
            "threads": args.threads if args.backend == "tabicl" else 1,
        },
        "split_rows": {"train": len(train), "calibration": len(calibration), "test": len(test)},
        "metrics": {
            "mae": float(mean_absolute_error(y[test], prediction)),
            "rmse": float(np.sqrt(mean_squared_error(y[test], prediction))),
            "r2": float(r2_score(y[test], prediction)),
            "raw_interval_coverage": float(np.mean((y[test] >= raw[:, 0]) & (y[test] <= raw[:, 1]))),
            "calibrated_interval_coverage": float(np.mean((y[test] >= lower) & (y[test] <= upper))),
            "mean_interval_width": float(np.mean(upper - lower)),
            "calibration_expansion": offset,
        },
        "timing": {
            "setup_seconds": setup_seconds,
            "test_batch_seconds_p50": p50,
            "test_batch_seconds_p95": float(np.quantile(durations, 0.95)),
            "test_rows_per_second_p50": len(test) / p50,
            "repeats": args.repeats,
        },
        "versions": {name: installed_version(name) for name in ("numpy", "pandas", "scikit-learn", "tabicl", "torch")},
    }
    args.output.mkdir(parents=True, exist_ok=True)
    (args.output / "report.json").write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")
    pd.DataFrame({"row_index": test, "observed": y[test], "prediction": prediction,
                  "raw_lower": raw[:, 0], "raw_upper": raw[:, 1], "lower": lower, "upper": upper}).to_csv(
        args.output / "predictions.csv", index=False
    )
    return report


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--backend", choices=["tabicl", "extra-trees"], default="tabicl")
    parser.add_argument("--csv", type=Path, help="Optional local CSV with numeric features and target.")
    parser.add_argument("--target", default="target")
    parser.add_argument("--rows", type=int, default=600, help="Synthetic dataset size.")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", choices=["cpu", "cuda", "mps"], default="cpu")
    parser.add_argument("--threads", type=int, default=2)
    parser.add_argument("--cache", choices=["none", "repr", "kv"], default="repr")
    parser.add_argument("--estimators", type=int, default=2)
    parser.add_argument("--ensemble-batch-size", type=int, default=2)
    parser.add_argument("--chunk-size", type=int, default=128)
    parser.add_argument("--point", choices=["mean", "median"], default="median")
    parser.add_argument("--coverage", type=float, default=0.8)
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--output", type=Path, default=Path("outputs/demo"))
    args = parser.parse_args(argv)
    if not 0 < args.coverage < 1:
        parser.error("--coverage must be between zero and one.")
    if min(args.rows, args.threads, args.estimators, args.ensemble_batch_size, args.chunk_size, args.repeats) < 1:
        parser.error("Size, thread, estimator, batch, and repeat counts must be positive.")
    return args


def main():
    args = parse_args()
    try:
        report = run(args)
    except (ValueError, RuntimeError) as error:
        raise SystemExit(str(error)) from error
    print(json.dumps(report, indent=2, allow_nan=False))


if __name__ == "__main__":
    main()
