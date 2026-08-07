"""Phase 7: compute and memory comparison across the model zoo.

Fit time, peak resident memory during the fit, inference latency on the full
test matrix, and serialised model size, measured on the same machine in one
process so the numbers are comparable. The kernel scaling exponents from
kernel_scaling.py are folded in where they exist.

    python src/models/compute_profile.py
"""

import pickle
import time

import numpy as np
import pandas as pd

from modeling_common import (RESULTS_DIR, fit_with_cost, prepare_matrices,
                             task_frame)
from model_zoo import classifier_zoo, regressor_zoo
from tuning import load_best_params


RANDOM_STATE = 0
PREDICT_REPEATS = 5
ZOO_TASK = {"C": "C", "R": "R", "Lc": "L", "Lr": "L"}
TASKS = ["C", "R", "Lc", "Lr"]


def model_size_mb(estimator):
    try:
        return len(pickle.dumps(estimator)) / 1024 ** 2
    except Exception:
        return float("nan")


def predict_seconds(estimator, X, is_classification):
    call = estimator.predict_proba if is_classification else estimator.predict
    call(X[:1])                                     # warm up lazy allocations
    timings = []
    for _ in range(PREDICT_REPEATS):
        started = time.perf_counter()
        call(X)
        timings.append(time.perf_counter() - started)
    return float(np.median(timings))


def profile_task(task, rows):
    print(f"\n=== compute profile: task {task} ===")
    is_classification = task in {"C", "Lc"}
    df, continuous, nominal, target, task_type = task_frame(task)
    matrices = prepare_matrices(df, continuous, nominal, target, task_type)
    tuned = load_best_params().get(task, {})
    zoo = (classifier_zoo(RANDOM_STATE, task=ZOO_TASK[task], tuned=tuned)
           if is_classification
           else regressor_zoo(RANDOM_STATE, task=ZOO_TASK[task], tuned=tuned))

    X_train, X_test = matrices["X_train"], matrices["X_test"]
    y_train = matrices["y_train"]
    for name, factory in zoo.items():
        estimator = factory()
        seconds, peak_mb = fit_with_cost(
            estimator, X_train,
            y_train if is_classification else y_train.astype(float))
        latency = predict_seconds(estimator, X_test, is_classification)
        used = int(getattr(estimator, "n_train_used_", X_train.shape[0]))
        rows.append({
            "task": task, "model": name,
            "n_train_available": int(X_train.shape[0]),
            "n_train_used": used,
            "n_features": int(X_train.shape[1]),
            "fit_seconds": round(seconds, 4),
            "peak_fit_memory_mb": round(peak_mb, 1),
            "model_size_mb": round(model_size_mb(estimator), 3),
            "predict_seconds_full_test": round(latency, 5),
            "predict_microseconds_per_row": round(
                latency / X_test.shape[0] * 1e6, 2),
            "subsampled_for_fit": used < X_train.shape[0]})
        print(f"  {name:22s} fit {seconds:8.3f}s  peak {peak_mb:7.1f} MB  "
              f"predict {latency * 1e3:7.2f} ms  size "
              f"{rows[-1]['model_size_mb']:7.2f} MB")


def main():
    rows = []
    for task in TASKS:
        profile_task(task, rows)
    profile = pd.DataFrame(rows)

    scaling_path = RESULTS_DIR / "kernel_scaling.csv"
    if scaling_path.exists():
        scaling = pd.read_csv(scaling_path, encoding="utf-8")
        exponents = (scaling.drop_duplicates("method")
                     [["method", "empirical_exponent", "theoretical_exponent",
                       "theory"]]
                     .rename(columns={"method": "model"}))
        profile = profile.merge(exponents, on="model", how="left")

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    output_path = RESULTS_DIR / "compute_profile.csv"
    profile.to_csv(output_path, index=False, encoding="utf-8")

    print("\nCompute and memory comparison:")
    print(profile[["task", "model", "n_train_used", "fit_seconds",
                   "peak_fit_memory_mb", "model_size_mb",
                   "predict_microseconds_per_row"]].to_string(index=False))
    print(f"\nWrote -> {output_path}")


if __name__ == "__main__":
    main()
