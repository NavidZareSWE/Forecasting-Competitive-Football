"""Main sweep: every model on every task, with the tuned configurations.

Writes model_results.csv and predictions_<task>.csv; the latter feed
market_comparison.py, inplay_curves.py and the diagnostic plots.
"""

import pandas as pd

from modeling_common import task_frame, RESULTS_DIR
from model_zoo import classifier_zoo, regressor_zoo
from train import evaluate_classification, evaluate_regression
from tuning import load_best_params


TASK_LABELS = {
    "C": "Task C - pre-match outcome",
    "R": "Task R - pre-match margin",
    "Lc": "Task L - in-play outcome",
    "Lr": "Task L - in-play margin",
}

ZOO_TASK = {"C": "C", "R": "R", "Lc": "L", "Lr": "L"}

# Pre-match table only; the six-arm comparison lives in resampling_study.py.
P1_ELIGIBLE_TASKS = {"C"}


def run_classification_task(task, random_state=0, resampling="none"):
    df, continuous, nominal, target, task_type = task_frame(task)
    data = (df, continuous, nominal, target, task_type)
    if resampling != "none" and task not in P1_ELIGIBLE_TASKS:
        raise ValueError(f"Resampling arm {resampling!r} is not permitted on "
                         f"task {task}: the snapshot table must not be "
                         f"oversampled across matches.")
    tuned = load_best_params().get(task, {})
    rows, frames = [], []
    for name, factory in classifier_zoo(random_state, task=ZOO_TASK[task],
                                        tuned=tuned).items():
        result, predictions = evaluate_classification(name, factory, data,
                                                      resampling)
        result["task"] = task
        result["tuned"] = name in tuned
        rows.append(result)
        frames.append(predictions)
        print(f"  [{task}] {name:18s} rps={result['rps']:.5f} "
              f"ece={result['ece_before']:.4f}->{result['ece_after']:.4f} "
              f"cal={result['calibration']} {result['train_seconds']:.1f}s")
    return rows, pd.concat(frames, ignore_index=True)


def run_regression_task(task, random_state=0):
    df, continuous, nominal, target, task_type = task_frame(task)
    data = (df, continuous, nominal, target, task_type)
    tuned = load_best_params().get(task, {})
    rows, frames = [], []
    for name, factory in regressor_zoo(random_state, task=ZOO_TASK[task],
                                       tuned=tuned).items():
        result, predictions = evaluate_regression(name, factory, data)
        result["task"] = task
        result["tuned"] = name in tuned
        rows.append(result)
        frames.append(predictions)
        print(f"  [{task}] {name:18s} mae={result['mae']:.5f} "
              f"rmse={result['rmse']:.5f} corr={result['corr']:.4f} "
              f"{result['train_seconds']:.1f}s")
    return rows, pd.concat(frames, ignore_index=True)


PER_CLASS_COLUMNS = [f"{metric}_{label}"
                     for label in ["H", "D", "A"]
                     for metric in ["precision", "recall", "f1", "support"]]

COLUMN_ORDER = ["task", "model", "resampling", "tuned", "calibration",
                "hs_lambda", "train_seconds", "peak_memory_mb", "n_train",
                "n_test", "subsampled", "rps", "rps_before", "log_loss",
                "brier", "ece_before", "ece_after", *PER_CLASS_COLUMNS,
                "mae", "rmse", "corr"]


def write_results(rows, filename):
    results = pd.DataFrame(rows)
    results = results.reindex(
        columns=[c for c in COLUMN_ORDER if c in results.columns])
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    output_path = RESULTS_DIR / filename
    results.to_csv(output_path, index=False, encoding="utf-8")
    print(f"\nWrote {len(results)} model rows -> {output_path}")
    return output_path


def write_predictions(frame, task):
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    output_path = RESULTS_DIR / f"predictions_{task}.csv"
    frame.to_csv(output_path, index=False, encoding="utf-8")
    print(f"  wrote {len(frame)} prediction rows -> {output_path.name}")
    return output_path


def main():
    all_rows = []
    for task in ["C", "Lc"]:
        print(f"\n=== {TASK_LABELS[task]} ===")
        rows, predictions = run_classification_task(task)
        all_rows.extend(rows)
        write_predictions(predictions, task)
    for task in ["R", "Lr"]:
        print(f"\n=== {TASK_LABELS[task]} ===")
        rows, predictions = run_regression_task(task)
        all_rows.extend(rows)
        write_predictions(predictions, task)

    write_results(all_rows, "model_results.csv")


if __name__ == "__main__":
    main()
