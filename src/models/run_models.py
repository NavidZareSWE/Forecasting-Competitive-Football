"""Run from the repository root with:

    python src/models/run_models.py
"""

from pathlib import Path
import os
import sys

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))

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

P1_ELIGIBLE_TASKS = {"C"}


def run_classification_task(task, random_state=0, resampling="none"):
    df, continuous, nominal, target, task_type = task_frame(task)
    data = (df, continuous, nominal, target, task_type)
    if resampling != "none" and task not in P1_ELIGIBLE_TASKS:
        raise ValueError(f"Resampling arm {resampling!r} is not permitted on "
                         f"task {task}: the snapshot table must not be "
                         f"oversampled across matches.")
    tuned = load_best_params().get(task, {})
    rows, frames, references = [], [], []
    for name, factory in classifier_zoo(random_state, task=ZOO_TASK[task],
                                        tuned=tuned).items():
        result, predictions, reference = evaluate_classification(
            name, factory, data, resampling, with_reference=True)
        result["task"] = task
        result["tuned"] = name in tuned
        rows.append(result)
        frames.append(predictions)
        if reference is not None:
            references.append(reference)
        print(f"  [{task}] {name:18s} rps={result['rps']:.5f} "
              f"ece={result['ece_before']:.4f}->{result['ece_after']:.4f} "
              f"cal={result['calibration']} {result['train_seconds']:.1f}s")
    reference = (pd.concat(references, ignore_index=True)
                 if references else None)
    return rows, pd.concat(frames, ignore_index=True), reference


def run_regression_task(task, random_state=0):
    df, continuous, nominal, target, task_type = task_frame(task)
    data = (df, continuous, nominal, target, task_type)
    tuned = load_best_params().get(task, {})
    rows, frames, references = [], [], []
    for name, factory in regressor_zoo(random_state, task=ZOO_TASK[task],
                                       tuned=tuned).items():
        result, predictions, reference = evaluate_regression(
            name, factory, data, with_reference=True)
        result["task"] = task
        result["tuned"] = name in tuned
        rows.append(result)
        frames.append(predictions)
        if reference is not None:
            references.append(reference)
        print(f"  [{task}] {name:18s} mae={result['mae']:.5f} "
              f"rmse={result['rmse']:.5f} corr={result['corr']:.4f} "
              f"{result['train_seconds']:.1f}s")
    reference = (pd.concat(references, ignore_index=True)
                 if references else None)
    return rows, pd.concat(frames, ignore_index=True), reference


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


def write_reference(frame, task):
    """Pre-match predictions on the excluded split.

    Kept in its own file so that everything reading
    predictions_{task}.csv keeps seeing test rows only, and only
    inplay_curves.py has to know these exist.
    """
    if frame is None or not len(frame):
        return None
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    output_path = RESULTS_DIR / f"predictions_ref_{task}.csv"
    frame.to_csv(output_path, index=False, encoding="utf-8")
    print(f"  wrote {len(frame)} frozen-reference rows "
          f"({frame['match_id'].nunique()} matches) -> "
          f"{output_path.name}")
    return output_path


def main():
    selected = [t for t in os.environ.get("TASKS", "C,Lc,R,Lr").split(",")
                if t.strip()]
    unknown = [t for t in selected if t not in TASK_LABELS]
    assert not unknown, f"unknown task(s) {unknown}; expected {list(TASK_LABELS)}"
    if selected != ["C", "Lc", "R", "Lr"]:
        print(f"Running a subset of tasks: {selected}")

    all_rows = []
    for task in [t for t in ["C", "Lc"] if t in selected]:
        print(f"\n=== {TASK_LABELS[task]} ===")
        rows, predictions, reference = run_classification_task(task)
        all_rows.extend(rows)
        write_predictions(predictions, task)
        write_reference(reference, task)
    for task in [t for t in ["R", "Lr"] if t in selected]:
        print(f"\n=== {TASK_LABELS[task]} ===")
        rows, predictions, reference = run_regression_task(task)
        all_rows.extend(rows)
        write_predictions(predictions, task)
        write_reference(reference, task)

    existing_path = RESULTS_DIR / "model_results.csv"
    if existing_path.exists() and len(selected) < len(TASK_LABELS):
        previous = pd.read_csv(existing_path, encoding="utf-8")
        carried = previous[~previous["task"].isin(selected)]
        if len(carried):
            carried = carried.copy()
            carried["carried_from_previous_run"] = True
            print(f"WARNING: carrying forward {len(carried)} rows for tasks "
                  f"{sorted(set(carried['task']))} from a previous run. Those "
                  f"rows were produced by whatever feature tables existed then "
                  f"and are NOT comparable with the tasks just run if the "
                  f"features have changed since. Re-run without TASKS before "
                  f"reporting any cross-task comparison.")
            all_rows = carried.to_dict("records") + all_rows
    for row in all_rows:
        row.setdefault("carried_from_previous_run", False)

    write_results(all_rows, "model_results.csv")


if __name__ == "__main__":
    main()
