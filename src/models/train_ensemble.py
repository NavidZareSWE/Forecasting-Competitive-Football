"""Evaluate the stacked ensemble and merge its rows into the sweep output.

    python src/models/train_ensemble.py
    TASKS=C,Lc python src/models/train_ensemble.py     # subset

Not folded into run_models.py because the in-play stack needs folds grouped by
``match_id``, which a zoo factory cannot receive - so the stack is not a zoo
member at all and no sweep loop picks it up.
Everything else is shared: the same ``task_frame`` tables, the same
``prepare_matrices`` splits, the same ``evaluate_classification`` /
``evaluate_regression`` used for every other model, so the appended rows are
directly comparable with the ones already in model_results.csv.

Two variants are measured for every task, differing only in where the
meta-learner's training signal comes from (see stacking._Stack._fit_bases):

    stack            out-of-fold over the training split
    stack_temporal   the earliest 60% of validation, which sits between train
                     and test in time

Outputs (upserted in place, existing stack rows replaced):
    reports/model_results.csv
    reports/predictions_{C,R,Lc,Lr}.csv
    reports/ensemble_comparison.csv   best stack vs. the best single model
"""
from pathlib import Path
import os
import sys

import pandas as pd

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

from modeling_common import (RESULTS_DIR, prepare_matrices,  # noqa: E402
                             task_frame)
from run_models import COLUMN_ORDER  # noqa: E402
from stacking import (STACK_MODELS, build_named_stack,  # noqa: E402
                      resolve_base_names, training_groups)
from train import evaluate_classification, evaluate_regression  # noqa: E402
from tuning import load_best_params  # noqa: E402

TASKS = ["C", "Lc", "R", "Lr"]
METRIC = {"C": "rps", "Lc": "rps", "R": "mae", "Lr": "mae"}


def evaluate(task, model_name):
    df, continuous, nominal, target, task_type = task_frame(task)
    data = (df, continuous, nominal, target, task_type)
    tuned = load_best_params().get(task, {})
    names = resolve_base_names(task, tuned)
    if model_name == "stack_temporal":
        how = "meta fitted on the earliest 60% of validation"
    else:
        how = ("meta fitted out-of-fold, folds grouped by match_id"
               if training_groups(df, task) is not None
               else "meta fitted out-of-fold, stratified folds")
    print(f"\n=== {task} / {model_name}: {len(names)} members {names}; "
          f"{how} ===")

    # prepare_matrices is deterministic on (df, split), so the preprocessor
    # built here is the one the evaluator rebuilds; the temporal variant needs
    # it up front to transform its meta-holdout block.
    transform = prepare_matrices(df, continuous, nominal, target,
                                 task_type)["transform"]

    def factory():
        return build_named_stack(model_name, task, tuned, df,
                                 transform=transform)

    evaluator = (evaluate_classification if task in {"C", "Lc"}
                 else evaluate_regression)
    row, predictions = evaluator(model_name, factory, data)
    row["task"] = task
    row["tuned"] = True
    row["carried_from_previous_run"] = False
    metric = METRIC[task]
    print(f"  [{task}] {model_name} {metric}={row[metric]:.5f} "
          f"({row['train_seconds']:.1f}s)")
    return row, predictions


def upsert_results(rows):
    path = RESULTS_DIR / "model_results.csv"
    previous = pd.read_csv(path, encoding="utf-8")
    tasks = {r["task"] for r in rows}
    keep = previous[~(previous["model"].isin(STACK_MODELS)
                      & (previous["task"].isin(tasks)))]
    merged = pd.concat([keep, pd.DataFrame(rows)], ignore_index=True)
    order = {t: i for i, t in enumerate(TASKS)}
    merged = merged.sort_values(
        "task", key=lambda s: s.map(order), kind="stable")
    merged = merged.reindex(
        columns=[c for c in COLUMN_ORDER + ["carried_from_previous_run"]
                 if c in merged.columns])
    merged.to_csv(path, index=False, encoding="utf-8")
    print(f"\nWrote {len(merged)} model rows -> {path}")
    return merged


def upsert_predictions(task, frames):
    path = RESULTS_DIR / f"predictions_{task}.csv"
    parts = list(frames)
    if path.exists():
        previous = pd.read_csv(path, encoding="utf-8")
        parts.insert(0, previous[~previous["model"].isin(STACK_MODELS)])
    frame = pd.concat(parts, ignore_index=True)
    frame.to_csv(path, index=False, encoding="utf-8")
    print(f"  wrote {len(frame)} prediction rows -> {path.name}")


def comparison(merged, tasks):
    rows = []
    for task in tasks:
        metric = METRIC[task]
        subset = merged[(merged["task"] == task)
                        & (merged["model"] != "dummy")].dropna(subset=[metric])
        stacks = subset[subset["model"].isin(STACK_MODELS)].sort_values(metric)
        others = subset[~subset["model"].isin(STACK_MODELS)].sort_values(metric)
        stack_name = str(stacks["model"].iloc[0])
        stack = float(stacks[metric].iloc[0])
        best_name = str(others["model"].iloc[0])
        best = float(others[metric].iloc[0])
        rows.append({"task": task, "metric": metric,
                     "stack": round(stack, 5),
                     "best_stack": stack_name,
                     "best_single": best_name,
                     "best_single_value": round(best, 5),
                     "delta": round(stack - best, 5),
                     "stack_wins": bool(stack < best),
                     "served": stack_name if stack < best else best_name,
                     "n_members": int(
                         len(resolve_base_names(
                             task, load_best_params().get(task, {}))))})
    frame = pd.DataFrame(rows)
    path = RESULTS_DIR / "ensemble_comparison.csv"
    frame.to_csv(path, index=False, encoding="utf-8")
    print(f"\n{frame.to_string(index=False)}")
    print(f"Wrote {path}")
    return frame


def main():
    selected = [t for t in os.environ.get("TASKS", ",".join(TASKS)).split(",")
                if t.strip()]
    unknown = [t for t in selected if t not in TASKS]
    assert not unknown, f"unknown task(s) {unknown}; expected {TASKS}"

    rows = []
    for task in selected:
        frames = []
        for model_name in STACK_MODELS:
            row, predictions = evaluate(task, model_name)
            rows.append(row)
            frames.append(predictions)
        upsert_predictions(task, frames)
    merged = upsert_results(rows)
    comparison(merged, selected)
    stack_rows = merged[merged["model"].isin(STACK_MODELS)]
    assert not stack_rows.duplicated(
        subset=["task", "model", "resampling"]).any(), \
        "duplicate stack rows in model_results.csv after the upsert"


if __name__ == "__main__":
    main()
