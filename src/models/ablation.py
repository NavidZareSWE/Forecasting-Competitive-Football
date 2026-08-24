"""Phase 8: ablation and honest re-training (brief flowchart stage 10).

Two axes, both scored on the VALIDATION split only. Ablation is model
selection: choosing a configuration by its test score would be tuning against
the test set, so the test split is never read here.

  feature_group       drop one group of columns at a time and refit, per task.
  snapshot_frequency  thin the TRAINING snapshots to a coarser minute grid;
                      validation stays at full density so every arm is scored
                      on identical rows.

The balancing axis is deliberately absent: it is the subject of the P1 paper
and lives in resampling_study.py (report section 5.2).

Writes src/reports/ablation.csv, read by report section 8.

    python src/models/ablation.py
"""

import numpy as np
import pandas as pd

from modeling_common import (CLASS_ORDER, RESULTS_DIR, _proba_in_order,
                             classification_metrics, per_class_metrics,
                             prepare_matrices, regression_metrics, task_frame)
from model_zoo import classifier_zoo, regressor_zoo
from tuning import load_best_params


ZOO_TASK = {"C": "C", "R": "R", "Lc": "L", "Lr": "L"}
TASKS = ["C", "R", "Lc", "Lr"]
MODELS_PER_TASK = 2          # best tuned tree model plus the runner-up
SNAPSHOT_GRIDS = [5, 15, 45]  # every-5 is the full table; the others thin it


# --- Feature groups ---------------------------------------------------------
# Every model column belongs to exactly one group (asserted below), so the
# drops partition the feature set. snapshot_minute is positional context for
# the in-play tasks and is never dropped.
RESULT_STATS = {"gf", "ga", "points", "win"}
XG_STATS = {"xgf", "xga"}
EVENT_RATE_STATS = {"shots", "shots_on_target", "pressures",
                    "defensive_actions", "corners", "free_kicks", "throw_ins"}
POSSESSION_STATS = {"possession_share", "passes", "pass_completion",
                    "passes_def", "passes_mid", "passes_fin",
                    "carries_final_third"}


def column_group(column):
    if column == "snapshot_minute":
        return None
    base = column
    for prefix in ["home_", "away_", "diff_"]:
        if base.startswith(prefix):
            base = base[len(prefix):]
            break
    if base.startswith("form_"):
        stat = base[len("form_"):]
        if stat in XG_STATS:
            return "expected_goals"
        if stat in RESULT_STATS:
            return "results_form"
        if stat in EVENT_RATE_STATS:
            return "event_rate"
        if stat in POSSESSION_STATS:
            return "possession_territory"
    if base in {"rest_days", "played_prior"}:
        return "schedule"
    if base.startswith("h2h_"):
        return "h2h"
    if column.startswith("inplay_"):
        if "xg" in column:
            return "inplay_xg"
        if any(key in column for key in ["goal", "man_advantage", "card",
                                         "foul"]):
            return "current_score"
        return "inplay_volume"
    raise AssertionError(f"feature column {column!r} belongs to no ablation "
                         "group; extend column_group so the drops stay a "
                         "partition")


def group_columns(continuous_cols):
    groups = {}
    for column in continuous_cols:
        group = column_group(column)
        if group is not None:
            groups.setdefault(group, []).append(column)
    return groups


# --- Model selection --------------------------------------------------------
def representative_models(task):
    """The strongest sweep models on the task, dummy excluded."""
    path = RESULTS_DIR / "model_results.csv"
    if not path.exists():
        raise FileNotFoundError(f"Missing {path}. Run run_models.py first.")
    results = pd.read_csv(path, encoding="utf-8")
    subset = results[(results["task"] == task)
                     & (results["model"] != "dummy")].copy()
    metric, ascending = (("rps", True) if task in {"C", "Lc"}
                         else ("mae", True))
    subset = subset.dropna(subset=[metric]).sort_values(metric,
                                                        ascending=ascending)
    names = subset["model"].head(MODELS_PER_TASK).tolist()
    assert names, f"no sweep results found for task {task}"
    return names


def build_factory(task, model_name, random_state=0):
    tuned = load_best_params().get(task, {})
    zoo = (classifier_zoo if task in {"C", "Lc"} else regressor_zoo)(
        random_state, task=ZOO_TASK[task], tuned=tuned)
    assert model_name in zoo, f"{model_name} not in the zoo for task {task}"
    return zoo[model_name]


# --- Scoring ----------------------------------------------------------------
def validation_score(task, df, continuous, nominal, target, task_type,
                     factory):
    data = prepare_matrices(df, continuous, nominal, target, task_type)
    estimator = factory()
    estimator.fit(data["X_train"], data["y_train"])
    row = {"n_train": int(len(data["X_train"]))}
    if task_type == "classification":
        proba = _proba_in_order(estimator, data["X_val"], CLASS_ORDER)
        row.update({k: v for k, v in
                    classification_metrics(proba, data["y_val"]).items()
                    if k in {"rps", "log_loss", "brier"}})
    else:
        predictions = np.clip(estimator.predict(data["X_val"]), -5.0, 5.0)
        row.update({k: v for k, v in
                    regression_metrics(predictions, data["y_val"]).items()
                    if k in {"mae", "rmse"}})
    return row


def run_feature_group_axis(task):
    df, continuous, nominal, target, task_type = task_frame(task)
    groups = group_columns(continuous)
    rows = []
    for model_name in representative_models(task):
        factory = build_factory(task, model_name)
        configurations = [("full", continuous)]
        for group, dropped in sorted(groups.items()):
            kept = [c for c in continuous if c not in set(dropped)]
            configurations.append((f"drop_{group}", kept))
        for configuration, columns in configurations:
            print(f"  [{task}] {model_name:18s} {configuration:28s} "
                  f"({len(columns)} columns)", flush=True)
            row = validation_score(task, df, columns, nominal, target,
                                   task_type, factory)
            row.update({"task": task, "axis": "feature_group",
                        "configuration": configuration, "model": model_name,
                        "n_features": len(columns)})
            rows.append(row)
    return rows


def run_snapshot_frequency_axis(task):
    df, continuous, nominal, target, task_type = task_frame(task)
    full_minutes = sorted(df["snapshot_minute"].unique())
    rows = []
    for model_name in representative_models(task):
        factory = build_factory(task, model_name)
        for grid in SNAPSHOT_GRIDS:
            kept_minutes = [m for m in full_minutes if m % grid == 0]
            # Thin the TRAINING rows only; validation keeps every minute so
            # all arms are scored on identical rows.
            thinned = df[(df["split"] != "train")
                         | df["snapshot_minute"].isin(kept_minutes)]
            print(f"  [{task}] {model_name:18s} every_{grid:<2d} "
                  f"({len(kept_minutes)} train snapshots/match)", flush=True)
            row = validation_score(task, thinned, continuous, nominal, target,
                                   task_type, factory)
            row.update({
                "task": task, "axis": "snapshot_frequency",
                "configuration": f"every_{grid}", "model": model_name,
                "train_snapshots_per_match": len(kept_minutes),
                "eval_snapshots_per_match": len(full_minutes),
            })
            rows.append(row)
    return rows


def run_p1_comparison():
    """With-P1 against without-P1 per classifier, validation only (report 5.2).

    The pre-match classification task is the only one where resampling is
    permitted (the snapshot table must never be oversampled across matches).
    """
    from model_zoo import classifier_zoo
    task = "C"
    df, continuous, nominal, target, task_type = task_frame(task)
    tuned = load_best_params().get(task, {})
    zoo = classifier_zoo(0, task=ZOO_TASK[task], tuned=tuned)
    rows = []
    for model_name, factory in zoo.items():
        if model_name == "dummy":
            continue
        row = {"model": model_name}
        for label, resampling in [("without_p1", "none"), ("with_p1", "p1")]:
            data = prepare_matrices(df, continuous, nominal, target,
                                    task_type, resampling=resampling)
            estimator = factory()
            estimator.fit(data["X_train"], data["y_train"])
            proba = _proba_in_order(estimator, data["X_val"], CLASS_ORDER)
            metrics = classification_metrics(proba, data["y_val"])
            per_class = per_class_metrics(proba, data["y_val"])
            row[f"rps_{label}"] = round(metrics["rps"], 5)
            row[f"recall_D_{label}"] = round(per_class["recall_D"], 5)
            row[f"f1_D_{label}"] = round(per_class["f1_D"], 5)
        row["p1_helps_rps"] = row["rps_with_p1"] < row["rps_without_p1"]
        print(f"  [C] {model_name:18s} rps {row['rps_without_p1']:.5f} -> "
              f"{row['rps_with_p1']:.5f}  recall_D "
              f"{row['recall_D_without_p1']:.3f} -> "
              f"{row['recall_D_with_p1']:.3f}", flush=True)
        rows.append(row)
    comparison = pd.DataFrame(rows)[[
        "model", "rps_without_p1", "rps_with_p1", "p1_helps_rps",
        "recall_D_without_p1", "recall_D_with_p1",
        "f1_D_without_p1", "f1_D_with_p1"]]
    assert len(comparison), "the P1 comparison produced no rows"
    output_path = RESULTS_DIR / "p1_comparison.csv"
    comparison.to_csv(output_path, index=False, encoding="utf-8")
    print(f"Wrote {len(comparison)} P1-comparison rows -> {output_path}")


def main():
    all_rows = []
    print("=== Ablation axis 1: feature groups (validation scores) ===")
    for task in TASKS:
        all_rows.extend(run_feature_group_axis(task))
    print("=== Ablation axis 2: snapshot frequency (validation scores) ===")
    for task in ["Lc", "Lr"]:
        all_rows.extend(run_snapshot_frequency_axis(task))
    print("=== P1 with/without comparison (validation scores) ===")
    run_p1_comparison()

    ablation = pd.DataFrame(all_rows)
    column_order = ["task", "axis", "configuration", "model", "n_features",
                    "train_snapshots_per_match", "eval_snapshots_per_match",
                    "n_train", "rps", "log_loss", "brier", "mae", "rmse"]
    ablation = ablation.reindex(
        columns=[c for c in column_order if c in ablation.columns])

    # --- Self-checks: the contract report section 8 relies on ---------------
    feature_axis = ablation[ablation["axis"] == "feature_group"]
    for (task, model), group in feature_axis.groupby(["task", "model"]):
        assert "full" in set(group["configuration"]), \
            f"missing the full baseline for {task}/{model}"
    assert set(ablation["task"]) == set(TASKS), "a task produced no rows"
    frequency_axis = ablation[ablation["axis"] == "snapshot_frequency"]
    assert (frequency_axis["eval_snapshots_per_match"]
            == frequency_axis["eval_snapshots_per_match"].iloc[0]).all(), \
        "evaluation density must be identical across frequency arms"

    output_path = RESULTS_DIR / "ablation.csv"
    ablation.to_csv(output_path, index=False, encoding="utf-8")
    print(f"\nWrote {len(ablation)} ablation rows -> {output_path}")


if __name__ == "__main__":
    main()
