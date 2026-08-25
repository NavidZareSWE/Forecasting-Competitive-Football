"""Run from the repository root with:

    python src/models/ablation.py

    set ABLATION_MODELS=random_forest && python src/models/ablation.py  cmd.exe
    ABLATION_MODELS=random_forest python src/models/ablation.py         bash
"""

from pathlib import Path
import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))

from modeling_common import (CLASS_ORDER, RESULTS_DIR, prepare_matrices,
                             task_frame, classification_metrics,
                             per_class_metrics, regression_metrics)
from model_zoo import classifier_zoo, regressor_zoo
from tuning import load_best_params


RANDOM_STATE = 0
MARGIN_CLIP = (-5.0, 5.0)
ZOO_TASK = {"C": "C", "R": "R", "Lc": "L", "Lr": "L"}

DEFAULT_MODELS = ["random_forest", "xgboost"]
ABLATION_MODELS = [m for m in os.environ.get(
    "ABLATION_MODELS", ",".join(DEFAULT_MODELS)).split(",") if m]

SNAPSHOT_STEPS = [5, 10, 15]

# --- Feature groups ---
PREMATCH_GROUP_RULES = [
    ("venue_form", lambda c: "venue_form_" in c or c.endswith(
        "venue_played_prior")),
    ("form_goals", lambda c: c.endswith(("_form_gf", "_form_ga"))),
    ("form_xg", lambda c: c.endswith(("_form_xgf", "_form_xga"))),
    ("form_results", lambda c: c.endswith(("_form_points", "_form_win"))),
    ("schedule", lambda c: c.endswith(("rest_days", "played_prior"))),
    ("league", lambda c: c == "competition_name"),
    ("head_to_head", lambda c: "h2h_" in c),
    ("agg_scoring", lambda c: _agg(c, ("goals",))),
    ("agg_discipline", lambda c: c.endswith("_form_red_cards")),
    ("agg_shooting", lambda c: _agg(c, ("shots", "shots_on_target", "xg",
                                        "shots_in_box"))),
    ("agg_passing", lambda c: _agg(c, ("passes", "passes_attacking_third",
                                       "passes_defensive_third",
                                       "pass_share_attacking_third"))),
    ("agg_territory", lambda c: _agg(c, ("carries", "carries_attacking_third",
                                         "dribbles", "mean_x",
                                         "touches_attacking_third"))),
    ("agg_pressure", lambda c: _agg(c, ("pressures",
                                        "pressure_rate_attacking_third",
                                        "events_under_pressure",
                                        "under_pressure_share"))),
    ("agg_defence", lambda c: _agg(c, ("defensive_actions",
                                       "defensive_actions_own_third",
                                       "fouls_committed", "fouls_won",
                                       "turnovers"))),
    ("agg_possession", lambda c: _agg(c, ("possession_chains",
                                          "possession_share", "events"))),
    ("agg_set_pieces", lambda c: "set_piece_" in c),
]

INPLAY_GROUP_RULES = [
    ("inplay_score", lambda c: c in {"inplay_goal_diff", "inplay_home_goals",
                                     "inplay_away_goals"}
     or _inplay(c, ("goals",))),
    ("inplay_xg", lambda c: c in {"inplay_xg_diff", "inplay_home_xg",
                                  "inplay_away_xg", "inplay_recent_xg_diff"}
     or _inplay(c, ("xg",))),
    ("inplay_shots", lambda c: c in {"inplay_shot_diff", "inplay_sot_diff"}
     or _inplay(c, ("shots", "shots_on_target", "shots_in_box"))),
    ("inplay_discipline", lambda c: c == "inplay_man_advantage"),
    ("inplay_tempo", lambda c: c in {"inplay_events_so_far", "snapshot_minute",
                                     "inplay_minutes_remaining"}
     or _inplay(c, ("events",))),
    ("inplay_passing", lambda c: _inplay(
        c, ("passes", "passes_attacking_third", "passes_defensive_third",
            "pass_share_attacking_third"))),
    ("inplay_territory", lambda c: _inplay(
        c, ("carries", "carries_attacking_third", "dribbles", "mean_x",
            "touches_attacking_third"))),
    ("inplay_pressure", lambda c: _inplay(
        c, ("pressures", "pressure_rate_attacking_third",
            "events_under_pressure", "under_pressure_share"))),
    ("inplay_defence", lambda c: _inplay(
        c, ("defensive_actions", "defensive_actions_own_third",
            "fouls_committed", "fouls_won", "turnovers"))),
    ("inplay_possession", lambda c: _inplay(
        c, ("possession_chains", "possession_share"))),
    ("inplay_set_pieces", lambda c: c.startswith("inplay_")
     and "set_piece_" in c),
    ("inplay_momentum", lambda c: c.startswith(("inplay_recent_",
                                                "inplay_rate_"))),
]


def _agg(column, quantities):
    if "_agg_" not in column:
        return False
    tail = column.split("_agg_", 1)[1]
    if tail.startswith("opp_"):
        tail = tail[4:]
    return tail in quantities


def _inplay(column, quantities):
    if not column.startswith("inplay_"):
        return False
    for prefix in ("inplay_home_", "inplay_away_", "inplay_diff_",
                   "inplay_recent_diff_", "inplay_rate_diff_"):
        if column.startswith(prefix):
            return column[len(prefix):] in quantities
    return False


def _resolve(rules, columns):
    resolved, claimed = {}, set()
    for name, predicate in rules:
        members = [c for c in columns
                   if c not in claimed and predicate(c)]
        if members:
            resolved[name] = members
            claimed.update(members)
    return resolved


def groups_for(task, columns):
    rules = list(PREMATCH_GROUP_RULES)
    if task in {"Lc", "Lr"}:
        rules = rules + list(INPLAY_GROUP_RULES)
    return _resolve(rules, columns)


# --- Scoring ----------------------------------------------------------------
def score_configuration(task, df, continuous, nominal, target, task_type,
                        resampling="none", class_weight=None):
    is_classification = task in {"C", "Lc"}
    tuned = load_best_params().get(task, {})
    matrices = prepare_matrices(df, continuous, nominal, target, task_type,
                                resampling=resampling,
                                random_state=RANDOM_STATE)
    rows = []
    for model in ABLATION_MODELS:
        params = dict(tuned.get(model, {}))
        if class_weight is not None:
            params["class_weight"] = class_weight
        zoo = (classifier_zoo(RANDOM_STATE, task=ZOO_TASK[task],
                              tuned={model: params})
               if is_classification
               else regressor_zoo(RANDOM_STATE, task=ZOO_TASK[task],
                                  tuned={model: params}))
        if model not in zoo:
            continue
        estimator = zoo[model]()
        y_train = matrices["y_train"]
        estimator.fit(matrices["X_train"],
                      y_train if is_classification else y_train.astype(float))
        if is_classification:
            proba = _aligned(estimator, matrices["X_val"])
            metrics = classification_metrics(proba, matrices["y_val"])
            metrics.update(per_class_metrics(proba, matrices["y_val"]))
        else:
            prediction = np.clip(estimator.predict(matrices["X_val"]),
                                 *MARGIN_CLIP)
            metrics = regression_metrics(prediction,
                                         matrices["y_val"].astype(float))
        rows.append({"model": model,
                     "n_train": int(matrices["X_train"].shape[0]),
                     "n_val": int(matrices["X_val"].shape[0]),
                     "n_features": int(matrices["X_train"].shape[1]),
                     **{k: (round(v, 5) if isinstance(v, float) else v)
                        for k, v in metrics.items()}})
    return rows


def _aligned(estimator, X):
    proba = estimator.predict_proba(X)
    classes = [str(c) for c in estimator.classes_]
    out = np.zeros((proba.shape[0], len(CLASS_ORDER)))
    for j, label in enumerate(CLASS_ORDER):
        if label in classes:
            out[:, j] = proba[:, classes.index(label)]
    totals = out.sum(axis=1, keepdims=True)
    totals[totals == 0] = 1.0
    return out / totals


def primary_metric(task):
    return "rps" if task in {"C", "Lc"} else "mae"


# --- Axis 1: feature groups -------------------------------------------------
def feature_group_ablation(task, rows):
    df, continuous, nominal, target, task_type = task_frame(task)
    groups = groups_for(task, list(continuous) + list(nominal))

    for record in score_configuration(task, df, continuous, nominal, target,
                                      task_type):
        rows.append({"task": task, "axis": "feature_group",
                     "configuration": "full", "dropped": "", **record})

    for group, columns in groups.items():
        present = [c for c in columns if c in continuous or c in nominal]
        if not present:
            continue
        kept_continuous = [c for c in continuous if c not in present]
        kept_nominal = [c for c in nominal if c not in present]
        if not kept_continuous and not kept_nominal:
            continue
        for record in score_configuration(task, df, kept_continuous,
                                          kept_nominal, target, task_type):
            rows.append({"task": task, "axis": "feature_group",
                         "configuration": f"drop_{group}",
                         "dropped": ",".join(present), **record})
        print(f"  [{task}] drop_{group}: -{len(present)} columns")

    if task in {"Lc", "Lr"}:
        inplay_columns = [c for c in list(continuous) + list(nominal)
                          if c.startswith("inplay_") or c == "snapshot_minute"]
        kept = [c for c in continuous if c not in inplay_columns]
        for record in score_configuration(task, df, kept, nominal, target,
                                          task_type):
            rows.append({"task": task, "axis": "feature_group",
                         "configuration": "prematch_only",
                         "dropped": ",".join(inplay_columns), **record})
        print(f"  [{task}] prematch_only control")


# --- Axis 2: snapshot frequency ---------------------------------------------
def thin_training_snapshots_ablation(task, rows):
    df, continuous, nominal, target, task_type = task_frame(task)
    for step in SNAPSHOT_STEPS:
        thinned = df["snapshot_minute"] % step == 0
        kept = df[thinned | (df["split"] != "train")]
        minutes = sorted(df.loc[thinned, "snapshot_minute"].unique())
        for record in score_configuration(task, kept, continuous, nominal,
                                          target, task_type):
            rows.append({"task": task, "axis": "snapshot_frequency",
                         "configuration": f"every_{step}_min",
                         "dropped": "",
                         "train_snapshots_per_match": len(minutes),
                         "eval_snapshots_per_match":
                             int(df[df["split"] == "validation"]
                                 ["snapshot_minute"].nunique()),
                         **record})
        print(f"  [{task}] every_{step}_min: {len(minutes)} training "
              f"snapshots/match, validation held at full density")


# --- Axes 3 and 4: balancing and P1 ----------------------------------------
BALANCING_ARMS = [("vanilla", "none", None),
                  ("smote", "smote", None),
                  ("borderline_smote", "borderline_smote", None),
                  ("adasyn", "adasyn", None),
                  ("class_weight", "none", "balanced"),
                  ("p1_gsmotenc", "p1", None)]


def balancing_ablation(task, rows):
    df, continuous, nominal, target, task_type = task_frame(task)
    for arm, resampling, class_weight in BALANCING_ARMS:
        for record in score_configuration(task, df, continuous, nominal,
                                          target, task_type,
                                          resampling=resampling,
                                          class_weight=class_weight):
            rows.append({"task": task, "axis": "balancing",
                         "configuration": arm, "dropped": "", **record})
        print(f"  [{task}] balancing arm {arm}")


def p1_comparison(rows, output_path):
    frame = pd.DataFrame(rows)
    balancing = frame[(frame["axis"] == "balancing") & (frame["task"] == "C")]
    if balancing.empty:
        return pd.DataFrame()
    pivot = balancing.pivot_table(index="model", columns="configuration",
                                  values=["rps", "recall_D", "f1_D"])
    records = []
    for model in pivot.index:
        without = float(pivot.loc[model, ("rps", "vanilla")])
        with_p1 = float(pivot.loc[model, ("rps", "p1_gsmotenc")])
        records.append({
            "model": model,
            "rps_without_p1": round(without, 5),
            "rps_with_p1": round(with_p1, 5),
            "rps_delta": round(with_p1 - without, 5),
            "recall_D_without_p1": round(
                float(pivot.loc[model, ("recall_D", "vanilla")]), 5),
            "recall_D_with_p1": round(
                float(pivot.loc[model, ("recall_D", "p1_gsmotenc")]), 5),
            "f1_D_without_p1": round(
                float(pivot.loc[model, ("f1_D", "vanilla")]), 5),
            "f1_D_with_p1": round(
                float(pivot.loc[model, ("f1_D", "p1_gsmotenc")]), 5),
            "p1_helps_rps": bool(with_p1 < without)})
    comparison = pd.DataFrame(records)
    comparison.to_csv(output_path, index=False, encoding="utf-8")
    return comparison


def main():
    print(f"Ablation learners: {ABLATION_MODELS}")
    print("All scores are validation-split scores; the test split is untouched.\n")
    rows = []

    for task in ["C", "R", "Lc", "Lr"]:
        print(f"=== feature groups: task {task} ===")
        feature_group_ablation(task, rows)

    for task in ["Lc", "Lr"]:
        print(f"\n=== snapshot frequency: task {task} ===")
        thin_training_snapshots_ablation(task, rows)

    print("\n=== balancing strategy: task C ===")
    balancing_ablation("C", rows)

    results = pd.DataFrame(rows)
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    output_path = RESULTS_DIR / "ablation.csv"
    results.to_csv(output_path, index=False, encoding="utf-8")

    comparison = p1_comparison(rows, RESULTS_DIR / "p1_comparison.csv")

    print("\nFeature-group ablation, change in the primary metric versus full:")
    for task in ["C", "R", "Lc", "Lr"]:
        metric = primary_metric(task)
        subset = results[(results["task"] == task)
                         & (results["axis"] == "feature_group")]
        if subset.empty:
            continue
        table = subset.pivot_table(index="configuration", columns="model",
                                   values=metric)
        if "full" not in table.index:
            continue
        delta = (table - table.loc["full"]).drop(index="full").round(5)
        print(f"\n  task {task} ({metric}; positive means the drop hurt):")
        print(delta.sort_values(delta.columns[0], ascending=False)
              .to_string().replace("\n", "\n  "))

    frequency = results[results["axis"] == "snapshot_frequency"]
    if not frequency.empty:
        print("\nSnapshot frequency:")
        print(frequency.pivot_table(
            index=["task", "configuration"], columns="model",
            values=["rps", "mae"]).round(5).to_string())

    if not comparison.empty:
        print("\nWith-P1 versus without-P1 on validation (task C):")
        print(comparison.to_string(index=False))

    print(f"\nWrote -> {output_path}")
    print(f"Wrote -> {RESULTS_DIR / 'p1_comparison.csv'}")


if __name__ == "__main__":
    main()
