"""Phase 8: TreeSHAP explanations and the worst-prediction post-mortem.

Produces, per task, a global beeswarm, the ten worst test predictions with a
local waterfall each, an adjudication of every failure as model error or bad
luck, and one full-match SHAP timeline for Model 3.

    python src/analysis/shap_analysis.py
"""

from tuning import load_best_params
from model_zoo import classifier_zoo, regressor_zoo, LabelEncodedClassifier
from modeling_common import (CLASS_ORDER, RESULTS_DIR, prepare_matrices,
                             task_frame, regression_metrics,
                             ranked_probability_score)
import shap
import matplotlib.pyplot as plt
from pathlib import Path
import os
import sys
import warnings

import numpy as np
import pandas as pd

import matplotlib
matplotlib.use("Agg")

HERE = Path(__file__).resolve().parent
SRC = HERE.parent
sys.path.insert(0, str(SRC / "models"))
sys.path.insert(0, str(SRC / "papers"))


warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=UserWarning)

PROCESSED_DIR = SRC / "reports" / "processed"
VIZ_DIR = SRC / "reports" / "visualizations" / "shap"

RANDOM_STATE = 0
N_WORST = 10
BEESWARM_SAMPLE = 1500
GLOBAL_SHAP_SAMPLE = int(os.environ.get("SHAP_SAMPLE", "2000"))
ZOO_TASK = {"C": "C", "R": "R", "Lc": "L", "Lr": "L"}

TREE_MODELS = ["random_forest", "gbm", "xgboost", "lightgbm",
               "p2_hier_shrinkage"]

MARKET_AGREEMENT = 0.05
LOW_PROBABILITY = 0.25
EXTREME_MARGIN = 3


# --- Model access -----------------------------------------------------------
def shap_ready(name, estimator):
    """Return the TreeSHAP-walkable estimator, or None if unavailable."""
    if name not in TREE_MODELS:
        return None
    if name == "p2_hier_shrinkage":
        return estimator.shap_base_estimator()
    return estimator


def model_class_order(name, estimator):
    if name == "p2_hier_shrinkage":
        return list(estimator.classes_)
    return list(getattr(estimator, "classes_", CLASS_ORDER))


def fit_candidates(task, matrices):
    """Fit every TreeSHAP-capable model on train, score on validation.

    Returns {name: (estimator, val_score)} sorted by val_score ascending.
    """
    best_params = load_best_params().get(task, {})
    is_classification = task in {"C", "Lc"}
    zoo_key = ZOO_TASK[task]
    zoo = (classifier_zoo(zoo_key, best_params)
           if is_classification else regressor_zoo(zoo_key, best_params))

    X_train = matrices["X_train"]
    y_train = matrices["y_train"]
    X_val = matrices["X_val"]
    y_val = matrices["y_val"]

    fitted = {}
    for name, estimator in zoo.items():
        if name not in TREE_MODELS:
            continue
        try:
            estimator.fit(X_train, y_train)
            if is_classification:
                proba = _proba_in_class_order(name, estimator, X_val)
                score = float(ranked_probability_score(proba, y_val))
            else:
                pred = np.clip(estimator.predict(X_val), -5, 5)
                score = float(np.abs(pred - y_val.astype(float)).mean())
            fitted[name] = (estimator, score)
            print(f"  [{task}] {name}: val score {score:.5f}")
        except Exception as exc:
            print(f"  [{task}] {name}: skipped ({exc})")
    return fitted


def _proba_in_class_order(name, estimator, X):
    proba = estimator.predict_proba(X)
    classes = model_class_order(name, estimator)
    aligned = np.zeros((proba.shape[0], len(CLASS_ORDER)))
    for j, label in enumerate(CLASS_ORDER):
        if label in classes:
            aligned[:, j] = proba[:, classes.index(label)]
    totals = aligned.sum(axis=1, keepdims=True)
    totals[totals == 0] = 1.0
    return aligned / totals


# --- SHAP values ------------------------------------------------------------
def shap_values_for(name, estimator, X, is_classification):
    """(values, expected_value). Classification values are (n, f, 3) in H/D/A."""
    walkable = shap_ready(name, estimator)
    if walkable is None:
        return None, None
    explainer = shap.TreeExplainer(walkable)
    values = np.asarray(explainer.shap_values(X))
    expected = np.asarray(explainer.expected_value, dtype=float)
    if not is_classification:
        return values.reshape(X.shape[0], X.shape[1]), float(expected.ravel()[0])

    classes = model_class_order(name, estimator)
    if values.ndim == 2:
        values = np.stack([-values, values], axis=2)
    order = [classes.index(c) for c in CLASS_ORDER if c in classes]
    return values[:, :, order], expected[order]


def beeswarm(values, X, feature_names, title, output_path, sample=BEESWARM_SAMPLE):
    rng = np.random.default_rng(RANDOM_STATE)
    index = (np.sort(rng.choice(X.shape[0], size=sample, replace=False))
             if X.shape[0] > sample else np.arange(X.shape[0]))
    plt.figure()
    shap.summary_plot(values[index], X[index], feature_names=feature_names,
                      show=False, max_display=18)
    plt.title(title, fontsize=10)
    plt.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, dpi=140, bbox_inches="tight")
    plt.close("all")


def waterfall(values, expected, x_row, feature_names, title, output_path):
    explanation = shap.Explanation(values=values, base_values=expected,
                                   data=x_row, feature_names=feature_names)
    plt.figure()
    shap.plots.waterfall(explanation, max_display=14, show=False)
    plt.title(title, fontsize=9)
    plt.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, dpi=140, bbox_inches="tight")
    plt.close("all")


# --- Failure adjudication ---------------------------------------------------
def load_market_probabilities():
    path = PROCESSED_DIR / "market_baseline.csv"
    if not path.exists():
        return {}
    market = pd.read_csv(path, encoding="utf-8")
    columns = {"H": "p_home", "D": "p_draw", "A": "p_away"}
    return {int(row.match_id): {c: float(getattr(row, columns[c]))
                                for c in CLASS_ORDER}
            for row in market.itertuples()}


def adjudicate_classification(realised, p_realised, market_row, base_rate,
                              state=None):
    if market_row is not None:
        market_p = market_row.get(realised, 0.0)
        if abs(p_realised - market_p) <= MARKET_AGREEMENT:
            return "market_agreement", (
                f"model P({realised})={p_realised:.3f} within "
                f"{MARKET_AGREEMENT} of market {market_p:.3f}")
    if market_row is not None and market_row.get(realised, 1.0) < LOW_PROBABILITY:
        return "low_probability", (
            f"market also rated P({realised})={market_row[realised]:.3f} < "
            f"{LOW_PROBABILITY}; outcome was unlikely for everyone")
    if p_realised < LOW_PROBABILITY and base_rate.get(realised, 1.0) >= LOW_PROBABILITY:
        return "model_error", (
            f"model assigned P({realised})={p_realised:.3f} well below "
            f"base rate {base_rate.get(realised, 0):.3f}")
    if state is not None and state.get("contradicted"):
        return "state_misleading", (
            f"goal diff={state['goal_diff']:+.0f} pointed away from "
            f"the eventual {realised}")
    return "uncertain", "no single cause identified"


def adjudicate_regression(error, y_true, typical_error, state=None):
    if abs(float(y_true)) >= EXTREME_MARGIN:
        return "extreme_margin", (
            f"true margin={y_true:+.0f} goals; high-margin matches "
            f"are dominated by finishing noise")
    if error <= typical_error:
        return "within_typical", (
            f"error={error:.2f} <= median {typical_error:.2f}; "
            f"not an unusually large failure")
    if state is not None and state.get("contradicted"):
        return "state_misleading", (
            f"goal diff={state['goal_diff']:+.0f} pointed away from "
            f"the eventual margin {y_true:+.0f}")
    return "model_error", (
        f"error={error:.2f} > median {typical_error:.2f} with no "
        f"mitigating factor found")


# --- Per-task driver --------------------------------------------------------
def run_task(task, market, rows, worst_rows):
    print(f"\n=== SHAP: task {task} ===")
    is_classification = task in {"C", "Lc"}
    df, continuous, nominal, target, task_type = task_frame(task)
    matrices = prepare_matrices(df, continuous, nominal, target, task_type)
    names = matrices["feature_names"]
    X_test = matrices["X_test"]
    y_test = matrices["y_test"]
    meta = matrices["meta_test"]

    fitted = fit_candidates(task, matrices)
    if not fitted:
        print(f"  [{task}] no TreeSHAP-capable model available; skipped")
        return None

    primary = min(fitted, key=lambda k: fitted[k][1])
    estimator = fitted[primary][0]
    print(f"  [{task}] primary explained model (validation-selected): {primary}")

    rng = np.random.default_rng(RANDOM_STATE)
    if X_test.shape[0] > GLOBAL_SHAP_SAMPLE:
        global_index = np.sort(rng.choice(X_test.shape[0],
                                          size=GLOBAL_SHAP_SAMPLE,
                                          replace=False))
        print(f"  [{task}] global SHAP on a {GLOBAL_SHAP_SAMPLE}-row sample "
              f"of {X_test.shape[0]} test rows; worst-case rows are explained "
              f"exactly")
    else:
        global_index = np.arange(X_test.shape[0])
    X_global = X_test[global_index]
    values, expected = shap_values_for(primary, estimator, X_global,
                                       is_classification)

    # --- Global beeswarm ---
    if is_classification:
        for position, label in enumerate(CLASS_ORDER):
            beeswarm(values[:, :, position], X_global, names,
                     f"Task {task} - {primary} - SHAP for P({label})",
                     VIZ_DIR / f"beeswarm_{task}_{primary}_{label}.png")
    else:
        beeswarm(values, X_global, names,
                 f"Task {task} - {primary} - SHAP for predicted margin",
                 VIZ_DIR / f"beeswarm_{task}_{primary}.png")

    # --- Mean |SHAP| importance table ---
    if is_classification:
        importance = np.abs(values).mean(axis=(0, 2))
    else:
        importance = np.abs(values).mean(axis=0)
    for name, value in sorted(zip(names, importance), key=lambda p: -p[1]):
        rows.append({"task": task, "model": primary, "feature": name,
                     "mean_abs_shap": round(float(value), 6)})

    # --- Worst predictions ---
    if is_classification:
        proba = _proba_in_class_order(primary, estimator, X_test)
        index_of = {c: i for i, c in enumerate(CLASS_ORDER)}
        realised_index = np.array([index_of[label] for label in y_test])
        loss = np.array([ranked_probability_score(proba[[i]], y_test[[i]])
                         for i in range(len(y_test))])
        p_realised = proba[np.arange(len(y_test)), realised_index]
        base_rate = {label: float((matrices["y_train"] == label).mean())
                     for label in CLASS_ORDER}
    else:
        prediction = np.clip(estimator.predict(X_test), -5, 5)
        loss = np.abs(prediction - y_test.astype(float))
        typical = float(np.median(loss))

    order = np.argsort(-loss)[:N_WORST]
    for rank, position in enumerate(order, start=1):
        match_id = int(meta.loc[position, "match_id"])
        minute = (int(meta.loc[position, "snapshot_minute"])
                  if "snapshot_minute" in meta.columns else None)
        state = _snapshot_state(df, match_id, minute, y_test[position],
                                is_classification)
        record = {"task": task, "model": primary, "rank": rank,
                  "match_id": match_id, "snapshot_minute": minute,
                  "loss": round(float(loss[position]), 5)}
        row_values, row_expected = shap_values_for(
            primary, estimator, X_test[[position]], is_classification)
        if is_classification:
            realised = str(y_test[position])
            verdict, reason = adjudicate_classification(
                realised, float(p_realised[position]),
                market.get(match_id), base_rate[realised], state)
            record.update({
                "y_true": realised,
                "p_realised": round(float(p_realised[position]), 5),
                "p_predicted_class": CLASS_ORDER[int(proba[position].argmax())],
                "p_max": round(float(proba[position].max()), 5),
                "market_p_realised": (round(market[match_id][realised], 5)
                                      if match_id in market else None)})
            shap_row = row_values[0, :, realised_index[position]]
            expected_row = float(row_expected[realised_index[position]])
            title = (f"Task {task} #{rank} match {match_id}"
                     f"{'' if minute is None else f' @ {minute}\''}"
                     f" - SHAP for P({realised})")
        else:
            verdict, reason = adjudicate_regression(
                float(loss[position]), float(y_test[position]), typical, state)
            record.update({"y_true": float(y_test[position]),
                           "y_pred": round(float(prediction[position]), 5)})
            shap_row = row_values[0]
            expected_row = row_expected
            title = (f"Task {task} #{rank} match {match_id}"
                     f"{'' if minute is None else f' @ {minute}\''}"
                     f" - SHAP for predicted margin")
        record["verdict"] = verdict
        record["reasoning"] = reason
        worst_rows.append(record)

        waterfall(shap_row, expected_row, X_test[position], names, title,
                  VIZ_DIR / f"worst_{task}_{rank:02d}_match{match_id}.png")

    task_rows = [r for r in worst_rows if r["task"] == task]
    verdicts = pd.Series([r["verdict"] for r in task_rows]).value_counts()
    distinct = len({r["match_id"] for r in task_rows})
    # On the snapshot tasks consecutive minutes of one match fail together, so
    # the worst ten rows are not ten independent failures.
    print(f"  [{task}] worst-{N_WORST} verdicts: {verdicts.to_dict()}  "
          f"({distinct} distinct matches)")
    return primary, estimator, matrices, names


def _snapshot_state(df, match_id, minute, y_true, is_classification):
    """Did the in-play state at time t point away from the final outcome?"""
    if minute is None or "snapshot_minute" not in df.columns:
        return None
    row = df[(df["match_id"] == match_id) & (df["snapshot_minute"] == minute)]
    if row.empty or "inplay_goal_diff" not in row.columns:
        return None
    goal_diff = float(row["inplay_goal_diff"].iloc[0])
    if is_classification:
        leader = "H" if goal_diff > 0 else ("A" if goal_diff < 0 else "D")
        contradicted = goal_diff != 0 and leader != str(y_true)
    else:
        contradicted = goal_diff * float(y_true) < 0
    return {"goal_diff": goal_diff, "contradicted": bool(contradicted)}


def main():
    market = load_market_probabilities()
    rows, worst_rows = [], []
    for task in ["C", "R", "Lc", "Lr"]:
        run_task(task, market, rows, worst_rows)

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    VIZ_DIR.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(RESULTS_DIR / "shap_importance.csv", index=False,
                              encoding="utf-8")
    worst = pd.DataFrame(worst_rows)
    worst.to_csv(RESULTS_DIR / "worst_predictions.csv", index=False,
                 encoding="utf-8")
    print(f"\nWrote -> {RESULTS_DIR / 'shap_importance.csv'}")
    print(f"Wrote -> {RESULTS_DIR / 'worst_predictions.csv'}")
    print(f"Plots -> {VIZ_DIR}")


if __name__ == "__main__":
    main()
