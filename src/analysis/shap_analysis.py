"""Phase 8: TreeSHAP explanations and the worst-prediction post-mortem.

Produces, per task, a global beeswarm, the ten worst test predictions with a
local waterfall each, an adjudication of every failure as model error or bad
luck, and one full-match SHAP timeline for Model 3.

    python src/analysis/shap_analysis.py
"""

from pathlib import Path
import os
import sys
import warnings

import numpy as np
import pandas as pd

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import shap

HERE = Path(__file__).resolve().parent
SRC = HERE.parent
sys.path.insert(0, str(SRC / "models"))
sys.path.insert(0, str(SRC / "papers"))

from modeling_common import (CLASS_ORDER, RESULTS_DIR, prepare_matrices,
                             task_frame, regression_metrics,
                             ranked_probability_score)
from model_zoo import classifier_zoo, regressor_zoo, LabelEncodedClassifier
from tuning import load_best_params

warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=UserWarning)

PROCESSED_DIR = SRC / "reports" / "processed"
VIZ_DIR = SRC / "reports" / "visualizations" / "shap"

RANDOM_STATE = 0
N_WORST = 10
BEESWARM_SAMPLE = 1500
# Rows used for the global beeswarm and the mean-|SHAP| table.
GLOBAL_SHAP_SAMPLE = int(os.environ.get("SHAP_SAMPLE", "2000"))
ZOO_TASK = {"C": "C", "R": "R", "Lc": "L", "Lr": "L"}

# Kernel and constant models have no tree structure for TreeSHAP to walk.
TREE_MODELS = ["random_forest", "gbm", "xgboost", "lightgbm",
               "p2_hier_shrinkage"]

# Adjudication thresholds, fixed before looking at the results.
MARKET_AGREEMENT = 0.05      # model within 5pp of the market: not a model error
LOW_PROBABILITY = 0.25       # everyone rated the realised outcome unlikely
EXTREME_MARGIN = 3           # 3+ goal margins are dominated by finishing noise


# --- Model access -----------------------------------------------------------
def shap_ready(name, estimator):
    """Return the estimator TreeSHAP should walk, or None if unsupported."""
    if name == "p2_hier_shrinkage":
        return estimator.shap_base_estimator()
    if isinstance(estimator, LabelEncodedClassifier):
        return estimator.estimator_
    if name in TREE_MODELS:
        return estimator
    return None


def model_class_order(name, estimator):
    if isinstance(estimator, LabelEncodedClassifier):
        return list(estimator.encoder_.classes_)
    return [str(c) for c in estimator.classes_]


def fit_candidates(task, matrices):
    """Fit every TreeSHAP-capable model, scored on validation only."""
    tuned = load_best_params().get(task, {})
    is_classification = task in {"C", "Lc"}
    zoo = (classifier_zoo(RANDOM_STATE, task=ZOO_TASK[task], tuned=tuned)
           if is_classification
           else regressor_zoo(RANDOM_STATE, task=ZOO_TASK[task], tuned=tuned))

    fitted = {}
    for name in TREE_MODELS:
        if name not in zoo:
            continue
        estimator = zoo[name]()
        y_train = matrices["y_train"]
        estimator.fit(matrices["X_train"],
                      y_train if is_classification else y_train.astype(float))
        if is_classification:
            proba = _proba_in_class_order(name, estimator, matrices["X_val"])
            score = ranked_probability_score(proba, matrices["y_val"])
        else:
            prediction = np.clip(estimator.predict(matrices["X_val"]), -5, 5)
            score = regression_metrics(prediction,
                                       matrices["y_val"].astype(float))["mae"]
        fitted[name] = (estimator, float(score))
        print(f"  [{task}] {name:18s} validation "
              f"{'rps' if is_classification else 'mae'}={score:.5f}")
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
    if values.ndim == 2:                       # binary fallback, unused here
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
    """Model error, uncertain, noisy state, or reasonable-but-unlucky."""
    market_p = market_row.get(realised) if market_row else None
    if state is not None and state.get("contradicted"):
        return ("noisy observation",
                "the in-play state at the snapshot pointed the other way; "
                "the match turned after time t")
    if market_p is not None and market_p - p_realised > MARKET_AGREEMENT:
        return ("model error",
                f"the market gave the realised outcome {market_p:.3f} against "
                f"the model's {p_realised:.3f}; the information was available")
    if market_p is not None and market_p < LOW_PROBABILITY:
        return ("inherently uncertain",
                f"the market also rated this outcome unlikely ({market_p:.3f}); "
                "the result, not the forecast, was the outlier")
    if p_realised >= base_rate:
        return ("reasonable despite outcome",
                f"the model rated the realised outcome at or above its base "
                f"rate ({p_realised:.3f} vs {base_rate:.3f})")
    return ("model error",
            f"the model rated the realised outcome below its unconditional "
            f"base rate ({p_realised:.3f} vs {base_rate:.3f})")


def adjudicate_regression(error, y_true, typical_error, state=None):
    if state is not None and state.get("contradicted"):
        return ("noisy observation",
                "the scoreline at the snapshot moved against the eventual "
                "margin after time t")
    if abs(y_true) >= EXTREME_MARGIN:
        return ("inherently uncertain",
                f"the realised margin was {y_true:+.0f}; margins of this size "
                "are dominated by finishing variance")
    if error > 3 * typical_error:
        return ("model error",
                f"absolute error {error:.2f} is more than three times the "
                f"model's typical {typical_error:.2f} on an ordinary scoreline")
    return ("reasonable despite outcome",
            f"absolute error {error:.2f} against a typical {typical_error:.2f}")


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

    # TreeSHAP is exact but its cost grows with rows, trees and features. On
    # the widened in-play table the full test split is thousands of rows by
    # hundreds of features, which is far more than the global plots need. The
    # global view is computed on a fixed random sample; the worst predictions
    # and the match timeline are computed exactly on the rows they concern, so
    # nothing that is reported per row is approximated.
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


# --- In-play SHAP timeline --------------------------------------------------
def inplay_timeline(bundle, output_csv, output_png):
    """One complete match: SHAP attributions at every snapshot minute."""
    primary, estimator, matrices, names = bundle
    meta = matrices["meta_test"]
    counts = meta.groupby("match_id").size()
    full = counts[counts == counts.max()]
    # A deterministic, non-cherry-picked choice: the lowest id with full cover.
    match_id = int(full.index.min())
    mask = (meta["match_id"] == match_id).to_numpy()
    X_match = matrices["X_test"][mask]
    minutes = meta.loc[mask, "snapshot_minute"].to_numpy()
    order = np.argsort(minutes)
    X_match, minutes = X_match[order], minutes[order]

    values, expected = shap_values_for(primary, estimator, X_match, True)
    proba = _proba_in_class_order(primary, estimator, X_match)

    records = []
    for position, minute in enumerate(minutes):
        for class_position, label in enumerate(CLASS_ORDER):
            for feature_position, feature in enumerate(names):
                records.append({
                    "match_id": match_id, "model": primary,
                    "snapshot_minute": int(minute), "class": label,
                    "feature": feature,
                    "shap": round(float(values[position, feature_position,
                                               class_position]), 6),
                    "base_value": round(float(expected[class_position]), 6),
                    "p_class": round(float(proba[position, class_position]), 6)})
    frame = pd.DataFrame(records)
    frame.to_csv(output_csv, index=False, encoding="utf-8")

    top = (frame[frame["class"] == "H"].groupby("feature")["shap"]
           .apply(lambda s: s.abs().mean()).nlargest(6).index.tolist())
    figure, axes = plt.subplots(2, 1, figsize=(10, 8), sharex=True)
    for label in CLASS_ORDER:
        series = frame[(frame["class"] == label)].drop_duplicates(
            "snapshot_minute").sort_values("snapshot_minute")
        axes[0].plot(series["snapshot_minute"], series["p_class"],
                     marker="o", label=f"P({label})")
    axes[0].set_ylabel("model probability (uncalibrated)")
    axes[0].set_title(f"Match {match_id} - {primary} - probabilities and SHAP "
                      f"attributions for P(H) by minute")
    axes[0].legend(fontsize=8)
    axes[0].grid(alpha=0.3)
    for feature in top:
        series = frame[(frame["class"] == "H") & (frame["feature"] == feature)] \
            .sort_values("snapshot_minute")
        axes[1].plot(series["snapshot_minute"], series["shap"], marker=".",
                     label=feature)
    axes[1].axhline(0.0, color="black", linewidth=0.8)
    axes[1].set_xlabel("snapshot minute")
    axes[1].set_ylabel("SHAP contribution to P(H)")
    axes[1].legend(fontsize=7, ncol=2)
    axes[1].grid(alpha=0.3)
    figure.tight_layout()
    output_png.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(output_png, dpi=140)
    plt.close("all")
    print(f"\nIn-play SHAP timeline for match {match_id} "
          f"({len(minutes)} snapshots) -> {output_png.name}")
    return match_id


def main():
    market = load_market_probabilities()
    rows, worst_rows = [], []
    bundles = {}
    for task in ["C", "R", "Lc", "Lr"]:
        bundle = run_task(task, market, rows, worst_rows)
        if bundle is not None:
            bundles[task] = bundle

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    VIZ_DIR.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(RESULTS_DIR / "shap_importance.csv", index=False,
                              encoding="utf-8")
    worst = pd.DataFrame(worst_rows)
    worst.to_csv(RESULTS_DIR / "worst_predictions.csv", index=False,
                 encoding="utf-8")

    if "Lc" in bundles:
        inplay_timeline(bundles["Lc"],
                        RESULTS_DIR / "shap_inplay_timeline.csv",
                        VIZ_DIR / "inplay_timeline.png")

    print("\nVerdict counts over all tasks:")
    print(worst.groupby(["task", "verdict"]).size().to_string())
    print(f"\nWrote -> {RESULTS_DIR / 'shap_importance.csv'}")
    print(f"Wrote -> {RESULTS_DIR / 'worst_predictions.csv'}")
    print(f"Wrote -> {RESULTS_DIR / 'shap_inplay_timeline.csv'}")
    print(f"Plots -> {VIZ_DIR}")


if __name__ == "__main__":
    main()
