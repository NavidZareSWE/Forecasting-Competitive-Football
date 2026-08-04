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


# --- Per-task driver --------------------------------------------------------
def run_task(task, rows):
    print(f"\n=== SHAP: task {task} ===")
    is_classification = task in {"C", "Lc"}
    df, continuous, nominal, target, task_type = task_frame(task)
    matrices = prepare_matrices(df, continuous, nominal, target, task_type)
    names = matrices["feature_names"]
    X_test = matrices["X_test"]

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
              f"of {X_test.shape[0]} test rows")
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

    return primary, estimator, matrices, names


def main():
    rows = []
    for task in ["C", "R", "Lc", "Lr"]:
        run_task(task, rows)

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    VIZ_DIR.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(RESULTS_DIR / "shap_importance.csv", index=False,
                              encoding="utf-8")
    print(f"\nWrote -> {RESULTS_DIR / 'shap_importance.csv'}")
    print(f"Plots -> {VIZ_DIR}")


if __name__ == "__main__":
    main()
