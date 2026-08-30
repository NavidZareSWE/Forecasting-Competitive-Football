"""Run from the repository root with:

    python src/models/train_market_blend.py

Declared experiment: de-vigged pre-match odds are used as a FEATURE here,
so the market baseline is not the yardstick inside this experiment - it is
one of the two components being blended. Tasks C and L keep odds out of
their feature sets everywhere else; this script writes its own predictions
and never feeds anything back into them.
"""
from pathlib import Path
import sys

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

from modeling_common import (CLASS_ORDER, RESULTS_DIR, PROCESSED_DIR,  # noqa: E402
                             classification_metrics, floor_probabilities,
                             prepare_matrices, task_frame)
from model_zoo import classifier_zoo, LabelEncodedClassifier  # noqa: E402
from stacking import STACK_MODELS, build_named_stack  # noqa: E402
from tuning import load_best_params  # noqa: E402
from train_final import pick_model  # noqa: E402
from sklearn.calibration import CalibratedClassifierCV  # noqa: E402
from modeling_common import HAS_FROZEN  # noqa: E402

if HAS_FROZEN:
    from sklearn.frozen import FrozenEstimator

ALPHA_GRID = np.round(np.arange(0.0, 1.01, 0.02), 2)
DECAY_GRID = [10.0, 20.0, 30.0, 45.0, 90.0]
ZOO_TASK = {"C": "C", "Lc": "L"}


def aligned_proba(estimator, X):
    proba = estimator.predict_proba(X)
    classes = (list(estimator.encoder_.classes_)
               if isinstance(estimator, LabelEncodedClassifier)
               else [str(c) for c in estimator.classes_])
    out = np.zeros((proba.shape[0], len(CLASS_ORDER)))
    for j, label in enumerate(CLASS_ORDER):
        if label in classes:
            out[:, j] = proba[:, classes.index(label)]
    totals = out.sum(axis=1, keepdims=True)
    totals[totals == 0] = 1.0
    return out / totals


def fit_task(task):
    model_name = pick_model(
        pd.read_csv(RESULTS_DIR / "model_results.csv", encoding="utf-8"), task)
    df, continuous, nominal, target, task_type = task_frame(task)
    matrices = prepare_matrices(df, continuous, nominal, target, task_type)
    tuned = load_best_params().get(task, {})
    if model_name in STACK_MODELS:
        estimator = build_named_stack(model_name, task, tuned, df,
                                      transform=matrices["transform"])
    else:
        estimator = classifier_zoo(0, task=ZOO_TASK[task],
                                   tuned=tuned)[model_name]()
    estimator.fit(matrices["X_train"], matrices["y_train"])
    source = estimator
    if HAS_FROZEN:
        for method in ["isotonic", "sigmoid"]:
            try:
                calibrated = CalibratedClassifierCV(FrozenEstimator(estimator),
                                                    method=method)
                calibrated.fit(matrices["X_val"], matrices["y_val"])
                source = calibrated
                break
            except Exception:
                continue
    frames = {}
    for split in ["validation", "test"]:
        subset = df[df["split"] == split]
        proba = floor_probabilities(
            aligned_proba(source, matrices["transform"](subset)))
        frame = subset[["match_id"] + (["snapshot_minute"]
                                       if "snapshot_minute" in subset.columns
                                       else [])].reset_index(drop=True)
        for j, label in enumerate(CLASS_ORDER):
            frame[f"p_{label}"] = proba[:, j]
        frame["y_true"] = subset[target].to_numpy()
        frames[split] = frame
    return model_name, frames


def market_probabilities():
    market = pd.read_csv(PROCESSED_DIR / "market_baseline_extended.csv",
                         encoding="utf-8",
                         usecols=["match_id", "p_home", "p_draw", "p_away"])
    return market.rename(columns={"p_home": "m_H", "p_draw": "m_D",
                                  "p_away": "m_A"})


def blend(frame, alpha):
    model = frame[[f"p_{c}" for c in CLASS_ORDER]].to_numpy()
    market = frame[[f"m_{c}" for c in CLASS_ORDER]].to_numpy()
    return alpha * market + (1.0 - alpha) * model


def rps_of(frame, proba):
    return classification_metrics(proba, frame["y_true"].to_numpy())["rps"]


def run_c(rows):
    model_name, frames = fit_task("C")
    market = market_probabilities()
    joined = {s: f.merge(market, on="match_id", how="inner")
              for s, f in frames.items()}
    scores = [(alpha, rps_of(joined["validation"],
                             blend(joined["validation"], alpha)))
              for alpha in ALPHA_GRID]
    best_alpha = min(scores, key=lambda t: t[1])[0]
    test = joined["test"]
    for name, proba in [
            ("model_only", test[[f"p_{c}" for c in CLASS_ORDER]].to_numpy()),
            ("market_only", test[[f"m_{c}" for c in CLASS_ORDER]].to_numpy()),
            ("blend", blend(test, best_alpha))]:
        rows.append({"task": "C", "arm": name, "base_model": model_name,
                     "alpha": best_alpha if name == "blend" else None,
                     "n": len(test), "rps": round(rps_of(test, proba), 5)})
    out = test[["match_id", "y_true"]].copy()
    blended = blend(test, best_alpha)
    for j, label in enumerate(CLASS_ORDER):
        out[f"p_{label}"] = np.round(blended[:, j], 5)
    out.to_csv(RESULTS_DIR / "predictions_Cm.csv", index=False,
               encoding="utf-8")
    print(f"[C] {model_name}: alpha={best_alpha} "
          f"(validation-chosen over {len(ALPHA_GRID)} points)")


def run_lc(rows):
    model_name, frames = fit_task("Lc")
    market = market_probabilities()
    joined = {s: f.merge(market, on="match_id", how="inner")
              for s, f in frames.items()}

    def decayed(frame, alpha0, tau):
        weight = alpha0 * np.exp(-frame["snapshot_minute"].to_numpy() / tau)
        model = frame[[f"p_{c}" for c in CLASS_ORDER]].to_numpy()
        mkt = frame[[f"m_{c}" for c in CLASS_ORDER]].to_numpy()
        return weight[:, None] * mkt + (1.0 - weight[:, None]) * model

    scores = [((alpha0, tau),
               rps_of(joined["validation"],
                      decayed(joined["validation"], alpha0, tau)))
              for alpha0 in ALPHA_GRID for tau in DECAY_GRID]
    (best_alpha0, best_tau), _ = min(scores, key=lambda t: t[1])
    test = joined["test"]
    for name, proba in [
            ("model_only", test[[f"p_{c}" for c in CLASS_ORDER]].to_numpy()),
            ("blend", decayed(test, best_alpha0, best_tau))]:
        rows.append({"task": "Lc", "arm": name, "base_model": model_name,
                     "alpha": best_alpha0 if name == "blend" else None,
                     "tau": best_tau if name == "blend" else None,
                     "n": len(test), "rps": round(rps_of(test, proba), 5)})
    print(f"[Lc] {model_name}: alpha0={best_alpha0}, tau={best_tau} minutes")


def main():
    rows = []
    run_c(rows)
    run_lc(rows)
    results = pd.DataFrame(rows)
    results.to_csv(RESULTS_DIR / "market_blend.csv", index=False,
                   encoding="utf-8")
    print(results.to_string(index=False))
    print(f"Wrote {RESULTS_DIR / 'market_blend.csv'}, predictions_Cm.csv")


if __name__ == "__main__":
    main()
