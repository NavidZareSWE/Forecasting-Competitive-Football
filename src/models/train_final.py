"""Run from the repository root with:

    python src/models/train_final.py
"""
from pathlib import Path
import hashlib
import json
import sys

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

import joblib  # noqa: E402
import pandas as pd  # noqa: E402
from sklearn.calibration import CalibratedClassifierCV  # noqa: E402

from modeling_common import (FEATURE_DIR, HAS_FROZEN, PROCESSED_DIR,  # noqa: E402
                             RESULTS_DIR, prepare_matrices, task_frame)
from model_zoo import classifier_zoo, regressor_zoo  # noqa: E402
from tuning import load_best_params  # noqa: E402

if HAS_FROZEN:
    from sklearn.frozen import FrozenEstimator

MODELS_DIR = RESULTS_DIR / "models"
ZOO_TASK = {"C": "C", "R": "R", "Lc": "L", "Lr": "L"}
TREE_MODELS = ["lightgbm", "xgboost", "gbm", "random_forest"]

COMMON_INPUTS = [RESULTS_DIR / "best_params.json",
                 RESULTS_DIR / "model_results.csv"]
TASK_INPUTS = {
    "C": [FEATURE_DIR / "prematch_features_extended.csv"] + COMMON_INPUTS,
    "R": [FEATURE_DIR / "prematch_features_extended.csv"] + COMMON_INPUTS,
    "Lc": [FEATURE_DIR / "inplay_features.csv",
           FEATURE_DIR / "prematch_features.csv",
           PROCESSED_DIR / "team_ratings.csv",
           PROCESSED_DIR / "rating_features.csv"] + COMMON_INPUTS,
    "Lr": [FEATURE_DIR / "inplay_features.csv",
           FEATURE_DIR / "prematch_features.csv",
           PROCESSED_DIR / "team_ratings.csv",
           PROCESSED_DIR / "rating_features.csv"] + COMMON_INPUTS,
}


def sha256_file(path):
    digest = hashlib.sha256()
    with open(path, "rb") as source:
        for block in iter(lambda: source.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def pick_model(results, task, allowed=None):
    metric = "rps" if task in {"C", "Lc"} else "mae"
    subset = results[(results["task"] == task) & (results["model"] != "dummy")]
    if allowed is not None:
        subset = subset[subset["model"].isin(allowed)]
    subset = subset.dropna(subset=[metric]).sort_values(metric)
    assert len(subset), f"no candidate model for task {task}"
    return str(subset["model"].iloc[0])


def fit_task(task, model_name):
    is_classification = task in {"C", "Lc"}
    df, continuous, nominal, target, task_type = task_frame(task)
    matrices = prepare_matrices(df, continuous, nominal, target, task_type)
    tuned = load_best_params().get(task, {})
    zoo = (classifier_zoo if is_classification else regressor_zoo)(
        0, task=ZOO_TASK[task], tuned=tuned)
    estimator = zoo[model_name]()
    y = matrices["y_train"]
    estimator.fit(matrices["X_train"], y if is_classification
                  else y.astype(float))
    calibrator, calibration = None, None
    if is_classification:
        calibration = "uncalibrated"
        if HAS_FROZEN:
            for method in ["isotonic", "sigmoid"]:
                try:
                    candidate = CalibratedClassifierCV(
                        FrozenEstimator(estimator), method=method)
                    candidate.fit(matrices["X_val"], matrices["y_val"])
                    calibrator, calibration = candidate, method
                    break
                except Exception:
                    continue
    return {
        "task": task,
        "model_name": model_name,
        "estimator": estimator,
        "calibrator": calibrator,
        "calibration": calibration,
        "preprocessor": matrices["preprocessor"],
        "feature_names": matrices["feature_names"],
        "continuous_cols": matrices["continuous_cols"],
        "nominal_cols": matrices["nominal_cols"],
    }


def main():
    results = pd.read_csv(RESULTS_DIR / "model_results.csv", encoding="utf-8")
    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    manifest = {}
    for task in ["C", "R", "Lc", "Lr"]:
        allowed = TREE_MODELS if task == "Lc" else None
        model_name = pick_model(results, task, allowed)
        print(f"[{task}] fitting {model_name}...")
        bundle = fit_task(task, model_name)
        path = MODELS_DIR / f"{task}.joblib"
        joblib.dump(bundle, path, compress=3)
        manifest[task] = {
            "model_name": model_name,
            "calibration": bundle["calibration"],
            "inputs": {str(p.relative_to(RESULTS_DIR.parent)): sha256_file(p)
                       for p in TASK_INPUTS[task]},
        }
        print(f"  wrote {path} ({path.stat().st_size / 1e6:.1f} MB)")
    with open(MODELS_DIR / "manifest.json", "w", encoding="utf-8") as sink:
        json.dump(manifest, sink, indent=2)
    print(f"Wrote {MODELS_DIR / 'manifest.json'}")


if __name__ == "__main__":
    main()
