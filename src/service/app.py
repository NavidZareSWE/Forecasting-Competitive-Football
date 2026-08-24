"""Bonus (brief section 11): FastAPI service wrapping Model 3, plus Models 1-2
for the pre-kickoff state.

Correctness first: no feature code is reimplemented here. The snapshot rows
come from the same inplay_features.csv / prematch_features.csv the models were
trained on, the design matrices come from modeling_common.prepare_matrices,
and the estimators come from the tuned model zoo. The service therefore cannot
disagree with the offline pipeline (measure_latency.py asserts this parity
against the sweep's stored predictions).

Latency: everything expensive happens once at startup - models fitted,
calibrators fitted on validation, the TreeSHAP explainer built, every servable
snapshot's design matrix precomputed. A request is a dictionary lookup plus
one model evaluation.

Only test-split matches are servable: the demo must run on held-out data.

    python src/service/app.py            # uvicorn on 127.0.0.1:8100 (PORT env)
"""

from pathlib import Path
import os
import sys

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
SRC = HERE.parent
sys.path.insert(0, str(SRC / "models"))

import shap  # noqa: E402
from fastapi import FastAPI, HTTPException  # noqa: E402
from fastapi.responses import FileResponse  # noqa: E402
from sklearn.calibration import CalibratedClassifierCV  # noqa: E402

from modeling_common import (CLASS_ORDER, HAS_FROZEN, RESULTS_DIR,  # noqa: E402
                             floor_probabilities, prepare_matrices, task_frame)
from model_zoo import LabelEncodedClassifier, classifier_zoo, regressor_zoo  # noqa: E402
from tuning import load_best_params  # noqa: E402

if HAS_FROZEN:
    from sklearn.frozen import FrozenEstimator

MARGIN_CLIP = (-5.0, 5.0)
TOP_SHAP = 5
ZOO_TASK = {"C": "C", "R": "R", "Lc": "L", "Lr": "L"}
# TreeSHAP needs a tree ensemble; the served in-play classifier is the best of
# these on validation, mirroring shap_analysis.py.
TREE_MODELS = ["lightgbm", "xgboost", "gbm", "random_forest"]


def _proba(estimator, X):
    proba = estimator.predict_proba(X)
    classes = (list(estimator.encoder_.classes_)
               if isinstance(estimator, LabelEncodedClassifier)
               else [str(c) for c in estimator.classes_])
    aligned = np.zeros((proba.shape[0], len(CLASS_ORDER)))
    for j, label in enumerate(CLASS_ORDER):
        if label in classes:
            aligned[:, j] = proba[:, classes.index(label)]
    totals = aligned.sum(axis=1, keepdims=True)
    totals[totals == 0] = 1.0
    return aligned / totals


def _pick_model(task, allowed=None):
    """Best sweep model for the task (validation-ranked sweep output)."""
    results = pd.read_csv(RESULTS_DIR / "model_results.csv", encoding="utf-8")
    metric = "rps" if task in {"C", "Lc"} else "mae"
    subset = results[(results["task"] == task) & (results["model"] != "dummy")]
    if allowed is not None:
        subset = subset[subset["model"].isin(allowed)]
    subset = subset.dropna(subset=[metric]).sort_values(metric)
    assert len(subset), f"no candidate model for task {task}"
    return str(subset["model"].iloc[0])


class TaskModel:
    """One fitted task: estimator, calibrator, and the frozen transform."""

    def __init__(self, task, model_name):
        self.task = task
        self.model_name = model_name
        self.is_classification = task in {"C", "Lc"}
        df, continuous, nominal, target, task_type = task_frame(task)
        self.matrices = prepare_matrices(df, continuous, nominal, target,
                                         task_type)
        tuned = load_best_params().get(task, {})
        zoo = (classifier_zoo if self.is_classification else regressor_zoo)(
            0, task=ZOO_TASK[task], tuned=tuned)
        self.estimator = zoo[model_name]()
        y = self.matrices["y_train"]
        self.estimator.fit(self.matrices["X_train"],
                           y if self.is_classification else y.astype(float))
        self.calibrator = None
        if self.is_classification and HAS_FROZEN:
            for method in ["isotonic", "sigmoid"]:
                try:
                    calibrated = CalibratedClassifierCV(
                        FrozenEstimator(self.estimator), method=method)
                    calibrated.fit(self.matrices["X_val"],
                                   self.matrices["y_val"])
                    self.calibrator = calibrated
                    self.calibration = method
                    break
                except Exception:
                    continue
        if self.is_classification and self.calibrator is None:
            self.calibration = "uncalibrated"
        # Precompute the design matrix of every servable (test) row.
        self.frame = df[df["split"] == "test"].reset_index(drop=True)
        self.X = self.matrices["transform"](self.frame)
        self.feature_names = self.matrices["feature_names"]

    def rows_for(self, match_id):
        return np.flatnonzero(self.frame["match_id"] == match_id)

    def probabilities(self, index):
        source = self.calibrator or self.estimator
        return floor_probabilities(_proba(source, self.X[index]))

    def margin(self, index):
        return np.clip(self.estimator.predict(self.X[index]), *MARGIN_CLIP)


class ServiceState:
    def __init__(self):
        print("Startup: fitting the four serving models "
              "(everything after this is cached)...")
        self.models = {
            "C": TaskModel("C", _pick_model("C")),
            "R": TaskModel("R", _pick_model("R")),
            "Lc": TaskModel("Lc", _pick_model("Lc", allowed=TREE_MODELS)),
            "Lr": TaskModel("Lr", _pick_model("Lr")),
        }
        inplay = self.models["Lc"]
        walkable = (inplay.estimator.estimator_
                    if isinstance(inplay.estimator, LabelEncodedClassifier)
                    else inplay.estimator)
        self.explainer = shap.TreeExplainer(walkable)
        # (match_id, minute) -> row index in the Lc/Lr test frames (identical
        # frames by construction: same source table, same split filter).
        pd.testing.assert_frame_equal(
            inplay.frame[["match_id", "snapshot_minute"]],
            self.models["Lr"].frame[["match_id", "snapshot_minute"]])
        self.snapshot_index = {
            (int(r.match_id), int(r.snapshot_minute)): i
            for i, r in enumerate(inplay.frame.itertuples())}
        self.prematch_index = {
            int(r.match_id): i
            for i, r in enumerate(self.models["C"].frame.itertuples())}
        self.minutes = sorted(inplay.frame["snapshot_minute"].unique().tolist())
        for task, model in self.models.items():
            print(f"  serving {task}: {model.model_name}"
                  + (f" ({model.calibration})" if model.is_classification
                     else ""))
        print(f"Serving {len(self.prematch_index)} held-out matches.")

    def top_attributions(self, index):
        """Top-|SHAP| features for the home-win probability at one snapshot."""
        inplay = self.models["Lc"]
        values = np.asarray(self.explainer.shap_values(inplay.X[index]))
        if values.ndim == 3:                      # (rows, features, classes)
            walkable = (inplay.estimator.estimator_
                        if isinstance(inplay.estimator, LabelEncodedClassifier)
                        else inplay.estimator)
            classes = (list(inplay.estimator.encoder_.classes_)
                       if isinstance(inplay.estimator, LabelEncodedClassifier)
                       else [str(c) for c in walkable.classes_])
            values = values[0, :, classes.index("H")]
        else:
            values = values[0]
        order = np.argsort(-np.abs(values))[:TOP_SHAP]
        return [{"feature": inplay.feature_names[i],
                 "value": round(float(inplay.X[index][0, i]), 4),
                 "shap": round(float(values[i]), 5)} for i in order]


app = FastAPI(title="Forecasting Competitive Football - in-play service")
state = None


@app.on_event("startup")
def _startup():
    global state
    state = ServiceState()


@app.get("/health")
def health():
    return {"status": "ok", "matches": len(state.prematch_index)}


@app.get("/matches")
def matches():
    frame = state.models["C"].frame
    return [{"match_id": int(r.match_id),
             "competition": str(r.competition_name),
             "date": str(r.match_date),
             "final_result": str(r.label_result),
             "final_margin": float(r.label_margin)}
            for r in frame.itertuples()]


def _snapshot_payload(match_id, minute):
    key = (match_id, minute)
    if key not in state.snapshot_index:
        raise HTTPException(status_code=404, detail=(
            f"match {match_id} minute {minute} is not servable; held-out "
            f"matches only, minutes {state.minutes}"))
    index = [state.snapshot_index[key]]
    inplay_c, inplay_r = state.models["Lc"], state.models["Lr"]
    probs = inplay_c.probabilities(index)[0]
    row = inplay_c.frame.iloc[index[0]]
    return {
        "match_id": match_id, "minute": minute, "state": "in_play",
        "models": {"outcome": inplay_c.model_name,
                   "margin": inplay_r.model_name},
        "probabilities": {c: round(float(p), 4)
                          for c, p in zip(CLASS_ORDER, probs)},
        "expected_margin": round(float(inplay_r.margin(index)[0]), 3),
        "score": {"home": int(row["inplay_home_goals"]),
                  "away": int(row["inplay_away_goals"])},
        "man_advantage": int(row["inplay_man_advantage"]),
        "top_shap": state.top_attributions(index),
    }


def _prematch_payload(match_id):
    if match_id not in state.prematch_index:
        raise HTTPException(status_code=404,
                            detail=f"match {match_id} is not a held-out match")
    index = [state.prematch_index[match_id]]
    pre_c, pre_r = state.models["C"], state.models["R"]
    probs = pre_c.probabilities(index)[0]
    return {
        "match_id": match_id, "minute": None, "state": "pre_match",
        "models": {"outcome": pre_c.model_name, "margin": pre_r.model_name},
        "probabilities": {c: round(float(p), 4)
                          for c, p in zip(CLASS_ORDER, probs)},
        "expected_margin": round(float(pre_r.margin(index)[0]), 3),
        "score": {"home": 0, "away": 0},
    }


@app.get("/predict")
def predict(match_id: int, minute: int = -1):
    if minute < 0:
        return _prematch_payload(match_id)
    return _snapshot_payload(match_id, minute)


@app.get("/replay/{match_id}")
def replay(match_id: int):
    pre = _prematch_payload(match_id)
    return {"match_id": match_id, "pre_match": pre,
            "snapshots": [_snapshot_payload(match_id, m)
                          for m in state.minutes]}


@app.get("/")
def dashboard():
    return FileResponse(HERE / "dashboard.html")


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="127.0.0.1", port=int(os.environ.get("PORT", 8100)),
                log_level="warning")
