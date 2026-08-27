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
snapshot's design matrix precomputed. On top of that, the first request for a
match builds all of that match's payloads in one batched pass (one
predict_proba, one predict, one TreeSHAP call over all its snapshot rows) and
caches them; every later request for the same match is a dictionary lookup.

The batching matters because a scikit-learn ensemble's predict cost is
dominated by fixed per-call overhead, not by row count: one call for 19 rows
costs little more than one call for 1 row. Serving a replay a row at a time
paid that overhead 19 times over, twice (once in /replay, once per tick).

Only test-split matches are servable: the demo must run on held-out data.

    python src/service/app.py            # uvicorn on 127.0.0.1:5500 (PORT env)
"""

from pathlib import Path
import os
import sys
import threading

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
# The only in-play feature columns the snapshot payload reads back.
SNAPSHOT_COLUMNS = ["inplay_home_goals", "inplay_away_goals",
                    "inplay_man_advantage", "inplay_home_xg", "inplay_away_xg",
                    "inplay_shot_diff", "inplay_sot_diff",
                    "inplay_corner_diff", "inplay_card_diff",
                    "inplay_pressure_diff", "inplay_momentum_xg_diff"]

_MODEL_RESULTS = None


def _model_results():
    """model_results.csv, read once instead of once per served task."""
    global _MODEL_RESULTS
    if _MODEL_RESULTS is None:
        _MODEL_RESULTS = pd.read_csv(RESULTS_DIR / "model_results.csv",
                                     encoding="utf-8")
    return _MODEL_RESULTS


def _dense_row(block, k):
    """Row k of a design-matrix block as a 1-D array (dense or sparse)."""
    row = block[k]
    if hasattr(row, "toarray"):
        return np.asarray(row.toarray()).ravel()
    return np.asarray(row).ravel()


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
    results = _model_results()
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
        self.minutes = sorted(
            inplay.frame["snapshot_minute"].unique().tolist())
        store = pd.read_csv(SRC / "reports" / "processed" / "match_store.csv",
                            encoding="utf-8")
        self.match_meta = {
            int(r.match_id): {"home_team": str(r.home_team),
                              "away_team": str(r.away_team),
                              "kick_off": str(r.kick_off)}
            for r in store.itertuples()}
        # Payload caches. Payloads depend only on fitted models and frozen
        # feature rows, so they are computed once per match and reused.
        self._lock = threading.Lock()
        self._snap_cache = {}
        self._pre_cache = {}
        self._match_list = None
        for task, model in self.models.items():
            print(f"  serving {task}: {model.model_name}"
                  + (f" ({model.calibration})" if model.is_classification
                     else ""))
        print(f"Serving {len(self.prematch_index)} held-out matches.")

    def attributions(self, rows):
        """Top-|SHAP| features for the home-win probability, one list per row.

        One TreeSHAP call for the whole block. TreeSHAP scales close to
        linearly in rows, so the saving here is modest; the batching matters
        because it lets the whole match be built in a single pass.
        """
        inplay = self.models["Lc"]
        block = inplay.X[rows]
        values = np.asarray(self.explainer.shap_values(block))
        if values.ndim == 3:
            walkable = (inplay.estimator.estimator_
                        if isinstance(inplay.estimator, LabelEncodedClassifier)
                        else inplay.estimator)
            classes = (list(inplay.estimator.encoder_.classes_)
                       if isinstance(inplay.estimator, LabelEncodedClassifier)
                       else [str(c) for c in walkable.classes_])
            home = classes.index("H")
            # Newer shap returns (rows, features, classes); older versions
            # return a list of per-class (rows, features) arrays.
            if values.shape[-1] == len(classes) and values.shape[0] == len(rows):
                values = values[:, :, home]
            else:
                values = values[home]
        out = []
        for k in range(len(rows)):
            row_values = values[k]
            row_features = _dense_row(block, k)
            order = np.argsort(-np.abs(row_values))[:TOP_SHAP]
            out.append([{"feature": inplay.feature_names[i],
                         "value": round(float(row_features[i]), 4),
                         "shap": round(float(row_values[i]), 5)}
                        for i in order])
        return out

    def _build_snapshots(self, match_id):
        """Every servable snapshot payload for one match, in one pass."""
        inplay_c, inplay_r = self.models["Lc"], self.models["Lr"]
        minutes = [m for m in self.minutes
                   if (match_id, m) in self.snapshot_index]
        if not minutes:
            return {}
        rows = [self.snapshot_index[(match_id, m)] for m in minutes]
        probs = inplay_c.probabilities(rows)
        margins = inplay_r.margin(rows)
        shap_rows = self.attributions(rows)
        meta = self.match_meta.get(match_id, {})
        models = {"outcome": inplay_c.model_name,
                  "margin": inplay_r.model_name,
                  "calibration": inplay_c.calibration}
        records = (inplay_c.frame.iloc[rows][SNAPSHOT_COLUMNS]
                   .to_dict("records"))
        payloads = {}
        for k, row in enumerate(records):
            payloads[minutes[k]] = {
                "match_id": match_id, "minute": minutes[k],
                "state": "in_play",
                **meta,
                "models": models,
                "probabilities": {c: round(float(p), 4)
                                  for c, p in zip(CLASS_ORDER, probs[k])},
                "expected_margin": round(float(margins[k]), 3),
                "score": {"home": int(row["inplay_home_goals"]),
                          "away": int(row["inplay_away_goals"])},
                "man_advantage": int(row["inplay_man_advantage"]),
                "stats": {
                    "xg": {"home": round(float(row["inplay_home_xg"]), 2),
                           "away": round(float(row["inplay_away_xg"]), 2)},
                    "shot_diff": int(row["inplay_shot_diff"]),
                    "sot_diff": int(row["inplay_sot_diff"]),
                    "corner_diff": int(row["inplay_corner_diff"]),
                    "card_diff": int(row["inplay_card_diff"]),
                    "pressure_diff": int(row["inplay_pressure_diff"]),
                    "momentum_xg_diff": round(
                        float(row["inplay_momentum_xg_diff"]), 3),
                },
                "top_shap": shap_rows[k],
            }
        return payloads

    def snapshots_for(self, match_id):
        cached = self._snap_cache.get(match_id)
        if cached is None:
            with self._lock:
                cached = self._snap_cache.get(match_id)
                if cached is None:
                    cached = self._build_snapshots(match_id)
                    self._snap_cache[match_id] = cached
        return cached

    def _build_prematch(self, match_id):
        index = [self.prematch_index[match_id]]
        pre_c, pre_r = self.models["C"], self.models["R"]
        probs = pre_c.probabilities(index)[0]
        row = pre_c.frame.iloc[index[0]]

        def _num(value, digits):
            value = float(value)
            return round(value, digits) if np.isfinite(value) else None

        def _form(side):
            return {"points": _num(row[f"{side}_form_points"], 2),
                    "xg_for": _num(row[f"{side}_form_xgf"], 2),
                    "xg_against": _num(row[f"{side}_form_xga"], 2),
                    "rest_days": _num(row[f"{side}_rest_days"], 1)}

        return {
            "match_id": match_id, "minute": None, "state": "pre_match",
            **self.match_meta.get(match_id, {}),
            "competition": str(row["competition_name"]),
            "date": str(row["match_date"]),
            "models": {"outcome": pre_c.model_name, "margin": pre_r.model_name,
                       "calibration": pre_c.calibration},
            "probabilities": {c: round(float(p), 4)
                              for c, p in zip(CLASS_ORDER, probs)},
            "expected_margin": round(float(pre_r.margin(index)[0]), 3),
            "score": {"home": 0, "away": 0},
            "form": {"home": _form("home"), "away": _form("away")},
        }

    def prematch_for(self, match_id):
        cached = self._pre_cache.get(match_id)
        if cached is None:
            with self._lock:
                cached = self._pre_cache.get(match_id)
                if cached is None:
                    cached = self._build_prematch(match_id)
                    self._pre_cache[match_id] = cached
        return cached

    def match_list(self):
        if self._match_list is None:
            frame = self.models["C"].frame
            self._match_list = [
                {"match_id": int(r.match_id),
                 "competition": str(r.competition_name),
                 "date": str(r.match_date),
                 **self.match_meta.get(int(r.match_id), {}),
                 "final_result": str(r.label_result),
                 "final_margin": float(r.label_margin)}
                for r in frame.itertuples()]
        return self._match_list


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
    return state.match_list()


@app.get("/minutes")
def snapshot_minutes():
    """The servable snapshot minutes, without computing any prediction."""
    return {"minutes": state.minutes}


def _snapshot_payload(match_id, minute):
    if match_id not in state.prematch_index:
        raise HTTPException(status_code=404,
                            detail=f"match {match_id} is not a held-out match")
    payloads = state.snapshots_for(match_id)
    if minute not in payloads:
        raise HTTPException(status_code=404, detail=(
            f"match {match_id} minute {minute} is not servable; held-out "
            f"matches only, minutes {state.minutes}"))
    return payloads[minute]


def _prematch_payload(match_id):
    if match_id not in state.prematch_index:
        raise HTTPException(status_code=404,
                            detail=f"match {match_id} is not a held-out match")
    return state.prematch_for(match_id)


@app.get("/predict")
def predict(match_id: int, minute: int = -1):
    if minute < 0:
        return _prematch_payload(match_id)
    return _snapshot_payload(match_id, minute)


@app.get("/replay/{match_id}")
def replay(match_id: int):
    pre = _prematch_payload(match_id)
    payloads = state.snapshots_for(match_id)
    return {"match_id": match_id, "pre_match": pre,
            "snapshots": [payloads[m] for m in state.minutes
                          if m in payloads]}


@app.get("/")
def dashboard():
    return FileResponse(HERE / "dashboard.html")


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="127.0.0.1", port=int(os.environ.get("PORT", 5500)),
                log_level="info", access_log=False)
