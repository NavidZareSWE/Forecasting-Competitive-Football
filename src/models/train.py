from pathlib import Path
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))

from modeling_common import (
    prepare_matrices, classification_metrics, per_class_metrics,
    regression_metrics, fit_calibrator, fit_with_cost, floor_probabilities,
    _proba_in_order, CLASS_ORDER,
)


MARGIN_CLIP = (-5, 5)


def evaluate_classification(model_name, factory, data, resampling="none",
                            with_reference=False):
    matrices = prepare_matrices(*data, resampling=resampling)
    estimator = factory()
    train_seconds, peak_mb = fit_with_cost(
        estimator, matrices["X_train"], matrices["y_train"])

    raw = floor_probabilities(
        _proba_in_order(estimator, matrices["X_test"], CLASS_ORDER))
    before = classification_metrics(raw, matrices["y_test"])
    calibrate, method = fit_calibrator(
        estimator, matrices["X_val"], matrices["y_val"])
    calibrated = calibrate(matrices["X_test"])
    after = classification_metrics(calibrated, matrices["y_test"])

    row = {
        "model": model_name, "resampling": resampling,
        "train_seconds": round(train_seconds, 3),
        "peak_memory_mb": round(peak_mb, 1),
        "n_train": int(matrices["X_train"].shape[0]),
        "n_test": int(matrices["X_test"].shape[0]),
        "calibration": method,
        "rps": round(after["rps"], 5), "log_loss": round(after["log_loss"], 5),
        "brier": round(after["brier"], 5),
        "rps_before": round(before["rps"], 5),
        "ece_before": round(before["ece"], 5), "ece_after": round(after["ece"], 5),
    }
    row.update(per_class_metrics(calibrated, matrices["y_test"]))
    if hasattr(estimator, "lam_"):
        row["hs_lambda"] = float(estimator.lam_)

    predictions = matrices["meta_test"].copy()
    predictions["model"] = model_name
    predictions["resampling"] = resampling
    predictions["y_true"] = matrices["y_test"]
    for position, label in enumerate(CLASS_ORDER):
        predictions[f"p_{label}"] = calibrated[:, position]
        predictions[f"raw_{label}"] = raw[:, position]
    if not with_reference:
        return row, predictions
    return row, predictions, _reference_frame(
        matrices, model_name, resampling, calibrate)


def _reference_frame(matrices, model_name, resampling, calibrate):
    """Calibrated probabilities on the excluded split.

    These rows are what the frozen pre-match reference curve is built
    from. The extended pre-match test split and the StatsBomb in-play
    test split share no match_id, so without them the reference series
    is empty and Task L has no baseline to be measured against.
    """
    if matrices["X_reference"] is None:
        return None
    proba = calibrate(matrices["X_reference"])
    frame = matrices["meta_reference"].copy()
    frame["model"] = model_name
    frame["resampling"] = resampling
    frame["y_true"] = matrices["y_reference"]
    for position, label in enumerate(CLASS_ORDER):
        frame[f"p_{label}"] = proba[:, position]
    return frame


def evaluate_regression(model_name, factory, data, resampling="none",
                        with_reference=False):
    matrices = prepare_matrices(*data, resampling="none")
    estimator = factory()
    train_seconds, peak_mb = fit_with_cost(
        estimator, matrices["X_train"], matrices["y_train"].astype(float))

    predictions_array = np.clip(estimator.predict(
        matrices["X_test"]), MARGIN_CLIP[0], MARGIN_CLIP[1])
    metrics = regression_metrics(predictions_array, matrices["y_test"])
    row = {
        "model": model_name, "resampling": "none",
        "train_seconds": round(train_seconds, 3),
        "peak_memory_mb": round(peak_mb, 1),
        "n_train": int(matrices["X_train"].shape[0]),
        "n_test": int(matrices["X_test"].shape[0]),
        "calibration": "n/a",
        "mae": round(metrics["mae"], 5), "rmse": round(metrics["rmse"], 5),
        "corr": round(metrics["corr"], 5),
    }
    if hasattr(estimator, "n_train_used_"):
        row["n_train"] = int(estimator.n_train_used_)
        row["subsampled"] = bool(estimator.subsampled_)
    if hasattr(estimator, "lam_"):
        row["hs_lambda"] = float(estimator.lam_)

    frame = matrices["meta_test"].copy()
    frame["model"] = model_name
    frame["resampling"] = "none"
    frame["y_true"] = matrices["y_test"].astype(float)
    frame["y_pred"] = predictions_array
    if not with_reference:
        return row, frame

    reference = None
    if matrices["X_reference"] is not None:
        reference = matrices["meta_reference"].copy()
        reference["model"] = model_name
        reference["resampling"] = "none"
        reference["y_true"] = matrices["y_reference"].astype(float)
        reference["y_pred"] = np.clip(
            estimator.predict(matrices["X_reference"]),
            MARGIN_CLIP[0], MARGIN_CLIP[1])
    return row, frame, reference
