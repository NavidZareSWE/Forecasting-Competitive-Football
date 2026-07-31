import numpy as np
import pandas as pd

from modeling_common import (
    prepare_matrices, classification_metrics, per_class_metrics,
    regression_metrics, calibrate_proba, fit_with_cost, floor_probabilities,
    _proba_in_order, CLASS_ORDER,
)


MARGIN_CLIP = (-5, 5)


def evaluate_classification(model_name, factory, data, resampling="none"):
    """Fit, calibrate, score. Returns (metrics_row, per-row test predictions)."""
    matrices = prepare_matrices(*data, resampling=resampling)
    estimator = factory()
    train_seconds, peak_mb = fit_with_cost(
        estimator, matrices["X_train"], matrices["y_train"])

    # Floored like the calibrated output, so before/after differ by the calibrator.
    raw = floor_probabilities(
        _proba_in_order(estimator, matrices["X_test"], CLASS_ORDER))
    before = classification_metrics(raw, matrices["y_test"])
    calibrated, method = calibrate_proba(
        estimator, matrices["X_val"], matrices["y_val"], matrices["X_test"])
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
    return row, predictions


def evaluate_regression(model_name, factory, data, resampling="none"):
    # Resampling is a classification study (brief 7.1).
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
    # The exact kernel subsamples; the compute table has to say so.
    if hasattr(estimator, "n_train_used_"):
        row["n_train"] = int(estimator.n_train_used_)
        row["subsampled"] = bool(estimator.subsampled_)
    # Which lambda the paper method's inner CV chose.
    if hasattr(estimator, "lam_"):
        row["hs_lambda"] = float(estimator.lam_)

    frame = matrices["meta_test"].copy()
    frame["model"] = model_name
    frame["resampling"] = "none"
    frame["y_true"] = matrices["y_test"].astype(float)
    frame["y_pred"] = predictions_array
    return row, frame
