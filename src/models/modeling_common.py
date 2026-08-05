from pathlib import Path
import sys
import threading
import time

import numpy as np
import pandas as pd
import psutil
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.calibration import CalibratedClassifierCV

try:
    from sklearn.frozen import FrozenEstimator
    HAS_FROZEN = True
except Exception:
    HAS_FROZEN = False


HERE = Path(__file__).resolve().parent
PROJECT = HERE.parent
FEATURE_DIR = PROJECT / "reports" / "features"
RESULTS_DIR = PROJECT / "reports"

PAPERS_DIR = PROJECT / "papers"
sys.path.insert(0, str(PAPERS_DIR))

CLASS_ORDER = ["H", "D", "A"]
NOMINAL_COLUMNS = ["competition_name"]
META_COLUMNS = {"match_id", "match_date", "split", "label_result",
                "label_margin", "competition_name", "snapshot_minute"}


def load_prematch():
    path = FEATURE_DIR / "prematch_features.csv"
    if not path.exists():
        raise FileNotFoundError(
            f"Missing {path}. Run build_prematch_features.py first.")
    return pd.read_csv(path, encoding="utf-8")


def load_inplay():
    prematch = load_prematch().drop(columns=["split", "label_result",
                                             "label_margin", "match_date"])
    path = FEATURE_DIR / "inplay_features.csv"
    if not path.exists():
        raise FileNotFoundError(
            f"Missing {path}. Run build_inplay_features.py first.")
    inplay = pd.read_csv(path, encoding="utf-8")
    return inplay.merge(prematch, on="match_id", how="inner")


def task_frame(task):
    if task in {"C", "R"}:
        df = load_prematch()
    elif task in {"Lc", "Lr"}:
        df = load_inplay()
    else:
        raise ValueError(f"Unknown task {task}")

    target = "label_result" if task in {"C", "Lc"} else "label_margin"
    task_type = "classification" if task in {"C", "Lc"} else "regression"
    feature_cols = [
        c for c in df.columns if c not in META_COLUMNS or c == "snapshot_minute"]
    nominal_cols = [c for c in NOMINAL_COLUMNS if c in df.columns]
    continuous_cols = [c for c in feature_cols if c not in nominal_cols]
    return df, continuous_cols, nominal_cols, target, task_type


# Brief 7.1 arms. class_weight is an estimator argument, not a resampler.
RESAMPLERS = ("none", "p1", "smote", "borderline_smote", "adasyn")


def _apply_p1(x_mixed, y, n_continuous, random_state):
    from g_smotenc import GSMOTENC
    categorical = list(range(n_continuous, x_mixed.shape[1]))
    sampler = GSMOTENC(categorical_features=categorical, k_neighbors=5,
                       selection_strategy="combined", random_state=random_state)
    return sampler.fit_resample(x_mixed, y)


def _apply_imblearn(name, X, y, random_state):
    from imblearn.over_sampling import ADASYN, BorderlineSMOTE, SMOTE
    samplers = {"smote": SMOTE, "borderline_smote": BorderlineSMOTE,
                "adasyn": ADASYN}
    return samplers[name](random_state=random_state).fit_resample(X, y)


def prepare_matrices(df, continuous_cols, nominal_cols, target, task_type,
                     resampling="none", random_state=0):
    if resampling not in RESAMPLERS:
        raise ValueError(f"Unknown resampling arm {resampling!r}; "
                         f"expected one of {RESAMPLERS}")
    # Brief 7.2: never oversample the snapshot table across matches.
    if resampling != "none" and "snapshot_minute" in df.columns:
        raise ValueError(
            f"Resampling arm {resampling!r} requested on the in-play snapshot "
            "table; oversampling across matches is forbidden (brief 7.2). "
            "Resample the pre-match table only.")

    parts = {}
    for name in ["train", "validation", "test"]:
        subset = df[df["split"] == name]
        parts[name] = subset

    imputer = SimpleImputer(strategy="median")
    scaler = StandardScaler()
    train_cont = imputer.fit_transform(parts["train"][continuous_cols])
    train_cont = scaler.fit_transform(train_cont)

    train_nom = parts["train"][nominal_cols].astype(str).to_numpy() \
        if nominal_cols else np.empty((len(parts["train"]), 0), dtype=object)
    y_train = parts["train"][target].to_numpy()

    # Train rows only; validation and test are never resampled.
    if resampling == "p1" and task_type == "classification":
        n_cont = train_cont.shape[1]
        mixed = np.hstack([train_cont.astype(object), train_nom])
        mixed, y_train = _apply_p1(mixed, y_train, n_cont, random_state)
        train_cont = mixed[:, :n_cont].astype(float)
        train_nom = mixed[:, n_cont:]
    elif resampling in {"smote", "borderline_smote", "adasyn"} \
            and task_type == "classification":
        # imbalanced-learn is numeric-only: one-hot, then snap back to one category.
        n_cont = train_cont.shape[1]
        pre_encoder = OneHotEncoder(
            handle_unknown="ignore", sparse_output=False)
        nom_encoded = (pre_encoder.fit_transform(train_nom) if nominal_cols
                       else np.empty((train_cont.shape[0], 0)))
        combined, y_train = _apply_imblearn(
            resampling, np.hstack([train_cont, nom_encoded]), y_train,
            random_state)
        train_cont = combined[:, :n_cont]
        if nominal_cols:
            assert len(nominal_cols) == 1, \
                "one-hot snap-back assumes a single nominal column"
            categories = pre_encoder.categories_[0]
            winners = combined[:, n_cont:].argmax(axis=1)
            train_nom = categories[winners].reshape(-1, 1)
        else:
            train_nom = np.empty((train_cont.shape[0], 0), dtype=object)

    encoder = OneHotEncoder(handle_unknown="ignore", sparse_output=False)
    if nominal_cols:
        train_nom_enc = encoder.fit_transform(train_nom)
    else:
        train_nom_enc = np.empty((train_cont.shape[0], 0))

    def transform(subset):
        cont = scaler.transform(imputer.transform(subset[continuous_cols]))
        if nominal_cols:
            nom = encoder.transform(
                subset[nominal_cols].astype(str).to_numpy())
        else:
            nom = np.empty((len(subset), 0))
        return np.hstack([cont, nom])

    def meta(subset):
        # Row identity travels with the matrices so downstream analyses can join back.
        columns = [c for c in ["match_id", "snapshot_minute", "match_date",
                               "competition_name"] if c in subset.columns]
        return subset[columns].reset_index(drop=True)
        # SHAP and the ablation need the design-matrix column order by name.
    encoded_names = (list(encoder.get_feature_names_out(nominal_cols))
                     if nominal_cols else [])
    feature_names = list(continuous_cols) + encoded_names

    return {
        "X_train": np.hstack([train_cont, train_nom_enc]),
        "y_train": y_train,
        "X_val": transform(parts["validation"]),
        "y_val": parts["validation"][target].to_numpy(),
        "X_test": transform(parts["test"]),
        "y_test": parts["test"][target].to_numpy(),
        "meta_val": meta(parts["validation"]),
        "meta_test": meta(parts["test"]),
        "feature_names": feature_names,
        "transform": transform,
        "continuous_cols": list(continuous_cols),
        "nominal_cols": list(nominal_cols),
    }


def fit_with_cost(estimator, X, y):
    # RSS sampled from a thread: a kernel fit peaks inside one BLAS call.
    process = psutil.Process()
    baseline = process.memory_info().rss
    peak = baseline
    stop = threading.Event()

    def sample():
        nonlocal peak
        while not stop.wait(0.01):
            peak = max(peak, process.memory_info().rss)

    sampler = threading.Thread(target=sample, daemon=True)
    sampler.start()
    start = time.perf_counter()
    try:
        estimator.fit(X, y)
    finally:
        stop.set()
        sampler.join()
    seconds = time.perf_counter() - start
    peak = max(peak, process.memory_info().rss)
    return seconds, (peak - baseline) / (1024 ** 2)


def _onehot(labels, order):
    index = {c: i for i, c in enumerate(order)}
    matrix = np.zeros((len(labels), len(order)))
    for row, label in enumerate(labels):
        matrix[row, index[label]] = 1.0
    return matrix


def ranked_probability_score(proba, y_true, order=CLASS_ORDER):
    observed = _onehot(y_true, order)
    cum_p = np.cumsum(proba, axis=1)
    cum_o = np.cumsum(observed, axis=1)
    return float((((cum_p - cum_o) ** 2)[:, :-1].sum(axis=1) / (len(order) - 1)).mean())


def brier_multiclass(proba, y_true, order=CLASS_ORDER):
    observed = _onehot(y_true, order)
    return float(((proba - observed) ** 2).sum(axis=1).mean())


def log_loss_safe(proba, y_true, order=CLASS_ORDER, eps=1e-15):
    observed = _onehot(y_true, order)
    clipped = np.clip(proba, eps, 1.0)
    return float(-(observed * np.log(clipped)).sum(axis=1).mean())


def expected_calibration_error(proba, y_true, order=CLASS_ORDER, bins=10):
    confidence = proba.max(axis=1)
    predicted = np.array([order[i] for i in proba.argmax(axis=1)])
    correct = (predicted == np.asarray(y_true)).astype(float)
    edges = np.linspace(0.0, 1.0, bins + 1)
    ece = 0.0
    for lo, hi in zip(edges[:-1], edges[1:]):
        mask = (confidence > lo) & (confidence <= hi)
        if mask.any():
            ece += mask.mean() * \
                abs(correct[mask].mean() - confidence[mask].mean())
    return float(ece)


def classification_metrics(proba, y_true, order=CLASS_ORDER):
    return {
        "rps": ranked_probability_score(proba, y_true, order),
        "log_loss": log_loss_safe(proba, y_true, order),
        "brier": brier_multiclass(proba, y_true, order),
        "ece": expected_calibration_error(proba, y_true, order),
    }


def per_class_metrics(proba, y_true, order=CLASS_ORDER):
    # Aggregates hide the draw class, which is the one every resampling arm moves.
    predicted = np.array([order[i] for i in proba.argmax(axis=1)])
    y_true = np.asarray(y_true)
    scores = {}
    for label in order:
        true_positive = int(((predicted == label) & (y_true == label)).sum())
        false_positive = int(((predicted == label) & (y_true != label)).sum())
        false_negative = int(((predicted != label) & (y_true == label)).sum())
        precision = (true_positive / (true_positive + false_positive)
                     if true_positive + false_positive else 0.0)
        recall = (true_positive / (true_positive + false_negative)
                  if true_positive + false_negative else 0.0)
        f1 = (2 * precision * recall / (precision + recall)
              if precision + recall else 0.0)
        scores[f"precision_{label}"] = round(precision, 5)
        scores[f"recall_{label}"] = round(recall, 5)
        scores[f"f1_{label}"] = round(f1, 5)
        scores[f"support_{label}"] = int((y_true == label).sum())
    return scores


def regression_metrics(y_pred, y_true):
    y_pred = np.asarray(y_pred, dtype=float)
    y_true = np.asarray(y_true, dtype=float)
    mae = float(np.abs(y_pred - y_true).mean())
    rmse = float(np.sqrt(((y_pred - y_true) ** 2).mean()))
    if y_pred.std() < 1e-12 or y_true.std() < 1e-12:
        corr = 0.0
    else:
        corr = float(np.corrcoef(y_pred, y_true)[0, 1])
    return {"mae": mae, "rmse": rmse, "corr": corr}


# Isotonic on ~340 validation rows emits exact zeros; floor them before log-loss.
PROBABILITY_FLOOR = 0.005


def floor_probabilities(proba, floor=PROBABILITY_FLOOR):
    floored = np.clip(proba, floor, 1.0)
    return floored / floored.sum(axis=1, keepdims=True)


def calibrate_proba(estimator, X_val, y_val, X_test, order=CLASS_ORDER):
    raw = _proba_in_order(estimator, X_test, order)
    if not HAS_FROZEN:
        return floor_probabilities(raw), "uncalibrated"
    for method in ["isotonic", "sigmoid"]:
        try:
            calibrated = CalibratedClassifierCV(
                FrozenEstimator(estimator), method=method)
            calibrated.fit(X_val, y_val)
            return (floor_probabilities(_proba_in_order(calibrated, X_test, order)),
                    method)
        except Exception:
            continue
    return floor_probabilities(raw), "uncalibrated"


def _proba_in_order(estimator, X, order):
    proba = estimator.predict_proba(X)
    classes = list(estimator.classes_)
    aligned = np.zeros((proba.shape[0], len(order)))
    for j, label in enumerate(order):
        if label in classes:
            aligned[:, j] = proba[:, classes.index(label)]
    row_sums = aligned.sum(axis=1, keepdims=True)
    row_sums[row_sums == 0] = 1.0
    return aligned / row_sums
