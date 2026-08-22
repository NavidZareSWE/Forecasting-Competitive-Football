"""Run from the repository root with:

    python src/models/tuning.py

    set TUNE_TASKS=C && python src/models/tuning.py    cmd.exe
    TUNE_TASKS=C python src/models/tuning.py           bash
"""

from pathlib import Path
import itertools
import json
import os
import sys
import time
import zlib

import numpy as np
import pandas as pd
from sklearn.model_selection import GroupKFold, KFold, StratifiedKFold

sys.path.insert(0, str(Path(__file__).resolve().parent))

from modeling_common import (task_frame, prepare_matrices, RESULTS_DIR,
                             CLASS_ORDER)
from model_zoo import (classifier_zoo, regressor_zoo, CLASSIFIER_SPACES,
                       REGRESSOR_SPACES, TASK_SUPPORT, _classifier_builders,
                       _regressor_builders)


N_CANDIDATES = 12
K_FOLDS = 3
SUBSAMPLE_ABOVE = 8000
MARGIN_CLIP = (-5.0, 5.0)
RANDOM_STATE = 0

TUNING_CSV = RESULTS_DIR / "tuning_results.csv"
BEST_PARAMS_JSON = RESULTS_DIR / "best_params.json"


def sample_candidates(space, n_candidates, seed):
    keys = sorted(space)
    full = [dict(zip(keys, values))
            for values in itertools.product(*(space[k] for k in keys))]
    if len(full) <= n_candidates:
        return full
    rng = np.random.default_rng(seed)
    index = rng.choice(len(full), size=n_candidates, replace=False)
    return [full[i] for i in sorted(index)]


def _log_loss(proba, y_true, classes, eps=1e-15):
    position = {label: i for i, label in enumerate(classes)}
    rows = np.arange(len(y_true))
    columns = np.array([position[label] for label in y_true])
    return float(-np.log(np.clip(proba[rows, columns], eps, 1.0)).mean())


def _folds(X, y, groups, task_type):
    if groups is not None:
        return list(GroupKFold(n_splits=K_FOLDS).split(X, y, groups))
    if task_type == "classification":
        return list(StratifiedKFold(n_splits=K_FOLDS, shuffle=True,
                                    random_state=RANDOM_STATE).split(X, y))
    return list(KFold(n_splits=K_FOLDS, shuffle=True,
                      random_state=RANDOM_STATE).split(X, y))


def _score_candidate(builder, params, X, y, folds, task_type):
    scores = []
    for train_index, val_index in folds:
        estimator = builder(RANDOM_STATE, **params)
        if task_type == "classification":
            estimator.fit(X[train_index], y[train_index])
            proba = estimator.predict_proba(X[val_index])
            classes = list(estimator.classes_)
            aligned = np.zeros((len(val_index), len(CLASS_ORDER)))
            for j, label in enumerate(CLASS_ORDER):
                if label in classes:
                    aligned[:, j] = proba[:, classes.index(label)]
            total = aligned.sum(axis=1, keepdims=True)
            total[total == 0] = 1.0
            scores.append(_log_loss(aligned / total, y[val_index], CLASS_ORDER))
        else:
            estimator.fit(X[train_index], y[train_index].astype(float))
            prediction = np.clip(estimator.predict(X[val_index]), *MARGIN_CLIP)
            scores.append(float(np.abs(prediction
                                       - y[val_index].astype(float)).mean()))
    return float(np.mean(scores))


def tune_task(task, rows):
    df, continuous, nominal, target, task_type = task_frame(task)
    matrices = prepare_matrices(df, continuous, nominal, target, task_type,
                                resampling="none")
    X, y = matrices["X_train"], matrices["y_train"]

    train_mask = (df["split"] == "train").to_numpy()
    groups = (df.loc[train_mask, "match_id"].to_numpy()
              if "snapshot_minute" in df.columns else None)

    subsampled = False
    if X.shape[0] > SUBSAMPLE_ABOVE:
        rng = np.random.default_rng(RANDOM_STATE)
        if groups is not None:
            matches = np.unique(groups)
            keep_share = SUBSAMPLE_ABOVE / X.shape[0]
            n_keep = min(len(matches),
                         max(2 * K_FOLDS, int(len(matches) * keep_share)))
            keep = rng.choice(matches, size=n_keep, replace=False)
            index = np.flatnonzero(np.isin(groups, keep))
            groups = groups[index]
        else:
            index = np.sort(rng.choice(X.shape[0], size=SUBSAMPLE_ABOVE,
                                       replace=False))
        X, y = X[index], y[index]
        subsampled = True

    folds = _folds(X, y, groups, task_type)
    is_classification = task_type == "classification"
    spaces = CLASSIFIER_SPACES if is_classification else REGRESSOR_SPACES
    builders = _classifier_builders() if is_classification else _regressor_builders()
    support_key = "L" if task in {"Lc", "Lr"} else task

    best = {}
    for name, space in spaces.items():
        if name not in builders:
            print(f"  [{task}] {name}: unavailable, skipped")
            continue
        if support_key not in TASK_SUPPORT.get(name, set()):
            continue
        seed = zlib.crc32(f"{name}|{task}".encode("utf-8"))
        candidates = sample_candidates(space, N_CANDIDATES, seed)
        started = time.perf_counter()
        scored = []
        for params in candidates:
            score = _score_candidate(builders[name], params, X, y, folds,
                                     task_type)
            scored.append((score, params))
            rows.append({"task": task, "model": name,
                         "n_candidates": len(candidates), "k_folds": K_FOLDS,
                         "n_tuning_rows": int(X.shape[0]),
                         "subsampled": subsampled,
                         "objective": "log_loss" if is_classification else "mae",
                         "score": round(score, 6),
                         "params": json.dumps(params, sort_keys=True)})
        scored.sort(key=lambda pair: pair[0])
        best[name] = scored[0][1]
        elapsed = time.perf_counter() - started
        print(f"  [{task}] {name:18s} {len(candidates):2d} configs x {K_FOLDS} folds"
              f"  best={scored[0][0]:.5f}  ({elapsed:.1f}s)  {scored[0][1]}")
    return best


def save_progress(rows, best):
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(TUNING_CSV, index=False, encoding="utf-8")
    BEST_PARAMS_JSON.write_text(json.dumps(best, indent=2, sort_keys=True),
                                encoding="utf-8")


def main():
    tasks = [t for t in os.environ.get("TUNE_TASKS", "C,R,Lc,Lr").split(",")
             if t.strip()]
    resume = os.environ.get("TUNE_RESUME", "1") != "0"

    rows = []
    best = {}
    if resume and BEST_PARAMS_JSON.exists():
        try:
            best = json.loads(BEST_PARAMS_JSON.read_text(encoding="utf-8"))
        except (OSError, ValueError):
            best = {}
        if TUNING_CSV.exists():
            try:
                rows = pd.read_csv(TUNING_CSV,
                                   encoding="utf-8").to_dict("records")
            except (OSError, ValueError):
                rows = []
        done = [t for t in tasks if best.get(t)]
        if done:
            print(f"Resuming: tasks {done} already searched, skipping them. "
                  f"Set TUNE_RESUME=0 to search everything again.")

    for task in tasks:
        if resume and best.get(task):
            continue
        print(f"\n=== tuning {task} ===")
        best[task] = tune_task(task, rows)
        save_progress(rows, best)
        print(f"  saved progress after task {task}")

    save_progress(rows, best)

    audit = pd.DataFrame(rows)
    for task, group in audit.groupby("task"):
        counts = group.groupby("model").size()
        budget = {str(model): int(count) for model, count in counts.items()}
        assert counts.nunique() == 1, \
            f"unequal tuning budget on task {task}: {budget}"

    print(f"\nWrote {len(rows)} candidate evaluations -> {TUNING_CSV}")
    print(f"Wrote selected configurations -> {BEST_PARAMS_JSON}")


def load_best_params():
    if not BEST_PARAMS_JSON.exists():
        return {}
    return json.loads(BEST_PARAMS_JSON.read_text(encoding="utf-8"))


if __name__ == "__main__":
    main()
