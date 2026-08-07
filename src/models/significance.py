"""Phase 7: statistical tests across repeated experiments.

Two independent sources of repetition, because a single point estimate is not
evidence:

  1. Seed repetition - every model refitted on the fixed splits under N_SEEDS
     random states. This measures STABILITY, not superiority: the only thing
     varying is the random state, so a deterministic model has zero spread and
     a paired t-test over seeds divides by ~0 and reports p ~ 0 for any gap,
     however small it is relative to match-level noise. These columns are
     therefore reported as stability diagnostics and are never used to claim
     that one model beats another.
  2. Match-clustered bootstrap of the evaluation set - snapshots of one match
     are not independent, so the bootstrap resamples matches, not rows. This
     is the ONLY test in this file that licenses a superiority claim, because
     it resamples the quantity that actually limits the conclusion: the finite
     set of 344 test matches.

Holm-Bonferroni controls the family-wise error rate within each task.

    python src/models/significance.py

N_SEEDS overrides the number of seed repetitions. `VAR=value command` is POSIX
shell syntax and fails on cmd.exe, so set it first on Windows:

    set N_SEEDS=3 && python src/models/significance.py    cmd.exe
    N_SEEDS=3 python src/models/significance.py           bash
"""

import os

import numpy as np
import pandas as pd
from scipy import stats

from modeling_common import (CLASS_ORDER, RESULTS_DIR, prepare_matrices,
                             task_frame, ranked_probability_score)
from model_zoo import classifier_zoo, regressor_zoo
from tuning import load_best_params


N_SEEDS = int(os.environ.get("N_SEEDS", "5"))
N_BOOTSTRAP = 2000
BOOTSTRAP_SEED = 20260808
ALPHA = 0.05
MARGIN_CLIP = (-5.0, 5.0)
ZOO_TASK = {"C": "C", "R": "R", "Lc": "L", "Lr": "L"}
TASKS = ["C", "R", "Lc", "Lr"]


# --- Per-row losses ---------------------------------------------------------
def row_losses(task):
    """Per-row loss from the sweep's stored test predictions, by model."""
    path = RESULTS_DIR / f"predictions_{task}.csv"
    if not path.exists():
        raise FileNotFoundError(f"Missing {path}. Run run_models.py first.")
    frame = pd.read_csv(path, encoding="utf-8")
    losses = {}
    for model, group in frame.groupby("model"):
        group = group.sort_values([c for c in ["match_id", "snapshot_minute"]
                                   if c in group.columns])
        if task in {"C", "Lc"}:
            proba = group[[f"p_{c}" for c in CLASS_ORDER]].to_numpy()
            y_true = group["y_true"].to_numpy()
            loss = np.array([ranked_probability_score(proba[[i]], y_true[[i]])
                             for i in range(len(group))])
        else:
            loss = np.abs(group["y_pred"].to_numpy()
                          - group["y_true"].to_numpy().astype(float))
        losses[model] = pd.Series(loss, index=group["match_id"].to_numpy())
    return losses


def clustered_bootstrap(loss_a, loss_b, n_bootstrap=N_BOOTSTRAP):
    """Resample matches, not rows: snapshots within a match are dependent."""
    frame = pd.DataFrame({"match_id": loss_a.index,
                          "a": loss_a.to_numpy(), "b": loss_b.to_numpy()})
    per_match = frame.groupby("match_id")[["a", "b"]].mean()
    difference = (per_match["a"] - per_match["b"]).to_numpy()
    observed = float(difference.mean())

    rng = np.random.default_rng(BOOTSTRAP_SEED)
    n = len(difference)
    draws = difference[rng.integers(0, n, size=(n_bootstrap, n))].mean(axis=1)
    # Two-sided p-value from the bootstrap distribution centred on zero.
    centred = draws - observed
    p_value = float((np.abs(centred) >= abs(observed)).mean())
    low, high = np.percentile(draws, [2.5, 97.5])
    return observed, float(low), float(high), p_value, per_match, n


def holm(p_values):
    """Holm-Bonferroni adjusted p-values, order preserved."""
    p_values = np.asarray(p_values, dtype=float)
    order = np.argsort(p_values)
    m = len(p_values)
    adjusted = np.empty(m)
    running = 0.0
    for rank, position in enumerate(order):
        value = (m - rank) * p_values[position]
        running = max(running, value)
        adjusted[position] = min(1.0, running)
    return adjusted


# --- Seed repetition --------------------------------------------------------
def seed_repetitions(task, rows):
    df, continuous, nominal, target, task_type = task_frame(task)
    tuned = load_best_params().get(task, {})
    is_classification = task in {"C", "Lc"}
    matrices = prepare_matrices(df, continuous, nominal, target, task_type)

    for seed in range(N_SEEDS):
        zoo = (classifier_zoo(seed, task=ZOO_TASK[task], tuned=tuned)
               if is_classification
               else regressor_zoo(seed, task=ZOO_TASK[task], tuned=tuned))
        for model, factory in zoo.items():
            estimator = factory()
            y_train = matrices["y_train"]
            estimator.fit(matrices["X_train"],
                          y_train if is_classification
                          else y_train.astype(float))
            if is_classification:
                proba = estimator.predict_proba(matrices["X_test"])
                classes = [str(c) for c in estimator.classes_]
                aligned = np.zeros((proba.shape[0], len(CLASS_ORDER)))
                for j, label in enumerate(CLASS_ORDER):
                    if label in classes:
                        aligned[:, j] = proba[:, classes.index(label)]
                totals = aligned.sum(axis=1, keepdims=True)
                totals[totals == 0] = 1.0
                metric = ranked_probability_score(aligned / totals,
                                                  matrices["y_test"])
                metric_name = "rps"
            else:
                prediction = np.clip(estimator.predict(matrices["X_test"]),
                                     *MARGIN_CLIP)
                metric = float(np.abs(prediction
                                      - matrices["y_test"].astype(float)).mean())
                metric_name = "mae"
            rows.append({"task": task, "model": model, "seed": seed,
                         "metric": metric_name, "value": round(metric, 6)})
        print(f"  [{task}] seed {seed} done")


def seed_tests(seed_frame, task):
    subset = seed_frame[seed_frame["task"] == task]
    wide = subset.pivot(index="seed", columns="model", values="value")
    models = sorted(wide.columns)
    records = []
    for i, model_a in enumerate(models):
        for model_b in models[i + 1:]:
            a, b = wide[model_a].to_numpy(), wide[model_b].to_numpy()
            difference = a - b
            if np.allclose(difference, 0.0):
                t_p, w_p = 1.0, 1.0
            else:
                t_p = float(stats.ttest_rel(a, b).pvalue)
                try:
                    w_p = float(stats.wilcoxon(a, b).pvalue)
                except ValueError:
                    w_p = 1.0
            records.append({"task": task, "model_a": model_a,
                            "model_b": model_b, "n_seeds": len(a),
                            "mean_a": round(float(a.mean()), 6),
                            "mean_b": round(float(b.mean()), 6),
                            "mean_difference": round(float(difference.mean()), 6),
                            "sd_difference": round(float(difference.std(ddof=1)), 6),
                            "seed_t_p": round(t_p, 6),
                            "seed_wilcoxon_p": round(w_p, 6)})
    if records:
        adjusted = holm([r["seed_t_p"] for r in records])
        for record, value in zip(records, adjusted):
            record["seed_t_p_holm"] = round(float(value), 6)
            # Deliberately not called "significant": see the module docstring.
            record["gap_exceeds_seed_noise"] = bool(value < ALPHA)
            record["licenses_superiority_claim"] = False
    return records


# --- Bootstrap tests --------------------------------------------------------
def bootstrap_tests(task):
    losses = row_losses(task)
    models = sorted(losses)
    records = []
    for i, model_a in enumerate(models):
        for model_b in models[i + 1:]:
            observed, low, high, p_value, per_match, n_matches = \
                clustered_bootstrap(losses[model_a], losses[model_b])
            difference = (per_match["a"] - per_match["b"]).to_numpy()
            try:
                w_p = float(stats.wilcoxon(difference).pvalue)
            except ValueError:
                w_p = 1.0
            records.append({
                "task": task, "model_a": model_a, "model_b": model_b,
                "n_matches": int(n_matches),
                "mean_loss_a": round(float(per_match["a"].mean()), 6),
                "mean_loss_b": round(float(per_match["b"].mean()), 6),
                "mean_difference": round(observed, 6),
                "ci_low": round(low, 6), "ci_high": round(high, 6),
                "bootstrap_p": round(p_value, 6),
                "wilcoxon_p": round(w_p, 6)})
    adjusted = holm([r["bootstrap_p"] for r in records])
    for record, value in zip(records, adjusted):
        record["bootstrap_p_holm"] = round(float(value), 6)
        record["significant"] = bool(value < ALPHA)
        record["verdict"] = (
            "no detectable difference" if value >= ALPHA
            else (f"{record['model_a']} better"
                  if record["mean_difference"] < 0
                  else f"{record['model_b']} better"))
    return records


def main():
    print(f"Seed repetitions: {N_SEEDS}   bootstrap resamples: {N_BOOTSTRAP}")

    seed_rows = []
    for task in TASKS:
        print(f"\n=== seed repetition: task {task} ===")
        seed_repetitions(task, seed_rows)
    seed_frame = pd.DataFrame(seed_rows)

    seed_records, bootstrap_records = [], []
    for task in TASKS:
        seed_records.extend(seed_tests(seed_frame, task))
        bootstrap_records.extend(bootstrap_tests(task))

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    seed_frame.to_csv(RESULTS_DIR / "seed_repetitions.csv", index=False,
                      encoding="utf-8")
    seed_test_frame = pd.DataFrame(seed_records)
    seed_test_frame.to_csv(RESULTS_DIR / "significance_seeds.csv", index=False,
                           encoding="utf-8")
    bootstrap_frame = pd.DataFrame(bootstrap_records)
    bootstrap_frame.to_csv(RESULTS_DIR / "significance_bootstrap.csv",
                           index=False, encoding="utf-8")

    summary = (seed_frame.groupby(["task", "model", "metric"])["value"]
               .agg(["mean", "std", "min", "max"]).round(5))
    print("\nSeed-wise test metric (mean +/- sd over seeds):")
    print(summary.to_string())

    print("\nSeed repetition measures stability only. Superiority claims come "
          "from the bootstrap below.")
    print("\nSignificant pairwise differences after Holm correction "
          "(match-clustered bootstrap):")
    significant = bootstrap_frame[bootstrap_frame["significant"]]
    if significant.empty:
        print("  none")
    else:
        print(significant[["task", "model_a", "model_b", "mean_difference",
                           "ci_low", "ci_high", "bootstrap_p_holm",
                           "verdict"]].to_string(index=False))
    print(f"\nWrote -> {RESULTS_DIR / 'seed_repetitions.csv'}")
    print(f"Wrote -> {RESULTS_DIR / 'significance_seeds.csv'}")
    print(f"Wrote -> {RESULTS_DIR / 'significance_bootstrap.csv'}")


if __name__ == "__main__":
    main()
