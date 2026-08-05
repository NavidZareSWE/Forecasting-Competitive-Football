"""Brief section 1: can the Task R margin be turned into useful H/D/A
probabilities, and does it beat the Task C classifier that was trained for the
job?

Two links, both fitted on the validation split only and scored once on test:

  ordinal   P(A) = Phi((-theta - m)/sigma),  P(H) = 1 - Phi((theta - m)/sigma)
  logistic  multinomial logistic regression on [m, m^2]

    python src/models/margin_to_probability.py
"""

import numpy as np
import pandas as pd
from scipy.optimize import minimize
from scipy.stats import norm
from sklearn.linear_model import LogisticRegression

from modeling_common import (CLASS_ORDER, RESULTS_DIR, classification_metrics,
                             floor_probabilities, prepare_matrices, task_frame)
from model_zoo import regressor_zoo, classifier_zoo
from tuning import load_best_params


RANDOM_STATE = 0
MARGIN_CLIP = (-5.0, 5.0)
EPS = 1e-12


# --- Ordinal threshold link -------------------------------------------------
def ordinal_probabilities(margin, theta, sigma):
    sigma = max(float(sigma), 1e-6)
    p_home = 1.0 - norm.cdf((theta - margin) / sigma)
    p_away = norm.cdf((-theta - margin) / sigma)
    p_draw = np.clip(1.0 - p_home - p_away, EPS, 1.0)
    stacked = np.column_stack([p_home, p_draw, p_away])
    return stacked / stacked.sum(axis=1, keepdims=True)


def fit_ordinal(margin, y_true):
    index = {c: i for i, c in enumerate(CLASS_ORDER)}
    columns = np.array([index[label] for label in y_true])
    rows = np.arange(len(y_true))

    def negative_log_likelihood(parameters):
        theta, log_sigma = parameters
        if theta <= 0:
            return 1e9
        proba = ordinal_probabilities(margin, theta, np.exp(log_sigma))
        return float(-np.log(np.clip(proba[rows, columns], EPS, 1.0)).mean())

    best = minimize(negative_log_likelihood, x0=np.array([0.5, np.log(1.3)]),
                    method="Nelder-Mead",
                    options={"maxiter": 2000, "xatol": 1e-6, "fatol": 1e-9})
    theta, log_sigma = best.x
    return float(theta), float(np.exp(log_sigma))


# --- Task assembly ----------------------------------------------------------
def margin_predictions(task="R"):
    """Refit the best Task R regressor and predict on validation and test."""
    df, continuous, nominal, target, task_type = task_frame(task)
    matrices = prepare_matrices(df, continuous, nominal, target, task_type)
    tuned = load_best_params().get(task, {})

    best_name, best_mae, best = None, np.inf, None
    for name, factory in regressor_zoo(RANDOM_STATE, task="R",
                                       tuned=tuned).items():
        if name == "dummy":
            continue
        estimator = factory()
        estimator.fit(matrices["X_train"], matrices["y_train"].astype(float))
        validation = np.clip(estimator.predict(matrices["X_val"]), *MARGIN_CLIP)
        mae = float(np.abs(validation
                           - matrices["y_val"].astype(float)).mean())
        print(f"  [R] {name:22s} validation mae={mae:.5f}")
        if mae < best_mae:
            best_name, best_mae, best = name, mae, estimator
    print(f"  selected on validation: {best_name} (mae={best_mae:.5f})")
    return (best_name,
            np.clip(best.predict(matrices["X_val"]), *MARGIN_CLIP),
            np.clip(best.predict(matrices["X_test"]), *MARGIN_CLIP),
            matrices)


def classifier_reference(matrices_c):
    """Task C classifier chosen on validation, for the head-to-head."""
    tuned = load_best_params().get("C", {})
    best_name, best_rps, best_proba = None, np.inf, None
    for name, factory in classifier_zoo(RANDOM_STATE, task="C",
                                        tuned=tuned).items():
        if name == "dummy":
            continue
        estimator = factory()
        estimator.fit(matrices_c["X_train"], matrices_c["y_train"])
        validation = _aligned(estimator, matrices_c["X_val"])
        rps = classification_metrics(validation, matrices_c["y_val"])["rps"]
        if rps < best_rps:
            best_name, best_rps = name, rps
            best_proba = _aligned(estimator, matrices_c["X_test"])
    print(f"  selected on validation: {best_name} (rps={best_rps:.5f})")
    return best_name, floor_probabilities(best_proba)


def _aligned(estimator, X):
    proba = estimator.predict_proba(X)
    classes = [str(c) for c in estimator.classes_]
    out = np.zeros((proba.shape[0], len(CLASS_ORDER)))
    for j, label in enumerate(CLASS_ORDER):
        if label in classes:
            out[:, j] = proba[:, classes.index(label)]
    totals = out.sum(axis=1, keepdims=True)
    totals[totals == 0] = 1.0
    return out / totals


def main():
    print("=== Task R regressors (validation) ===")
    regressor_name, margin_val, margin_test, matrices_r = margin_predictions()

    df_c, continuous, nominal, target, task_type = task_frame("C")
    matrices_c = prepare_matrices(df_c, continuous, nominal, target, task_type)

    # Task C and Task R share the pre-match table, so the rows already align.
    assert (matrices_c["meta_test"]["match_id"].to_numpy()
            == matrices_r["meta_test"]["match_id"].to_numpy()).all(), \
        "Task C and Task R test rows are not in the same order"

    y_val = matrices_c["y_val"]
    y_test = matrices_c["y_test"]

    theta, sigma = fit_ordinal(margin_val, y_val)
    print(f"\nOrdinal link fitted on validation: theta={theta:.4f} "
          f"sigma={sigma:.4f}")
    ordinal_test = floor_probabilities(
        ordinal_probabilities(margin_test, theta, sigma))

    design_val = np.column_stack([margin_val, margin_val ** 2])
    design_test = np.column_stack([margin_test, margin_test ** 2])
    logistic = LogisticRegression(max_iter=2000)
    logistic.fit(design_val, y_val)
    logistic_test = floor_probabilities(_aligned(logistic, design_test))

    print("\n=== Task C classifiers (validation) ===")
    classifier_name, classifier_test = classifier_reference(matrices_c)

    rows = []
    for label, proba, source in [
            (f"model2_ordinal[{regressor_name}]", ordinal_test, "Task R margin"),
            (f"model2_logistic[{regressor_name}]", logistic_test, "Task R margin"),
            (f"model1_direct[{classifier_name}]", classifier_test,
             "Task C classifier")]:
        metrics = classification_metrics(proba, y_test)
        rows.append({"converter": label, "source": source,
                     **{k: round(v, 5) for k, v in metrics.items()}})

    results = pd.DataFrame(rows).sort_values("rps").reset_index(drop=True)
    results["theta"] = round(theta, 5)
    results["sigma"] = round(sigma, 5)
    results["n_test"] = len(y_test)

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    output_path = RESULTS_DIR / "margin_to_probability.csv"
    results.to_csv(output_path, index=False, encoding="utf-8")

    per_row = pd.DataFrame({
        "match_id": matrices_c["meta_test"]["match_id"].to_numpy(),
        "y_true": y_test, "margin_pred": np.round(margin_test, 5)})
    for position, klass in enumerate(CLASS_ORDER):
        per_row[f"ordinal_p_{klass}"] = np.round(ordinal_test[:, position], 5)
        per_row[f"logistic_p_{klass}"] = np.round(logistic_test[:, position], 5)
        per_row[f"direct_p_{klass}"] = np.round(classifier_test[:, position], 5)
    per_row.to_csv(RESULTS_DIR / "margin_to_probability_predictions.csv",
                   index=False, encoding="utf-8")

    print("\nTest-set comparison (RPS lower is better):")
    print(results[["converter", "source", "rps", "log_loss", "brier",
                   "ece"]].to_string(index=False))
    print(f"\nWrote -> {output_path}")
    print(f"Wrote -> {RESULTS_DIR / 'margin_to_probability_predictions.csv'}")


if __name__ == "__main__":
    main()
