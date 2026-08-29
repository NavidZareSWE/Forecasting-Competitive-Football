"""Run directly - there is no pytest config in this repo:

    python src/models/test_frozen_reference.py

Guards the frozen pre-match reference curve required for Task L. The extended
pre-match test split and the StatsBomb in-play test split share no match_id,
so the reference has to be built from pre-match predictions on the excluded
split. When that plumbing breaks the failure is silent: the merge returns an
empty frame, the curve loop appends nothing, and the old spread assertion
passes vacuously over zero rows.
"""
from pathlib import Path
import sys

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

from modeling_common import prepare_matrices  # noqa: E402
from train import evaluate_classification, evaluate_regression  # noqa: E402
from inplay_curves import freeze_prematch  # noqa: E402

from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor  # noqa: E402

PASSED = []
FAILED = []


def check(name, function):
    try:
        function()
        PASSED.append(name)
        print(f"  PASS  {name}")
    except AssertionError as error:
        FAILED.append((name, error))
        print(f"  FAIL  {name}: {error}")


# --- Fixtures ---------------------------------------------------------------
def _prematch(n=400, seed=0):
    rng = np.random.default_rng(seed)
    splits = (["train"] * int(n * 0.6) + ["validation"] * int(n * 0.15)
              + ["test"] * int(n * 0.15))
    splits = splits + ["excluded"] * (n - len(splits))
    strength = rng.normal(size=n)
    margin = np.clip(np.round(strength * 1.5 + rng.normal(size=n)), -5, 5)
    return pd.DataFrame({
        "match_id": np.arange(1, n + 1),
        "match_date": pd.date_range("2015-08-01", periods=n, freq="D"),
        "split": splits,
        "competition_name": rng.choice(["La Liga", "Serie A"], size=n),
        "elo_diff": strength * 50.0,
        "form_points": rng.normal(size=n),
        "label_margin": margin,
        "label_result": np.where(margin > 0, "H",
                                 np.where(margin < 0, "A", "D")),
    })


CONTINUOUS = ["elo_diff", "form_points"]
NOMINAL = ["competition_name"]


def _classification_data(df):
    return (df, CONTINUOUS, NOMINAL, "label_result", "classification")


def _regression_data(df):
    return (df, CONTINUOUS, NOMINAL, "label_margin", "regression")


def _forest_classifier():
    return RandomForestClassifier(n_estimators=25, random_state=0)


def _forest_regressor():
    return RandomForestRegressor(n_estimators=25, random_state=0)


# --- Tests ------------------------------------------------------------------
def test_excluded_split_reaches_the_matrices():
    df = _prematch()
    matrices = prepare_matrices(*_classification_data(df))
    expected = int((df["split"] == "excluded").sum())
    assert matrices["X_reference"] is not None, \
        "excluded rows were dropped before the design matrix"
    assert matrices["X_reference"].shape[0] == expected, \
        f"expected {expected} reference rows, got {matrices['X_reference'].shape[0]}"
    assert matrices["X_reference"].shape[1] == matrices["X_train"].shape[1], \
        "reference matrix has a different width from the training matrix"


def test_reference_is_absent_when_no_rows_are_excluded():
    df = _prematch()
    df = df[df["split"] != "excluded"]
    matrices = prepare_matrices(*_classification_data(df))
    assert matrices["X_reference"] is None, \
        "an empty excluded split must not produce a reference matrix"


def test_classification_emits_calibrated_reference_rows():
    df = _prematch()
    _, _, reference = evaluate_classification(
        "random_forest", _forest_classifier, _classification_data(df),
        with_reference=True)
    excluded = df[df["split"] == "excluded"]
    assert reference is not None and len(reference) == len(excluded), \
        "reference frame does not cover every excluded match"
    assert set(reference["match_id"]) == set(excluded["match_id"]), \
        "reference frame covers the wrong matches"
    probabilities = reference[["p_H", "p_D", "p_A"]].to_numpy()
    assert np.allclose(probabilities.sum(axis=1), 1.0), \
        "reference probabilities do not sum to 1"


def test_regression_emits_clipped_reference_rows():
    df = _prematch()
    _, _, reference = evaluate_regression(
        "random_forest", _forest_regressor, _regression_data(df),
        with_reference=True)
    excluded = df[df["split"] == "excluded"]
    assert reference is not None and len(reference) == len(excluded), \
        "reference frame does not cover every excluded match"
    assert reference["y_pred"].between(-5, 5).all(), \
        "reference margins escaped the [-5, +5] clip"


def test_default_call_keeps_the_two_tuple_signature():
    df = _prematch()
    result = evaluate_classification(
        "random_forest", _forest_classifier, _classification_data(df))
    assert len(result) == 2, \
        "callers that do not ask for a reference must still get (row, frame)"


def test_freeze_rejects_a_reference_that_misses_matches():
    inplay = pd.DataFrame({
        "match_id": np.repeat([1, 2], 3),
        "snapshot_minute": [0, 45, 90] * 2,
        "y_true": ["H"] * 6,
    })
    covered = pd.DataFrame({"match_id": [1, 2], "p_H": [0.5, 0.4],
                            "p_D": [0.3, 0.3], "p_A": [0.2, 0.3]})
    frozen = freeze_prematch(inplay, covered, ["p_H", "p_D", "p_A"])
    assert len(frozen) == len(inplay), "a full reference should merge cleanly"

    partial = covered[covered["match_id"] == 1]
    try:
        freeze_prematch(inplay, partial, ["p_H", "p_D", "p_A"])
    except AssertionError:
        return
    raise AssertionError(
        "a reference missing half the in-play matches was accepted silently")


def main():
    print("frozen pre-match reference")
    for name, function in sorted(globals().items()):
        if name.startswith("test_"):
            check(name, function)
    print(f"\n{len(PASSED)} passed, {len(FAILED)} failed")
    if FAILED:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
