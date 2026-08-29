"""Run directly - there is no pytest config in this repo:

    python src/models/test_stacking.py

Guards the two things that make the stack legitimate rather than merely
better-scoring: the meta-learner sees out-of-fold predictions only, and for the
in-play tasks no match straddles a fold boundary.
"""
from pathlib import Path
import io
import sys

import joblib
import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

from stacking import (BASE_MODELS, STACK_MODELS, StackedClassifier,  # noqa: E402
                      StackedRegressor, _splits, build_named_stack,
                      build_stack, resolve_base_names, temporal_meta_holdout,
                      training_groups)

import pandas as pd  # noqa: E402

SMALL = {"xgboost": {"n_estimators": 25}, "lightgbm": {"n_estimators": 25},
         "random_forest": {"n_estimators": 25}, "gbm": {"max_iter": 25},
         "p2_hier_shrinkage": {"n_estimators": 25}}

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


def _toy(n_matches=80, per_match=5, seed=0):
    rng = np.random.default_rng(seed)
    n = n_matches * per_match
    X = rng.normal(size=(n, 6))
    score = X[:, 0] + 0.5 * X[:, 1] + rng.normal(scale=0.4, size=n)
    y = np.where(score > 0.6, "H", np.where(score < -0.6, "A", "D"))
    groups = np.repeat(np.arange(n_matches), per_match)
    return X, y, groups


def test_grouped_folds_keep_matches_whole():
    X, y, groups = _toy()
    folds = _splits(X, y, groups, 5, 0, True)
    assert len(folds) == 5, f"expected 5 folds, got {len(folds)}"
    for train_index, hold_index in folds:
        overlap = set(groups[train_index]) & set(groups[hold_index])
        assert not overlap, \
            f"{len(overlap)} match(es) appear on both sides of a fold boundary"


def test_folds_partition_every_row():
    X, y, groups = _toy()
    seen = np.zeros(len(X), dtype=int)
    for _, hold_index in _splits(X, y, groups, 5, 0, True):
        seen[hold_index] += 1
    assert (seen == 1).all(), \
        f"{int((seen != 1).sum())} rows are held out zero or multiple times"


def test_ungrouped_falls_back_to_stratified():
    X, y, _ = _toy()
    folds = _splits(X, y, None, 5, 0, True)
    assert len(folds) == 5
    for _, hold_index in folds:
        assert len(set(y[hold_index])) == len(set(y)), \
            "stratified fold lost a class"


def test_groups_length_is_asserted():
    X, y, groups = _toy()
    try:
        _splits(X, y, groups[:-3], 5, 0, True)
    except AssertionError as error:
        assert "groups length" in str(error), f"unexpected message: {error}"
        return
    raise AssertionError("mismatched groups length was not rejected")


def test_classifier_probabilities_are_a_distribution():
    X, y, groups = _toy()
    model = build_stack("Lc", tuned=SMALL, groups=groups).fit(X, y)
    assert isinstance(model, StackedClassifier)
    proba = model.predict_proba(X[:50])
    assert np.allclose(proba.sum(axis=1), 1.0), \
        "stacked probabilities do not sum to 1"
    assert (proba >= 0).all(), "stacked probabilities contain negatives"
    assert list(model.classes_) == ["A", "D", "H"], \
        f"class order drifted: {list(model.classes_)}"


def test_regressor_learns_a_combination():
    X, y, groups = _toy()
    model = build_stack("Lr", tuned=SMALL, groups=groups).fit(X, X[:, 0])
    assert isinstance(model, StackedRegressor)
    assert len(model.meta_.coef_) == len(model.base_names_), \
        "meta coefficients do not match the member count"
    error = np.abs(model.predict(X) - X[:, 0]).mean()
    assert error < 0.5, f"stacked regressor MAE {error:.3f} is not a fit"


def test_bundle_round_trips_through_joblib():
    X, y, groups = _toy()
    model = build_stack("Lc", tuned=SMALL, groups=groups).fit(X, y)
    expected = model.predict_proba(X[:20])
    buffer = io.BytesIO()
    joblib.dump(model, buffer)
    buffer.seek(0)
    restored = joblib.load(buffer)
    assert np.allclose(restored.predict_proba(X[:20]), expected), \
        "joblib round-trip changed the served probabilities"


def test_tree_base_is_walkable_for_shap():
    X, y, groups = _toy()
    model = build_stack("Lc", tuned=SMALL, groups=groups).fit(X, y)
    tree, classes, name = model.tree_base()
    assert tree is not None, "no tree member available for TreeSHAP"
    assert name in BASE_MODELS["Lc"], f"{name} is not a stack member"
    assert classes is not None and "H" in classes, \
        f"tree member reports classes {classes}; no home-win column"
    from model_zoo import LabelEncodedClassifier
    assert not isinstance(tree, LabelEncodedClassifier), \
        "tree_base returned the label-encoding wrapper, which TreeSHAP cannot walk"


def test_training_groups_only_for_inplay():
    frame = pd.DataFrame({"match_id": [1, 1, 2, 3],
                          "split": ["train", "train", "train", "test"]})
    assert training_groups(frame, "C") is None, \
        "pre-match tasks must not be grouped; one row is one match"
    grouped = training_groups(frame, "Lc")
    assert list(grouped) == [1, 1, 2], f"wrong training groups: {grouped}"


def _validation_frame():
    """Two-per-match snapshot table with a validation block spanning 5 dates."""
    rows = []
    for match, day in enumerate([3, 1, 5, 2, 4]):
        for minute in (0, 45):
            rows.append({"match_id": 100 + match,
                         "match_date": f"2022-01-0{day}",
                         "split": "validation", "snapshot_minute": minute,
                         "label_result": "H"})
    rows.append({"match_id": 1, "match_date": "2019-01-01", "split": "train",
                 "snapshot_minute": 0, "label_result": "A"})
    return pd.DataFrame(rows)


def test_temporal_holdout_takes_the_earliest_matches():
    frame = _validation_frame()
    seen = []
    holdout = temporal_meta_holdout(
        frame, "Lc", lambda subset: seen.append(subset) or np.zeros(
            (len(subset), 1)), frac=0.6)
    early = seen[0]
    assert set(early["match_date"]) == {"2022-01-01", "2022-01-02",
                                        "2022-01-03"}, \
        f"holdout took the wrong dates: {sorted(set(early['match_date']))}"
    assert len(holdout[1]) == len(early), "targets and rows disagree"
    assert "train" not in set(early["split"]), \
        "the meta holdout must come from validation only"


def test_temporal_holdout_cuts_on_match_boundaries():
    frame = _validation_frame()
    seen = []
    temporal_meta_holdout(frame, "Lc",
                          lambda subset: seen.append(subset) or np.zeros(
                              (len(subset), 1)), frac=0.5)
    counts = seen[0].groupby("match_id").size()
    assert (counts == 2).all(), \
        f"a match was split across the holdout boundary: {counts.to_dict()}"


def test_temporal_holdout_never_takes_everything():
    frame = _validation_frame()
    try:
        temporal_meta_holdout(frame, "Lc", lambda subset: np.zeros(
            (len(subset), 1)), frac=1.0)
    except AssertionError as error:
        assert "swallowed" in str(error), f"unexpected message: {error}"
        return
    raise AssertionError("frac=1.0 left no rows for calibration and was allowed")


def test_named_builders_cover_every_variant():
    X, y, groups = _toy()
    frame = pd.DataFrame({"match_id": np.repeat(np.arange(80), 5),
                          "match_date": "2020-01-01",
                          "split": "train", "label_result": y})
    for model_name in STACK_MODELS:
        if model_name == "stack_temporal":
            continue  # needs a validation block; covered above
        built = build_named_stack(model_name, "Lc", SMALL, frame)
        assert isinstance(built, StackedClassifier), model_name
    try:
        build_named_stack("not_a_stack", "Lc", SMALL, frame)
    except AssertionError:
        return
    raise AssertionError("an unknown model name was accepted as a stack")


def test_member_list_is_available():
    names = resolve_base_names("C", SMALL)
    assert len(names) >= 2, f"stack would have too few members: {names}"
    assert "kernel_svm" not in names, \
        "kernel_svm must stay out of the stack (940 s for the worst RPS)"


def main():
    print("stacking.py")
    for name, function in sorted(globals().items()):
        if name.startswith("test_"):
            check(name, function)
    print(f"\n{len(PASSED)} passed, {len(FAILED)} failed")
    if FAILED:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
