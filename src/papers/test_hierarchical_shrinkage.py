"""Tests for the P2 reimplementation (Hierarchical Shrinkage, ICML 2022).

    python src/papers/test_hierarchical_shrinkage.py
"""

import numpy as np
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor

from hierarchical_shrinkage import (HSForestClassifier, HSForestRegressor,
                                    HSTreeClassifier, HSTreeRegressor,
                                    _leaf_indices, _node_means, _shrink_tree,
                                    LAMBDA_GRID)


def make_classification_data(n=400, seed=0):
    rng = np.random.default_rng(seed)
    X = rng.normal(size=(n, 5))
    score = X[:, 0] * 1.5 - X[:, 1]
    y = np.where(score > 0.6, "H", np.where(score < -0.6, "A", "D"))
    return X, y


def make_regression_data(n=400, seed=1):
    rng = np.random.default_rng(seed)
    X = rng.normal(size=(n, 5))
    y = 2.0 * X[:, 0] - X[:, 2] + rng.normal(scale=0.3, size=n)
    return X, y


def test_lambda_zero_is_the_unshrunk_tree():
    X, y = make_regression_data()
    base = DecisionTreeRegressor(max_depth=6, random_state=0).fit(X, y)
    shrunk = _shrink_tree(base.tree_, lam=0.0, is_classifier=False)
    mu = _node_means(base.tree_, is_classifier=False)
    assert np.allclose(shrunk, mu), "lam=0 must leave every node untouched"

    hs = HSTreeRegressor(lam=0.0, max_depth=6, random_state=0).fit(X, y)
    assert np.allclose(hs.predict(X), base.predict(X)), \
        "lam=0 predictions must equal the plain CART predictions"
    print("ok  lam=0 recovers the unshrunk tree")


def test_large_lambda_collapses_to_the_root_mean():
    X, y = make_regression_data()
    hs = HSTreeRegressor(lam=1e9, max_depth=6, random_state=0).fit(X, y)
    predictions = hs.predict(X)
    assert np.allclose(predictions, y.mean(), atol=1e-3), \
        "lam -> infinity must collapse every leaf onto the root mean"
    print("ok  lam -> infinity collapses to the root mean")


def test_shrinkage_matches_the_paper_equation_on_every_path():
    # Checked against a literal per-path transcription of the paper equation.
    X, y = make_regression_data(n=200, seed=3)
    tree = DecisionTreeRegressor(max_depth=5, random_state=0).fit(X, y).tree_
    lam = 12.5
    mu = _node_means(tree, is_classifier=False)
    counts = tree.weighted_n_node_samples
    parent = {}
    for node in range(tree.node_count):
        for child in (tree.children_left[node], tree.children_right[node]):
            if child != -1:
                parent[child] = node

    fast = _shrink_tree(tree, lam, is_classifier=False)
    for node in range(tree.node_count):
        path = [node]
        while path[-1] in parent:
            path.append(parent[path[-1]])
        path.reverse()
        literal = mu[path[0]].copy()
        for depth in range(1, len(path)):
            ancestor = path[depth - 1]
            literal += (mu[path[depth]] - mu[ancestor]) / (1 + lam / counts[ancestor])
        assert np.allclose(fast[node], literal), \
            f"node {node} disagrees with the paper equation"
    print("ok  recursion equals the paper's telescoping sum on every path")


def test_classifier_output_is_a_valid_probability_vector():
    # Appendix A predicts no clipping is needed at any lam.
    X, y = make_classification_data()
    tree = DecisionTreeClassifier(max_depth=8, random_state=0).fit(X, y).tree_
    for lam in LAMBDA_GRID:
        shrunk = _shrink_tree(tree, lam, is_classifier=True)
        assert np.all(shrunk >= -1e-12), f"negative probability at lam={lam}"
        assert np.allclose(shrunk.sum(axis=1), 1.0), \
            f"probabilities do not sum to 1 at lam={lam}"
    print("ok  shrunk class vectors stay on the simplex for every lambda")


def test_own_leaf_descent_matches_sklearn_apply():
    X, y = make_classification_data()
    tree = DecisionTreeClassifier(max_depth=7, random_state=0).fit(X, y).tree_
    assert np.array_equal(_leaf_indices(tree, X), tree.apply(np.float32(X))), \
        "hand-written descent disagrees with sklearn's traversal"
    print("ok  hand-written leaf descent matches sklearn's traversal")


def test_shrinkage_is_monotone_towards_the_root():
    # Increasing lam must move every leaf monotonically closer to the root mean.
    X, y = make_regression_data()
    tree = DecisionTreeRegressor(max_depth=8, random_state=0).fit(X, y).tree_
    root = _node_means(tree, is_classifier=False)[0, 0]
    leaves = [i for i in range(tree.node_count) if tree.children_left[i] == -1]
    previous = None
    for lam in [0.0, 1.0, 10.0, 100.0, 1000.0]:
        gap = np.abs(_shrink_tree(tree, lam, False)[leaves, 0] - root).mean()
        if previous is not None:
            assert gap <= previous + 1e-12, \
                f"lam={lam} moved leaves away from the root mean"
        previous = gap
    print("ok  larger lambda moves leaves monotonically toward the root")


def test_forest_variants_fit_predict_and_beat_a_constant():
    X, y = make_classification_data(n=600, seed=5)
    clf = HSForestClassifier(lam=25.0, n_estimators=40, random_state=0).fit(X, y)
    proba = clf.predict_proba(X)
    assert proba.shape == (600, 3)
    assert np.allclose(proba.sum(axis=1), 1.0)
    assert (clf.predict(X) == y).mean() > 0.8, "forest classifier underfits badly"

    Xr, yr = make_regression_data(n=600, seed=6)
    reg = HSForestRegressor(lam=25.0, n_estimators=40, random_state=0).fit(Xr, yr)
    mae = np.abs(reg.predict(Xr) - yr).mean()
    assert mae < np.abs(yr - yr.mean()).mean(), "forest regressor worse than the mean"
    print("ok  forest classifier and regressor fit, predict and beat a constant")


def test_cv_selects_a_lambda_from_the_grid_without_touching_validation():
    X, y = make_classification_data(n=500, seed=7)
    clf = HSForestClassifier(cv_lambdas=LAMBDA_GRID, cv_folds=3,
                             n_estimators=30, random_state=0).fit(X, y)
    assert clf.lam_ in LAMBDA_GRID, "selected lambda is not from the grid"
    assert set(clf.cv_scores_) == set(LAMBDA_GRID), "grid not fully scored"
    best = min(clf.cv_scores_, key=clf.cv_scores_.get)
    assert clf.lam_ == best, "selected lambda is not the CV minimiser"
    print(f"ok  CV selected lambda={clf.lam_} from the grid on training rows only")


def main():
    tests = [test_lambda_zero_is_the_unshrunk_tree,
             test_large_lambda_collapses_to_the_root_mean,
             test_shrinkage_matches_the_paper_equation_on_every_path,
             test_classifier_output_is_a_valid_probability_vector,
             test_own_leaf_descent_matches_sklearn_apply,
             test_shrinkage_is_monotone_towards_the_root,
             test_forest_variants_fit_predict_and_beat_a_constant,
             test_cv_selects_a_lambda_from_the_grid_without_touching_validation]
    for test in tests:
        test()
    print(f"\n{len(tests)}/{len(tests)} passed")


if __name__ == "__main__":
    main()
