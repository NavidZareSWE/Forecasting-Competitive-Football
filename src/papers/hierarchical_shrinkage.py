"""P2 reimplementation: Hierarchical Shrinkage.

Agarwal, Tan, Ronen, Singh, Yu. "Hierarchical Shrinkage: Improving the accuracy
and interpretability of tree-based models." ICML 2022, PMLR 162:111-135.

    f_HS(x) = mu(t_0) + sum_l [mu(t_l) - mu(t_{l-1})] / (1 + lam/N(t_{l-1}))

over the root-to-leaf path. Derivation: docs/appendix_a_hierarchical_shrinkage.md
"""

import numpy as np
from sklearn.base import BaseEstimator, ClassifierMixin, RegressorMixin
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.model_selection import KFold, StratifiedKFold
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor


LAMBDA_GRID = (0.0, 0.1, 1.0, 10.0, 25.0, 50.0, 100.0)

_LEAF = -1


def _node_means(tree, is_classifier):
    values = np.asarray(tree.value, dtype=float)[:, 0, :]
    if is_classifier:
        totals = values.sum(axis=1, keepdims=True)
        totals[totals == 0.0] = 1.0
        values = values / totals
    return values


def _shrink_tree(tree, lam, is_classifier):
    left = tree.children_left
    right = tree.children_right
    counts = np.asarray(tree.weighted_n_node_samples, dtype=float)
    mu = _node_means(tree, is_classifier)

    shrunk = np.empty_like(mu)
    shrunk[0] = mu[0]

    stack = [0]
    while stack:
        parent = stack.pop()
        if left[parent] == _LEAF:
            continue
        weight = 1.0 / (1.0 + lam / counts[parent])
        for child in (left[parent], right[parent]):
            shrunk[child] = shrunk[parent] + (mu[child] - mu[parent]) * weight
            stack.append(child)
    return shrunk


def _leaf_indices(tree, X):
    left = tree.children_left
    right = tree.children_right
    feature = tree.feature
    threshold = tree.threshold

    X = np.asarray(X, dtype=np.float64)
    nodes = np.zeros(X.shape[0], dtype=np.intp)
    active = np.flatnonzero(left[nodes] != _LEAF)
    while active.size:
        here = nodes[active]
        go_left = X[active, feature[here]] <= threshold[here]
        nodes[active] = np.where(go_left, left[here], right[here])
        active = active[left[nodes[active]] != _LEAF]
    return nodes


class _HSBase(BaseEstimator):

    def __init__(self, lam=25.0, n_estimators=300, max_depth=None,
                 max_leaf_nodes=None, cv_lambdas=None, cv_folds=3,
                 n_jobs=-1, random_state=None):
        self.lam = lam
        self.n_estimators = n_estimators
        self.max_depth = max_depth
        self.max_leaf_nodes = max_leaf_nodes
        self.cv_lambdas = cv_lambdas
        self.cv_folds = cv_folds
        self.n_jobs = n_jobs
        self.random_state = random_state

    _is_classifier = None

    def _make_base(self):
        raise NotImplementedError

    def _score(self, X, y):
        raise NotImplementedError

    def _fit_shrunk(self, X, y, lam):
        base = self._make_base()
        base.fit(X, y)
        trees = getattr(base, "estimators_", [base])
        self._trees_ = [t.tree_ for t in trees]
        self._shrunk_ = [_shrink_tree(t.tree_, lam, self._is_classifier)
                         for t in trees]
        self._base_ = base
        return base

    def fit(self, X, y):
        X = np.asarray(X, dtype=np.float64)
        y = np.asarray(y)
        self.n_features_in_ = X.shape[1]
        if self._is_classifier:
            self.classes_ = np.unique(y)

        if self.cv_lambdas:
            self.lam_ = self._select_lambda(X, y)
        else:
            self.lam_ = float(self.lam)

        self._fit_shrunk(X, y, self.lam_)
        return self

    def _select_lambda(self, X, y):
        grid = list(self.cv_lambdas)
        if len(grid) == 1:
            return float(grid[0])
        if self._is_classifier:
            splitter = StratifiedKFold(n_splits=self.cv_folds, shuffle=True,
                                       random_state=self.random_state)
        else:
            splitter = KFold(n_splits=self.cv_folds, shuffle=True,
                             random_state=self.random_state)

        totals = np.zeros(len(grid))
        for train_index, val_index in splitter.split(X, y):
            probe = self.__class__(**{**self.get_params(), "cv_lambdas": None})
            base = probe._make_base()
            base.fit(X[train_index], y[train_index])
            trees = getattr(base, "estimators_", [base])
            probe.n_features_in_ = X.shape[1]
            if self._is_classifier:
                probe.classes_ = np.unique(y[train_index])
            probe._trees_ = [t.tree_ for t in trees]
            probe._base_ = base
            for position, lam in enumerate(grid):
                probe._shrunk_ = [_shrink_tree(t.tree_, lam,
                                               self._is_classifier)
                                  for t in trees]
                probe.lam_ = lam
                totals[position] += probe._score(X[val_index], y[val_index])
        self.cv_scores_ = dict(zip(grid, totals / self.cv_folds))
        return float(grid[int(np.argmin(totals))])

    def shap_base_estimator(self):
        base = self._base_
        trees = getattr(base, "estimators_", [base])
        for estimator, shrunk in zip(trees, self._shrunk_):
            estimator.tree_.value[:, 0, :] = shrunk
        self._materialised_ = True
        return base

    def _raw_predict(self, X):
        X = np.asarray(X, dtype=np.float64)
        total = None
        for tree, shrunk in zip(self._trees_, self._shrunk_):
            values = shrunk[_leaf_indices(tree, X)]
            total = values if total is None else total + values
        return total / len(self._trees_)


class HSForestClassifier(ClassifierMixin, _HSBase):
    _is_classifier = True

    def _make_base(self):
        return RandomForestClassifier(
            n_estimators=self.n_estimators, max_depth=self.max_depth,
            max_leaf_nodes=self.max_leaf_nodes, n_jobs=self.n_jobs,
            random_state=self.random_state)

    def predict_proba(self, X):
        proba = np.clip(self._raw_predict(X), 0.0, 1.0)
        return proba / proba.sum(axis=1, keepdims=True)

    def predict(self, X):
        return self.classes_[self.predict_proba(X).argmax(axis=1)]

    def _score(self, X, y):
        proba = np.clip(self.predict_proba(X), 1e-15, 1.0)
        index = {label: i for i, label in enumerate(self.classes_)}
        rows = np.arange(len(y))
        columns = np.array([index.get(label, 0) for label in y])
        return float(-np.log(proba[rows, columns]).mean())


class HSForestRegressor(RegressorMixin, _HSBase):
    _is_classifier = False

    def _make_base(self):
        return RandomForestRegressor(
            n_estimators=self.n_estimators, max_depth=self.max_depth,
            max_leaf_nodes=self.max_leaf_nodes, n_jobs=self.n_jobs,
            random_state=self.random_state)

    def predict(self, X):
        return self._raw_predict(X)[:, 0]

    def _score(self, X, y):
        return float(((self.predict(X) - np.asarray(y, float)) ** 2).mean())


class HSTreeClassifier(HSForestClassifier):
    def _make_base(self):
        return DecisionTreeClassifier(
            max_depth=self.max_depth, max_leaf_nodes=self.max_leaf_nodes,
            random_state=self.random_state)


class HSTreeRegressor(HSForestRegressor):
    def _make_base(self):
        return DecisionTreeRegressor(
            max_depth=self.max_depth, max_leaf_nodes=self.max_leaf_nodes,
            random_state=self.random_state)
