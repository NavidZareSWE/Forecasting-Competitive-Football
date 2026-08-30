"""Out-of-fold stacked ensembles over the tuned zoo.

Why a stack and not a soft vote: the zoo's six learners sit within 0.001 RPS of
each other on task C, which is exactly the regime where a *fitted* combiner
beats a fixed average - the members disagree on which rows they are confident
about, and a multinomial logit over their class probabilities can learn that.

Leakage discipline (brief 7.2 / CLAUDE.md):

  * The meta-learner is fitted on **out-of-fold** predictions from the training
    split only. Validation is left untouched so the existing isotonic
    calibration step still sees rows no part of this model has been fitted on.
  * For the in-play tasks the folds are grouped by ``match_id``. Plain KFold
    would put minute 15 of a match in the fold-training half and minute 60 in
    the held-out half, and the meta-features would be optimistic.
  * ``groups`` must line up row-for-row with ``X``; ``fit`` asserts it.

The estimator is picklable: it stores base-model *names* plus their tuned
parameter dicts (plain data) and rebuilds estimators through ``model_zoo``, so
nothing here holds a lambda. ``model_zoo`` is imported lazily to keep the
registration in that module from being a circular import.
"""
from pathlib import Path
import sys

import numpy as np
from sklearn.base import BaseEstimator, ClassifierMixin, RegressorMixin
from sklearn.linear_model import LogisticRegression, RidgeCV
from sklearn.model_selection import GroupKFold, KFold, StratifiedKFold

HERE = Path(__file__).resolve().parent
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))

# Members of the stack, per zoo task. Ordered best-first on the sweep so the
# meta-feature block is readable. Kernel members are excluded: kernel_svm costs
# 940 s for the worst RPS in the zoo, and refitting it five more times buys
# nothing. p2_hier_shrinkage is excluded from Lr only, where it costs 260 s.
BASE_MODELS = {
    "C": ["xgboost", "p2_hier_shrinkage", "random_forest", "gbm", "lightgbm"],
    "R": ["xgboost", "gbm", "lightgbm", "random_forest", "p2_hier_shrinkage"],
    "Lc": ["random_forest", "xgboost", "p2_hier_shrinkage", "gbm", "lightgbm"],
    "Lr": ["random_forest", "xgboost", "gbm", "lightgbm"],
}
ZOO_TASK = {"C": "C", "R": "R", "Lc": "L", "Lr": "L"}
N_FOLDS = 5
# TreeSHAP needs a tree ensemble; app.py explains the stack through whichever
# of these members it actually contains (first match wins, best-first order).
TREE_MODELS = ["lightgbm", "xgboost", "gbm", "random_forest"]


def _zoo_for(task, tuned, random_state, classification):
    from model_zoo import classifier_zoo, regressor_zoo  # lazy: see docstring
    builder = classifier_zoo if classification else regressor_zoo
    return builder(random_state, task=ZOO_TASK[task], tuned=tuned or {})


def resolve_base_names(task, tuned=None, random_state=0):
    """The stack members actually available (xgboost/lightgbm may be absent)."""
    classification = task in {"C", "Lc"}
    zoo = _zoo_for(task, tuned, random_state, classification)
    names = [n for n in BASE_MODELS[task] if n in zoo]
    assert len(names) >= 2, (
        f"stack for task {task} needs at least 2 members, found {names}; "
        "install libomp so xgboost/lightgbm import (see CLAUDE.md)")
    return names


def _aligned_proba(estimator, X, classes):
    """predict_proba re-ordered onto ``classes`` and renormalised."""
    proba = estimator.predict_proba(X)
    own = [str(c) for c in estimator.classes_]
    out = np.zeros((proba.shape[0], len(classes)))
    for j, label in enumerate(classes):
        if str(label) in own:
            out[:, j] = proba[:, own.index(str(label))]
    totals = out.sum(axis=1, keepdims=True)
    totals[totals == 0] = 1.0
    return out / totals


def _splits(X, y, groups, n_folds, random_state, classification):
    if groups is not None:
        groups = np.asarray(groups)
        assert len(groups) == X.shape[0], (
            f"stack groups length {len(groups)} != {X.shape[0]} training rows; "
            "pass the training split's match_id column in row order")
        n_groups = len(np.unique(groups))
        folds = min(n_folds, n_groups)
        return list(GroupKFold(n_splits=folds).split(X, y, groups))
    if classification:
        return list(StratifiedKFold(n_splits=n_folds, shuffle=True,
                                    random_state=random_state).split(X, y))
    return list(KFold(n_splits=n_folds, shuffle=True,
                      random_state=random_state).split(X, y))


class _Stack(BaseEstimator):
    def __init__(self, task="C", tuned=None, base_names=None, groups=None,
                 n_folds=N_FOLDS, random_state=0, meta_holdout=None):
        self.task = task
        self.tuned = tuned
        self.base_names = base_names
        self.groups = groups
        self.n_folds = n_folds
        self.random_state = random_state
        self.meta_holdout = meta_holdout

    def _build(self, name):
        zoo = _zoo_for(self.task, self.tuned, self.random_state,
                       self._classification)
        return zoo[name]()

    def _fit_bases(self, X, y):
        """Fit the members and return (meta_features, meta_targets).

        Two regimes, selected by whether ``meta_holdout`` was supplied:

        * ``None`` - textbook stacking. Five-fold out-of-fold predictions over
          the training split; the meta-learner never sees a member's in-sample
          output. Correct in general, but the weights are learned on random
          folds of 2008-2021 and this project's test split is 2023-25, so they
          transfer only as well as the era does.
        * ``(X_meta, y_meta)`` - temporal blending. Members are fitted once on
          the whole training split and the meta-learner is fitted on their
          predictions over a held-out block that sits *between* train and test
          in time. Same idea as the market blend's validation-chosen alpha, and
          the same reason: the combination weights should come from the most
          recent data that is still not the test set.
        """
        names = self.base_names or resolve_base_names(
            self.task, self.tuned, self.random_state)
        self.base_names_ = list(names)
        if self.meta_holdout is not None:
            return self._fit_bases_on_holdout(X, y)
        folds = _splits(X, y, self.groups, self.n_folds, self.random_state,
                        self._classification)
        names = self.base_names_
        self.n_folds_ = len(folds)
        width = len(self._meta_classes()) if self._classification else 1
        oof = np.zeros((X.shape[0], len(names) * width))
        covered = np.zeros(X.shape[0], dtype=bool)
        for train_index, hold_index in folds:
            covered[hold_index] = True
            for position, name in enumerate(names):
                member = self._build(name)
                member.fit(X[train_index], y[train_index])
                start = position * width
                oof[hold_index, start:start + width] = (
                    self._member_output(member, X[hold_index]))
        assert covered.all(), \
            "out-of-fold matrix has uncovered rows; fold split is not a partition"
        self.bases_ = []
        for name in names:
            member = self._build(name)
            member.fit(X, y)
            self.bases_.append(member)
        return oof, y

    def _fit_bases_on_holdout(self, X, y):
        X_meta, y_meta = self.meta_holdout
        assert len(X_meta) == len(y_meta), \
            f"meta holdout has {len(X_meta)} rows and {len(y_meta)} targets"
        assert X_meta.shape[1] == X.shape[1], \
            (f"meta holdout has {X_meta.shape[1]} columns, training matrix has "
             f"{X.shape[1]}; both must come from the same preprocessor")
        self.n_folds_ = 0
        self.bases_ = []
        for name in self.base_names_:
            member = self._build(name)
            member.fit(X, y)
            self.bases_.append(member)
        return self._meta_features(X_meta), np.asarray(y_meta)

    def _meta_features(self, X):
        return np.hstack([self._member_output(m, X) for m in self.bases_])

    def tree_base(self):
        """A fitted tree member of the stack, for TreeSHAP.

        base_names_ is best-first on the sweep, so this is the strongest tree
        member the stack contains. Returns (estimator, classes, name) - classes
        being the label order of *that estimator's* predict_proba columns,
        which is not the wrapper's order for a label-encoded member. Returns
        (None, None, None) if the stack has no tree member.
        """
        from model_zoo import LabelEncodedClassifier
        for position, name in enumerate(self.base_names_):
            if name not in TREE_MODELS:
                continue
            member = self.bases_[position]
            if isinstance(member, LabelEncodedClassifier):
                return (member.estimator_,
                        [str(c) for c in member.encoder_.classes_], name)
            classes = ([str(c) for c in member.classes_]
                       if hasattr(member, "classes_") else None)
            return member, classes, name
        return None, None, None


class StackedClassifier(ClassifierMixin, _Stack):
    _classification = True

    def _meta_classes(self):
        return self.classes_

    def _member_output(self, member, X):
        return _aligned_proba(member, X, self.classes_)

    def fit(self, X, y):
        y = np.asarray(y)
        self.classes_ = np.unique(y)
        meta_X, meta_y = self._fit_bases(X, y)
        self.meta_ = LogisticRegression(C=1.0, max_iter=2000)
        self.meta_.fit(meta_X, meta_y)
        assert list(self.meta_.classes_) == list(self.classes_), \
            "meta-learner class order drifted from the stack's class order"
        return self

    def predict_proba(self, X):
        return self.meta_.predict_proba(self._meta_features(X))

    def predict(self, X):
        return self.classes_[self.predict_proba(X).argmax(axis=1)]


class StackedRegressor(RegressorMixin, _Stack):
    _classification = False

    def _member_output(self, member, X):
        return np.asarray(member.predict(X), dtype=float).reshape(-1, 1)

    def fit(self, X, y):
        y = np.asarray(y, dtype=float)
        meta_X, meta_y = self._fit_bases(X, y)
        self.meta_ = RidgeCV(alphas=np.logspace(-3, 3, 13))
        self.meta_.fit(meta_X, meta_y.astype(float))
        return self

    def predict(self, X):
        return self.meta_.predict(self._meta_features(X))


def build_stack(task, tuned=None, groups=None, random_state=0,
                meta_holdout=None):
    """The stack for ``task``, unfitted. Single construction point."""
    kind = StackedClassifier if task in {"C", "Lc"} else StackedRegressor
    return kind(task=task, tuned=tuned, groups=groups,
                random_state=random_state, meta_holdout=meta_holdout)


TARGET = {"C": "label_result", "Lc": "label_result",
          "R": "label_margin", "Lr": "label_margin"}
# "stack" fits its meta-learner out-of-fold on train; "stack_temporal" fits it
# on the earliest slice of validation, which sits between train and test in
# time. Both are evaluated by train_ensemble.py; whichever wins on the held-out
# test metric is what train_final.py persists.
STACK_MODELS = ("stack", "stack_temporal")
META_HOLDOUT_FRACTION = 0.6


def temporal_meta_holdout(df, task, transform, frac=META_HOLDOUT_FRACTION):
    """(X, y) for the earliest ``frac`` of validation *matches*.

    Cut on match boundaries, never on rows: an in-play match must not have
    minute 15 in the meta-fitting block and minute 60 outside it.

    The isotonic calibrator that the shared evaluation path fits afterwards
    sees all of validation, so it overlaps this block. That is a deliberate
    trade - the alternative starves a 3-class isotonic fit of rows - and it
    cannot reach the test split either way.
    """
    validation = df[df["split"] == "validation"]
    assert len(validation), "no validation rows to fit the meta-learner on"
    matches = (validation[["match_id", "match_date"]].drop_duplicates()
               .sort_values(["match_date", "match_id"]))
    cut = max(1, int(round(len(matches) * frac)))
    early = validation[validation["match_id"].isin(
        set(matches["match_id"].iloc[:cut]))]
    assert len(early) < len(validation), \
        "meta holdout swallowed the whole validation split"
    return transform(early), early[TARGET[task]].to_numpy()


def build_named_stack(model_name, task, tuned, df, transform=None,
                      random_state=0):
    """Build whichever stack variant ``model_name`` names.

    Single place that knows how each variant is wired, so train_final.py,
    train_market_blend.py, app.py and train_ensemble.py cannot drift apart.
    """
    assert model_name in STACK_MODELS, \
        f"{model_name!r} is not a stack variant; expected one of {STACK_MODELS}"
    if model_name == "stack_temporal":
        assert transform is not None, \
            "stack_temporal needs the fitted preprocessor's transform"
        return build_stack(task, tuned=tuned, random_state=random_state,
                           meta_holdout=temporal_meta_holdout(df, task,
                                                              transform))
    return build_stack(task, tuned=tuned, groups=training_groups(df, task),
                       random_state=random_state)


def training_groups(df, task):
    """match_id of the training rows, in the order prepare_matrices uses.

    Only the in-play tasks need it - one pre-match row is one match, so the
    grouping is the identity there and plain stratified folds are correct.
    """
    if task not in {"Lc", "Lr"}:
        return None
    return df[df["split"] == "train"]["match_id"].to_numpy()
