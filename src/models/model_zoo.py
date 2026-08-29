from pathlib import Path
import sys

import numpy as np
from sklearn.base import (BaseEstimator, ClassifierMixin, RegressorMixin,
                          clone)
from sklearn.dummy import DummyClassifier, DummyRegressor
from sklearn.ensemble import (RandomForestClassifier, RandomForestRegressor,
                              HistGradientBoostingClassifier,
                              HistGradientBoostingRegressor)
from sklearn.kernel_approximation import Nystroem
from sklearn.linear_model import Ridge
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import LabelEncoder
from sklearn.svm import SVC, SVR
from sklearn.kernel_ridge import KernelRidge

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "papers"))
from hierarchical_shrinkage import (HSForestClassifier, HSForestRegressor,
                                    LAMBDA_GRID)


EXACT_KERNEL_MAX_TRAIN = 8000

# The stacked ensemble (stacking.py) is deliberately NOT a zoo member: its
# folds must be grouped by match_id for the in-play tasks, and a zoo factory
# has no way to receive them. Callers that serve it build it through
# stacking.build_stack; every zoo loop here therefore stays stack-free.


class ClampedNystroem(Nystroem):

    def fit(self, X, y=None):
        self.n_components = min(self.n_components, X.shape[0])
        return super().fit(X, y)


class SubsampledKernelRidge(RegressorMixin, BaseEstimator):
    def __init__(self, alpha=1.0, kernel="rbf", gamma=None,
                 max_train_rows=None, random_state=None):
        self.alpha = alpha
        self.kernel = kernel
        self.gamma = gamma
        self.max_train_rows = max_train_rows
        self.random_state = random_state

    def fit(self, X, y):
        self.n_train_available_ = int(X.shape[0])
        cap = self.max_train_rows
        if cap is not None and self.n_train_available_ > cap:
            rng = np.random.default_rng(self.random_state)
            index = rng.choice(self.n_train_available_,
                               size=cap, replace=False)
            index.sort()
            X, y = X[index], y[index]
            self.subsampled_ = True
        else:
            self.subsampled_ = False
        self.n_train_used_ = int(X.shape[0])
        self.estimator_ = KernelRidge(kernel=self.kernel, alpha=self.alpha,
                                      gamma=self.gamma)
        self.estimator_.fit(X, y)
        return self

    def predict(self, X):
        return self.estimator_.predict(X)


class LabelEncodedClassifier(ClassifierMixin, BaseEstimator):
    def __init__(self, estimator=None):
        self.estimator = estimator

    def fit(self, X, y):
        self.encoder_ = LabelEncoder().fit(y)
        self.classes_ = self.encoder_.classes_
        self.estimator_ = clone(self.estimator)
        self.estimator_.fit(X, self.encoder_.transform(y))
        return self

    def predict_proba(self, X):
        return self.estimator_.predict_proba(X)

    def predict(self, X):
        return self.encoder_.inverse_transform(self.estimator_.predict(X))


def _has(module):
    try:
        __import__(module)
        return True
    except Exception:
        return False


HAS_XGBOOST = _has("xgboost")
HAS_LIGHTGBM = _has("lightgbm")

for _library, _present in [("xgboost", HAS_XGBOOST),
                           ("lightgbm", HAS_LIGHTGBM)]:
    if not _present:
        print(f"WARNING: {_library} is not importable. The zoo will run "
              f"without it and the sweep will silently come back one "
              f"model short. On macOS this is usually a missing libomp.")


TASK_SUPPORT = {
    "dummy": {"C", "R", "L"},
    "kernel_svm": {"C"},
    "kernel_svr": {"R"},
    "kernel_ridge_exact": {"R"},
    "kernel_ridge_nystroem": {"R"},
    "random_forest": {"C", "R", "L"},
    "gbm": {"C", "R", "L"},
    "xgboost": {"C", "R", "L"},
    "lightgbm": {"C", "R", "L"},
    "p2_hier_shrinkage": {"C", "R", "L"},
}


def _classifier_builders():
    builders = {
        "dummy": lambda rs, **p: DummyClassifier(strategy="prior"),
        "random_forest": lambda rs, **p: RandomForestClassifier(
            n_estimators=p.get("n_estimators", 300),
            max_depth=p.get("max_depth", None),
            min_samples_leaf=p.get("min_samples_leaf", 1),
            max_features=p.get("max_features", "sqrt"),
            class_weight=p.get("class_weight", None),
            n_jobs=-1, random_state=rs),
        "gbm": lambda rs, **p: HistGradientBoostingClassifier(
            max_iter=p.get("max_iter", 300),
            learning_rate=p.get("learning_rate", 0.05),
            max_leaf_nodes=p.get("max_leaf_nodes", 31),
            min_samples_leaf=p.get("min_samples_leaf", 20),
            l2_regularization=p.get("l2_regularization", 0.0),
            class_weight=p.get("class_weight", None),
            random_state=rs),
        "kernel_svm": lambda rs, **p: SVC(
            kernel="rbf", C=p.get("C", 1.0), gamma=p.get("gamma", "scale"),
            class_weight=p.get("class_weight", None),
            probability=True, random_state=rs),
        "p2_hier_shrinkage": lambda rs, **p: HSForestClassifier(
            lam=p.get("lam", 25.0),
            cv_lambdas=p.get("cv_lambdas", LAMBDA_GRID),
            n_estimators=p.get("n_estimators", 300),
            max_depth=p.get("max_depth", None),
            n_jobs=-1, random_state=rs),
    }
    if HAS_XGBOOST:
        from xgboost import XGBClassifier
        builders["xgboost"] = lambda rs, **p: LabelEncodedClassifier(
            XGBClassifier(
                n_estimators=p.get("n_estimators", 300),
                max_depth=p.get("max_depth", 4),
                learning_rate=p.get("learning_rate", 0.05),
                subsample=p.get("subsample", 0.8),
                colsample_bytree=p.get("colsample_bytree", 0.8),
                reg_lambda=p.get("reg_lambda", 1.0),
                tree_method="hist", eval_metric="mlogloss", random_state=rs))
    if HAS_LIGHTGBM:
        from lightgbm import LGBMClassifier
        builders["lightgbm"] = lambda rs, **p: LGBMClassifier(
            n_estimators=p.get("n_estimators", 400),
            num_leaves=p.get("num_leaves", 31),
            learning_rate=p.get("learning_rate", 0.05),
            subsample=p.get("subsample", 0.8),
            colsample_bytree=p.get("colsample_bytree", 0.8),
            reg_lambda=p.get("reg_lambda", 0.0),
            class_weight=p.get("class_weight", None),
            subsample_freq=1, verbose=-1, random_state=rs)
    return builders


def _regressor_builders():
    builders = {
        "dummy": lambda rs, **p: DummyRegressor(strategy="mean"),
        "random_forest": lambda rs, **p: RandomForestRegressor(
            n_estimators=p.get("n_estimators", 300),
            max_depth=p.get("max_depth", None),
            min_samples_leaf=p.get("min_samples_leaf", 1),
            max_features=p.get("max_features", 1.0),
            n_jobs=-1, random_state=rs),
        "gbm": lambda rs, **p: HistGradientBoostingRegressor(
            max_iter=p.get("max_iter", 300),
            learning_rate=p.get("learning_rate", 0.05),
            max_leaf_nodes=p.get("max_leaf_nodes", 31),
            min_samples_leaf=p.get("min_samples_leaf", 20),
            l2_regularization=p.get("l2_regularization", 0.0),
            random_state=rs),
        "kernel_svr": lambda rs, **p: SVR(
            kernel="rbf", C=p.get("C", 1.0), gamma=p.get("gamma", "scale"),
            epsilon=p.get("epsilon", 0.1)),
        "kernel_ridge_exact": lambda rs, **p: SubsampledKernelRidge(
            alpha=p.get("alpha", 1.0), kernel="rbf", gamma=p.get("gamma", None),
            max_train_rows=EXACT_KERNEL_MAX_TRAIN, random_state=rs),
        "kernel_ridge_nystroem": lambda rs, **p: make_pipeline(
            ClampedNystroem(kernel="rbf", gamma=p.get("gamma", None),
                            n_components=p.get("n_components", 500),
                            random_state=rs),
            Ridge(alpha=p.get("alpha", 1.0))),
        "p2_hier_shrinkage": lambda rs, **p: HSForestRegressor(
            lam=p.get("lam", 25.0),
            cv_lambdas=p.get("cv_lambdas", LAMBDA_GRID),
            n_estimators=p.get("n_estimators", 300),
            max_depth=p.get("max_depth", None),
            n_jobs=-1, random_state=rs),
    }
    if HAS_XGBOOST:
        from xgboost import XGBRegressor
        builders["xgboost"] = lambda rs, **p: XGBRegressor(
            n_estimators=p.get("n_estimators", 300),
            max_depth=p.get("max_depth", 4),
            learning_rate=p.get("learning_rate", 0.05),
            subsample=p.get("subsample", 0.8),
            colsample_bytree=p.get("colsample_bytree", 0.8),
            reg_lambda=p.get("reg_lambda", 1.0),
            tree_method="hist", random_state=rs)
    if HAS_LIGHTGBM:
        from lightgbm import LGBMRegressor
        builders["lightgbm"] = lambda rs, **p: LGBMRegressor(
            n_estimators=p.get("n_estimators", 400),
            num_leaves=p.get("num_leaves", 31),
            learning_rate=p.get("learning_rate", 0.05),
            subsample=p.get("subsample", 0.8),
            colsample_bytree=p.get("colsample_bytree", 0.8),
            reg_lambda=p.get("reg_lambda", 0.0),
            subsample_freq=1, verbose=-1, random_state=rs)
    return builders


CLASSIFIER_SPACES = {
    "random_forest": {"n_estimators": [200, 300, 500],
                      "max_depth": [None, 8, 14, 20],
                      "min_samples_leaf": [1, 3, 10, 25],
                      "max_features": ["sqrt", 0.3, 0.6]},
    "gbm": {"max_iter": [200, 300, 500],
            "learning_rate": [0.02, 0.05, 0.1],
            "max_leaf_nodes": [15, 31, 63],
            "min_samples_leaf": [10, 20, 50],
            "l2_regularization": [0.0, 0.1, 1.0]},
    "kernel_svm": {"C": [0.1, 0.5, 1.0, 3.0, 10.0],
                   "gamma": ["scale", 0.01, 0.03, 0.1]},
    "xgboost": {"n_estimators": [200, 300, 500],
                "max_depth": [3, 4, 6, 8],
                "learning_rate": [0.02, 0.05, 0.1],
                "subsample": [0.6, 0.8, 1.0],
                "colsample_bytree": [0.6, 0.8, 1.0],
                "reg_lambda": [0.5, 1.0, 5.0]},
    "lightgbm": {"n_estimators": [200, 400, 600],
                 "num_leaves": [15, 31, 63],
                 "learning_rate": [0.02, 0.05, 0.1],
                 "subsample": [0.6, 0.8, 1.0],
                 "colsample_bytree": [0.6, 0.8, 1.0],
                 "reg_lambda": [0.0, 1.0, 5.0]},
    "p2_hier_shrinkage": {"n_estimators": [200, 300, 500],
                          "max_depth": [None, 8, 14, 20]},
}

REGRESSOR_SPACES = {
    "random_forest": {"n_estimators": [200, 300, 500],
                      "max_depth": [None, 8, 14, 20],
                      "min_samples_leaf": [1, 3, 10, 25],
                      "max_features": [1.0, 0.3, 0.6]},
    "gbm": {"max_iter": [200, 300, 500],
            "learning_rate": [0.02, 0.05, 0.1],
            "max_leaf_nodes": [15, 31, 63],
            "min_samples_leaf": [10, 20, 50],
            "l2_regularization": [0.0, 0.1, 1.0]},
    "kernel_svr": {"C": [0.1, 0.5, 1.0, 3.0, 10.0],
                   "gamma": ["scale", 0.01, 0.03, 0.1],
                   "epsilon": [0.05, 0.1, 0.3]},
    "kernel_ridge_exact": {"alpha": [0.1, 0.3, 1.0, 3.0, 10.0],
                           "gamma": [None, 0.003, 0.01, 0.03]},
    "kernel_ridge_nystroem": {"alpha": [0.1, 0.3, 1.0, 3.0, 10.0],
                              "gamma": [None, 0.003, 0.01, 0.03],
                              "n_components": [250, 500, 1000]},
    "xgboost": {"n_estimators": [200, 300, 500],
                "max_depth": [3, 4, 6, 8],
                "learning_rate": [0.02, 0.05, 0.1],
                "subsample": [0.6, 0.8, 1.0],
                "colsample_bytree": [0.6, 0.8, 1.0],
                "reg_lambda": [0.5, 1.0, 5.0]},
    "lightgbm": {"n_estimators": [200, 400, 600],
                 "num_leaves": [15, 31, 63],
                 "learning_rate": [0.02, 0.05, 0.1],
                 "subsample": [0.6, 0.8, 1.0],
                 "colsample_bytree": [0.6, 0.8, 1.0],
                 "reg_lambda": [0.0, 1.0, 5.0]},
    "p2_hier_shrinkage": {"n_estimators": [200, 300, 500],
                          "max_depth": [None, 8, 14, 20]},
}


def _zoo(builders, random_state, task, tuned):
    tuned = tuned or {}
    zoo = {}
    for name, builder in builders.items():
        if task is not None and task not in TASK_SUPPORT.get(name, set()):
            continue
        params = dict(tuned.get(name, {}))
        zoo[name] = (lambda b=builder, p=params: b(random_state, **p))
    return zoo


def classifier_zoo(random_state=0, task=None, tuned=None):
    return _zoo(_classifier_builders(), random_state, task, tuned)


def regressor_zoo(random_state=0, task=None, tuned=None):
    return _zoo(_regressor_builders(), random_state, task, tuned)
