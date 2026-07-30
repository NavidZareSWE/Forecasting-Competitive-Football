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


EXACT_KERNEL_MAX_TRAIN = 8000


class SubsampledKernelRidge(RegressorMixin, BaseEstimator):
    # Fits an exact RBF kernel ridge model on a random subset of the training
    # data. The subset is reproducible, and the number of available and used
    # training rows is stored.
    def __init__(self, alpha=1.0, kernel="rbf", max_train_rows=None,
                 random_state=None):
        self.alpha = alpha
        self.kernel = kernel
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
        self.estimator_ = KernelRidge(kernel=self.kernel, alpha=self.alpha)
        self.estimator_.fit(X, y)
        return self

    def predict(self, X):
        return self.estimator_.predict(X)


# --- String-label adapter for integer-only classifiers ---------------------
class LabelEncodedClassifier(ClassifierMixin, BaseEstimator):
    # XGBClassifier only accepts integer labels. Project labels are encoded for
    # training and decoded for prediction, preserving sklearn's class order.
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


# --- Guarded optional boosters ---------------------------------------------
def _xgboost_models(random_state):
    try:
        from xgboost import XGBClassifier, XGBRegressor
    except Exception:
        return {}, {}
    classifiers = {"xgboost": lambda: LabelEncodedClassifier(XGBClassifier(
        n_estimators=300, max_depth=4, learning_rate=0.05,
        subsample=0.8, colsample_bytree=0.8, tree_method="hist",
        eval_metric="mlogloss", random_state=random_state))}
    regressors = {"xgboost": lambda: XGBRegressor(
        n_estimators=300, max_depth=4, learning_rate=0.05,
        subsample=0.8, colsample_bytree=0.8, tree_method="hist",
        random_state=random_state)}
    return classifiers, regressors


def _lightgbm_models(random_state):
    try:
        from lightgbm import LGBMClassifier, LGBMRegressor
    except Exception:
        return {}, {}
    classifiers = {"lightgbm": lambda: LGBMClassifier(
        n_estimators=400, num_leaves=31, learning_rate=0.05,
        subsample=0.8, colsample_bytree=0.8, verbose=-1,
        random_state=random_state)}
    regressors = {"lightgbm": lambda: LGBMRegressor(
        n_estimators=400, num_leaves=31, learning_rate=0.05,
        subsample=0.8, colsample_bytree=0.8, verbose=-1,
        random_state=random_state)}
    return classifiers, regressors


def classifier_zoo(random_state=0):
    classifiers = {
        "dummy": lambda: DummyClassifier(strategy="prior"),
        "random_forest": lambda: RandomForestClassifier(
            n_estimators=300, max_depth=None, n_jobs=-1,
            random_state=random_state),
        "gbm": lambda: HistGradientBoostingClassifier(
            max_iter=300, learning_rate=0.05, random_state=random_state),
        "kernel_svm": lambda: SVC(
            kernel="rbf", C=1.0, gamma="scale", probability=True,
            random_state=random_state),
    }
    xgb_classifiers, _ = _xgboost_models(random_state)
    lgb_classifiers, _ = _lightgbm_models(random_state)
    classifiers.update(xgb_classifiers)
    classifiers.update(lgb_classifiers)
    return classifiers


def regressor_zoo(random_state=0):
    regressors = {
        "dummy": lambda: DummyRegressor(strategy="mean"),
        "random_forest": lambda: RandomForestRegressor(
            n_estimators=300, max_depth=None, n_jobs=-1,
            random_state=random_state),
        "gbm": lambda: HistGradientBoostingRegressor(
            max_iter=300, learning_rate=0.05, random_state=random_state),
        "kernel_svr": lambda: SVR(kernel="rbf", C=1.0, gamma="scale"),
        "kernel_ridge_exact": lambda: SubsampledKernelRidge(
            alpha=1.0, kernel="rbf", max_train_rows=EXACT_KERNEL_MAX_TRAIN,
            random_state=random_state),
        "kernel_ridge_nystroem": lambda: make_pipeline(
            Nystroem(kernel="rbf", n_components=500,
                     random_state=random_state),
            Ridge(alpha=1.0)),
    }
    _, xgb_regressors = _xgboost_models(random_state)
    _, lgb_regressors = _lightgbm_models(random_state)
    regressors.update(xgb_regressors)
    regressors.update(lgb_regressors)
    return regressors
