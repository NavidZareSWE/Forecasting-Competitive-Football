from g_smotenc import GSMOTENC, _geometric_continuous, _sigma_med, JointOneHot
from collections import Counter
from pathlib import Path
import sys

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))


def _toy_dataset(seed=0):
    rng = np.random.default_rng(seed)
    cont_maj = rng.normal(0.0, 1.0, size=(120, 2))
    cont_min = rng.normal(4.0, 1.0, size=(30, 2))
    nom_maj = rng.choice(["a", "b"], size=(120, 1))
    nom_min = rng.choice(["b", "c"], size=(30, 1))
    X = np.vstack([
        np.hstack([cont_maj.astype(object), nom_maj]),
        np.hstack([cont_min.astype(object), nom_min]),
    ])
    y = np.array(["maj"] * 120 + ["min"] * 30)
    return X, y


def test_balances_binary():
    X, y = _toy_dataset()
    sampler = GSMOTENC(categorical_features=[2], random_state=1)
    _, y_res = sampler.fit_resample(X, y)
    counts = Counter(y_res.tolist())
    assert counts["maj"] == counts["min"], "classes not balanced to majority count"


def test_no_original_rows_lost():
    X, y = _toy_dataset()
    sampler = GSMOTENC(categorical_features=[2], random_state=1)
    X_res, y_res = sampler.fit_resample(X, y)
    assert len(X_res) == len(X) + (120 - 30), "unexpected resampled row count"
    assert (y_res[:len(y)] == y).all(
    ), "original labels must be preserved in order"


def test_synthetic_nominal_values_are_valid():
    X, y = _toy_dataset()
    sampler = GSMOTENC(categorical_features=[2], random_state=2)
    X_res, y_res = sampler.fit_resample(X, y)
    seen = set(X[:, 2].tolist())
    synth = X_res[len(X):]
    assert set(synth[:, 2].tolist()
               ) <= seen, "synthetic nominal value not in data"


def test_synthetic_continuous_bounded():
    X, y = _toy_dataset()
    cont = X[:, :2].astype(float)
    lo, hi = cont.min(axis=0), cont.max(axis=0)
    sampler = GSMOTENC(categorical_features=[2], random_state=3)
    X_res, _ = sampler.fit_resample(X, y)
    synth = X_res[len(X):, :2].astype(float)
    assert np.isfinite(synth).all(), "non-finite synthetic continuous value"
    span = hi - lo
    assert (synth >= lo - span).all() and (synth <= hi + span).all(), \
        "synthetic continuous values fall far outside data range"


def test_reproducible_with_random_state():
    X, y = _toy_dataset()
    a = GSMOTENC(categorical_features=[2],
                 random_state=7).fit_resample(X, y)[0]
    b = GSMOTENC(categorical_features=[2],
                 random_state=7).fit_resample(X, y)[0]
    assert np.array_equal(a.astype(str), b.astype(
        str)), "not reproducible under fixed seed"


def test_deformation_one_collapses_to_segment():
    # Algorithm 3: with deformation_factor = 1 the perpendicular component is
    # removed, so the continuous sample must lie on the line through x_c along
    # (x_nn - x_c). Verified on a controlled 2-D pair.
    rng = np.random.default_rng(0)
    x_c = np.array([0.0, 0.0])
    x_nn = np.array([2.0, 0.0])
    for _ in range(200):
        gen = _geometric_continuous(x_c, x_nn, truncation=1.0,
                                    deformation=1.0, rng=rng)
        assert abs(
            gen[1]) < 1e-9, "deformation=1 did not collapse to the segment axis"


def test_sigma_med_downweights_nominal():
    X, y = _toy_dataset()
    x_min = X[y == "min"]
    sig = _sigma_med(x_min[:, :2].astype(float))
    encoder = JointOneHot([2]).fit(X)
    block = encoder.transform_nominal(x_min, weight=sig / 2.0)
    active = block[block > 0]
    assert np.allclose(
        active, sig / 2.0), "one-hot active value must equal sigma_med/2"


def test_multiclass_balances_all_minorities():
    rng = np.random.default_rng(0)
    X = np.vstack([
        np.hstack([rng.normal(0, 1, (100, 2)).astype(object),
                   rng.choice(["a", "b"], (100, 1))]),
        np.hstack([rng.normal(3, 1, (40, 2)).astype(object),
                   rng.choice(["b", "c"], (40, 1))]),
        np.hstack([rng.normal(6, 1, (25, 2)).astype(object),
                   rng.choice(["a", "c"], (25, 1))]),
    ])
    y = np.array(["H"] * 100 + ["D"] * 40 + ["A"] * 25)
    _, y_res = GSMOTENC(categorical_features=[
                        2], random_state=0).fit_resample(X, y)
    counts = Counter(y_res.tolist())
    assert counts["H"] == counts["D"] == counts["A"] == 100, "3-class balance failed"


if __name__ == "__main__":
    tests = [v for k, v in sorted(globals().items()) if k.startswith("test_")]
    passed = 0
    for test in tests:
        test()
        passed += 1
        print(f"PASS {test.__name__}")
    print(f"\n{passed}/{len(tests)} G-SMOTENC tests passed")
