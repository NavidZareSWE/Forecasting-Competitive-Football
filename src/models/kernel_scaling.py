"""Run from the repository root with:

    python src/models/kernel_scaling.py
"""

from pathlib import Path
import sys

import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from sklearn.kernel_approximation import Nystroem
from sklearn.linear_model import Ridge
from sklearn.kernel_ridge import KernelRidge
from sklearn.pipeline import make_pipeline
from sklearn.svm import SVR

sys.path.insert(0, str(Path(__file__).resolve().parent))

from modeling_common import RESULTS_DIR, fit_with_cost, prepare_matrices, task_frame


VIZ_DIR = Path(__file__).resolve().parent.parent / "reports" / "visualizations"

SIZES = [500, 1000, 2000, 4000, 8000, 12000]
RANDOM_STATE = 0

THEORY = {"kernel_ridge_exact": ("O(n^2) space, O(n^3) solve", 3.0),
          "kernel_svr_exact": ("O(n^2) space, ~O(n^2..3) SMO", 2.0),
          "kernel_ridge_nystroem": ("O(nm) space, O(nm^2) fit", 1.0)}


def methods():
    return {
        "kernel_ridge_exact": lambda: KernelRidge(kernel="rbf", alpha=1.0),
        "kernel_svr_exact": lambda: SVR(kernel="rbf", C=1.0, gamma="scale"),
        "kernel_ridge_nystroem": lambda: make_pipeline(
            Nystroem(kernel="rbf", n_components=500,
                     random_state=RANDOM_STATE),
            Ridge(alpha=1.0)),
    }


def fit_power_law(sizes, seconds):
    sizes = np.asarray(sizes, dtype=float)
    seconds = np.asarray(seconds, dtype=float)
    usable = seconds > 0
    if usable.sum() < 2:
        return float("nan"), float("nan")
    slope, intercept = np.polyfit(np.log(sizes[usable]),
                                  np.log(seconds[usable]), 1)
    return float(slope), float(np.exp(intercept))


def main():
    df, continuous, nominal, target, task_type = task_frame("Lr")
    matrices = prepare_matrices(df, continuous, nominal, target, task_type)
    X = matrices["X_train"]
    y = matrices["y_train"].astype(float)
    available = X.shape[0]
    sizes = [n for n in SIZES if n <= available]
    if sizes[-1] < available:
        sizes.append(available)
    print(f"In-play training matrix: {available} rows x {X.shape[1]} features")
    print(f"Sizes measured: {sizes}\n")

    rng = np.random.default_rng(RANDOM_STATE)
    order = rng.permutation(available)

    rows = []
    for name, factory in methods().items():
        for n in sizes:
            nested_subsample = np.sort(order[:n])
            seconds, peak_mb = fit_with_cost(factory(), X[nested_subsample],
                                             y[nested_subsample])
            gram_mb = (n * n * 8) / 1024 ** 2
            rows.append({"method": name, "n_train": n,
                         "fit_seconds": round(seconds, 4),
                         "peak_memory_mb": round(peak_mb, 1),
                         "theoretical_gram_mb": round(gram_mb, 1),
                         "exact": "nystroem" not in name})
            print(f"  {name:24s} n={n:6d}  {seconds:8.3f}s  "
                  f"peak {peak_mb:7.1f} MB  (Gram would be {gram_mb:.0f} MB)")

    results = pd.DataFrame(rows)

    fits = []
    print()
    for name, group in results.groupby("method"):
        exponent, coefficient = fit_power_law(group["n_train"],
                                              group["fit_seconds"])
        description, theoretical = THEORY[name]
        fits.append({"method": name, "empirical_exponent": round(exponent, 3),
                     "theoretical_exponent": theoretical,
                     "theory": description,
                     "coefficient": coefficient})
        print(f"  {name:24s} t ~ n^{exponent:.2f}   "
              f"(theory: {description})")
    fit_frame = pd.DataFrame(fits)
    results = results.merge(fit_frame[["method", "empirical_exponent",
                                       "theoretical_exponent", "theory"]],
                            on="method", how="left")

    exact_exponents = fit_frame[fit_frame["method"].str.contains("exact")]
    nystroem = fit_frame[fit_frame["method"] == "kernel_ridge_nystroem"]
    assert (exact_exponents["empirical_exponent"] > 1.2).all(), \
        f"exact kernels did not scale super-linearly: {fits}"
    assert nystroem["empirical_exponent"].iloc[0] < \
        exact_exponents["empirical_exponent"].min(), \
        "Nystroem did not scale better than the exact kernels"

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    results.to_csv(RESULTS_DIR / "kernel_scaling.csv", index=False,
                   encoding="utf-8")

    figure = make_subplots(rows=1, cols=2,
                           subplot_titles=("Fit time vs training size (log-log)",
                                           "Peak memory vs training size"))
    for name, group in results.groupby("method"):
        group = group.sort_values("n_train")
        exponent = group["empirical_exponent"].iloc[0]
        figure.add_trace(go.Scatter(x=group["n_train"], y=group["fit_seconds"],
                                    mode="lines+markers",
                                    name=f"{name} (n^{exponent:.2f})"),
                         row=1, col=1)
        figure.add_trace(go.Scatter(x=group["n_train"],
                                    y=group["peak_memory_mb"],
                                    mode="lines+markers", name=name,
                                    showlegend=False), row=1, col=2)
    reference = results[results["method"] == "kernel_ridge_exact"].sort_values("n_train")
    figure.add_trace(go.Scatter(x=reference["n_train"],
                                y=reference["theoretical_gram_mb"],
                                mode="lines", name="theoretical Gram matrix",
                                line=dict(dash="dot", color="black")),
                     row=1, col=2)
    figure.update_xaxes(type="log", title_text="training rows n")
    figure.update_yaxes(type="log", title_text="seconds", row=1, col=1)
    figure.update_yaxes(type="log", title_text="MB", row=1, col=2)
    figure.update_layout(height=520, template="plotly_white",
                         title_text="Kernel scaling: exact vs Nystroem "
                                    "approximation on the in-play table")
    VIZ_DIR.mkdir(parents=True, exist_ok=True)
    figure.write_html(str(VIZ_DIR / "kernel_scaling.html"),
                      include_plotlyjs="cdn")

    print(f"\nWrote -> {RESULTS_DIR / 'kernel_scaling.csv'}")
    print(f"Wrote -> {VIZ_DIR / 'kernel_scaling.html'}")


if __name__ == "__main__":
    main()
