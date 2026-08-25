"""Run from the repository root with:

    python src/viz/plot_calibration.py
"""

from pathlib import Path

import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots


PROJECT = Path(__file__).resolve().parent.parent
RESULTS_DIR = PROJECT / "reports"
VIZ_DIR = RESULTS_DIR / "visualizations"

CLASS_ORDER = ["H", "D", "A"]
N_BINS = 10


def reliability_table(proba, y_true, bins=N_BINS):
    confidence = proba.max(axis=1)
    predicted = np.array([CLASS_ORDER[i] for i in proba.argmax(axis=1)])
    correct = (predicted == np.asarray(y_true)).astype(float)
    edges = np.linspace(0.0, 1.0, bins + 1)
    rows = []
    for low, high in zip(edges[:-1], edges[1:]):
        mask = (confidence > low) & (confidence <= high)
        if not mask.any():
            continue
        rows.append({"bin_low": round(low, 3), "bin_high": round(high, 3),
                     "n": int(mask.sum()),
                     "mean_confidence": float(confidence[mask].mean()),
                     "accuracy": float(correct[mask].mean()),
                     "share": float(mask.mean())})
    return pd.DataFrame(rows)


def _read_predictions(task):
    path = RESULTS_DIR / f"predictions_{task}.csv"
    if not path.exists():
        raise FileNotFoundError(f"Missing {path}. Run run_models.py first.")
    return pd.read_csv(path, encoding="utf-8")


def collect(task, rows):
    predictions = _read_predictions(task)
    for model, group in predictions.groupby("model"):
        y_true = group["y_true"].to_numpy()
        for stage, prefix in [("raw", "raw_"), ("calibrated", "p_")]:
            columns = [f"{prefix}{c}" for c in CLASS_ORDER]
            if not all(c in group.columns for c in columns):
                continue
            table = reliability_table(group[columns].to_numpy(), y_true)
            table.insert(0, "stage", stage)
            table.insert(0, "model", model)
            table.insert(0, "task", task)
            rows.append(table)
    return predictions


def figure_for(bins, task, title):
    subset = bins[bins["task"] == task]
    models = sorted(subset["model"].unique())
    columns = 2
    rows_needed = int(np.ceil(len(models) / columns)) or 1
    figure = make_subplots(rows=rows_needed, cols=columns,
                           subplot_titles=models,
                           shared_xaxes=True, shared_yaxes=True)
    for position, model in enumerate(models):
        row = position // columns + 1
        col = position % columns + 1
        figure.add_trace(go.Scatter(x=[0, 1], y=[0, 1], mode="lines",
                                    line=dict(dash="dot", color="grey"),
                                    showlegend=False), row=row, col=col)
        for stage, colour in [("raw", "#d62728"), ("calibrated", "#1f77b4")]:
            group = subset[(subset["model"] == model)
                           & (subset["stage"] == stage)].sort_values("bin_low")
            if group.empty:
                continue
            figure.add_trace(go.Scatter(
                x=group["mean_confidence"], y=group["accuracy"],
                mode="lines+markers", name=stage, legendgroup=stage,
                showlegend=(position == 0), marker=dict(size=6, color=colour),
                line=dict(color=colour),
                text=[f"n={n}" for n in group["n"]]), row=row, col=col)
    figure.update_xaxes(title_text="mean predicted confidence", range=[0, 1])
    figure.update_yaxes(title_text="observed accuracy", range=[0, 1])
    figure.update_layout(height=340 * rows_needed, template="plotly_white",
                         title_text=title)
    return figure


def main():
    rows = []
    for task in ["C", "Lc"]:
        collect(task, rows)
    bins = pd.concat(rows, ignore_index=True)

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    VIZ_DIR.mkdir(parents=True, exist_ok=True)
    bins.round(5).to_csv(RESULTS_DIR / "reliability_bins.csv", index=False,
                         encoding="utf-8")

    parts = []
    for task, title in [("C", "Task C - pre-match outcome"),
                        ("Lc", "Task L - in-play outcome (all snapshots)")]:
        if (bins["task"] == task).any():
            figure = figure_for(bins, task, f"Reliability diagrams - {title}")
            parts.append(f"<h2>{title}</h2>")
            parts.append(figure.to_html(full_html=False,
                                        include_plotlyjs="cdn" if not parts
                                        else False))

    output_path = VIZ_DIR / "reliability_diagrams.html"
    output_path.write_text(
        "<html><head><meta charset='utf-8'>"
        "<title>Reliability diagrams</title></head><body>"
        "<h1>Reliability diagrams (red = raw, blue = after calibration)</h1>"
        + "".join(parts) + "</body></html>", encoding="utf-8")

    print(f"Wrote {len(bins)} reliability bins -> "
          f"{RESULTS_DIR / 'reliability_bins.csv'}")
    print(f"Wrote -> {output_path}")


if __name__ == "__main__":
    main()
