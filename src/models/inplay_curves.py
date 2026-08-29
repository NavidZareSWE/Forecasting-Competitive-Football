"""Run from the repository root with:

    python src/models/inplay_curves.py
"""

from pathlib import Path
import sys

import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots

sys.path.insert(0, str(Path(__file__).resolve().parent))

from modeling_common import (CLASS_ORDER, RESULTS_DIR, classification_metrics,
                             expected_calibration_error, regression_metrics)


VIZ_DIR = Path(__file__).resolve().parent.parent / "reports" / "visualizations"

PHASE_EDGES = [(0, 15), (15, 30), (30, 45), (45, 60), (60, 75), (75, 91)]
PHASE_LABELS = ["0-15", "15-30", "30-45", "45-60", "60-75", "75-90"]

PROBABILITY_COLUMNS = [f"p_{c}" for c in CLASS_ORDER]


def _read_predictions(name):
    path = RESULTS_DIR / f"predictions_{name}.csv"
    if not path.exists():
        raise FileNotFoundError(f"Missing {path}. Run run_models.py first.")
    return pd.read_csv(path, encoding="utf-8")


def phase_of(minute):
    for label, (low, high) in zip(PHASE_LABELS, PHASE_EDGES):
        if low <= minute < high:
            return label
    return PHASE_LABELS[-1]


def prematch_reference(task, inplay_ids, columns):
    """Pre-match forecast for the in-play matches, per model.

    The frozen reference is the whole point of these curves: it answers "how
    much did watching the match actually buy over knowing only the team sheet".

    It used to come straight out of predictions_{C,R}.csv, but the extended
    split deliberately marks the StatsBomb in-play validation/test matches as
    ``excluded`` from tasks C and R, so those matches are no longer scored
    there and the join silently returned zero rows - taking the whole frozen
    series with it. When that happens, score the excluded rows here with the
    persisted serving bundle, which is exactly the model the API would use for
    a pre-match panel on those same matches.
    """
    predictions = _read_predictions(task)
    overlap = predictions[predictions["match_id"].isin(inplay_ids)]
    if len(overlap):
        return overlap, "sweep predictions"

    bundle_path = RESULTS_DIR / "models" / f"{task}.joblib"
    if not bundle_path.exists():
        print(f"  WARNING: no pre-match reference for task {task} - the in-play "
              f"matches are excluded from predictions_{task}.csv and "
              f"{bundle_path.name} is absent. Run train_final.py first.")
        return None, None

    import joblib
    from modeling_common import floor_probabilities, task_frame, _proba_in_order

    bundle = joblib.load(bundle_path)
    frame, _, _, target, _ = task_frame("C" if task == "C" else "R")
    rows = frame[frame["match_id"].isin(inplay_ids)]
    if not len(rows):
        print(f"  WARNING: none of the {len(inplay_ids)} in-play matches are in "
              f"the pre-match feature table; no frozen reference for {task}.")
        return None, None

    X = bundle["preprocessor"].transform(rows)
    out = rows[["match_id"]].copy()
    if task == "C":
        source = bundle["calibrator"] or bundle["estimator"]
        proba = floor_probabilities(_proba_in_order(source, X, CLASS_ORDER))
        for j, label in enumerate(CLASS_ORDER):
            out[f"p_{label}"] = proba[:, j]
    else:
        out["y_pred"] = np.clip(bundle["estimator"].predict(X), -5, 5)
    out["model"] = bundle["model_name"]
    missing = set(columns) - set(out.columns)
    assert not missing, f"reference is missing {sorted(missing)}"
    return out, f"served bundle ({bundle['model_name']}, excluded split)"


def freeze_prematch(inplay, prematch, columns):
    reference = prematch[["match_id", *columns]].drop_duplicates("match_id")
    merged = inplay[["match_id", "snapshot_minute", "y_true"]].merge(
        reference, on="match_id", how="inner")
    # The previous assert here was `len(merged) == len(inplay) or
    # merged.nunique() <= inplay.nunique()`. The second clause is true for any
    # subset, including the empty one, so the assert could never fire - and it
    # sat directly on top of the join that silently returned zero rows. Assert
    # the property that actually matters: every in-play match must be covered.
    covered = merged["match_id"].nunique()
    wanted = inplay["match_id"].nunique()
    assert covered == wanted, (
        f"frozen reference covers {covered} of {wanted} in-play matches. "
        "The pre-match predictions and the in-play snapshots live in different "
        "match_id spaces (the in-play matches are 'excluded' from tasks C/R), "
        "so this join needs prematch_reference()'s bundle fallback.")
    return merged


def classification_curves(rows):
    inplay = _read_predictions("Lc")
    columns = (PROBABILITY_COLUMNS if "C" == "C" else ["y_pred"])
    prematch, source = prematch_reference(
        "C", set(inplay["match_id"]), columns)
    if prematch is None:
        shared_models = []
    else:
        shared_models = sorted(set(inplay["model"]) & set(prematch["model"]))
        print(f"  frozen pre-match reference for task C: {source}, "
              f"{len(shared_models)} shared model(s)")

    for model in sorted(inplay["model"].unique()):
        subset = inplay[inplay["model"] == model]
        for minute, group in subset.groupby("snapshot_minute"):
            metrics = classification_metrics(
                group[PROBABILITY_COLUMNS].to_numpy(),
                group["y_true"].to_numpy())
            rows.append({"task": "Lc", "series": "in-play", "model": model,
                         "snapshot_minute": int(minute), "n": len(group),
                         **{k: round(v, 5) for k, v in metrics.items()}})

    for model in shared_models:
        frozen = freeze_prematch(inplay[inplay["model"] == model],
                                 prematch[prematch["model"] == model],
                                 PROBABILITY_COLUMNS)
        for minute, group in frozen.groupby("snapshot_minute"):
            metrics = classification_metrics(
                group[PROBABILITY_COLUMNS].to_numpy(),
                group["y_true"].to_numpy())
            rows.append({"task": "Lc", "series": "frozen pre-match",
                         "model": model, "snapshot_minute": int(minute),
                         "n": len(group),
                         **{k: round(v, 5) for k, v in metrics.items()}})
    return shared_models


def regression_curves(rows):
    inplay = _read_predictions("Lr")
    columns = (PROBABILITY_COLUMNS if "R" == "C" else ["y_pred"])
    prematch, source = prematch_reference(
        "R", set(inplay["match_id"]), columns)
    if prematch is None:
        shared_models = []
    else:
        shared_models = sorted(set(inplay["model"]) & set(prematch["model"]))
        print(f"  frozen pre-match reference for task R: {source}, "
              f"{len(shared_models)} shared model(s)")

    for model in sorted(inplay["model"].unique()):
        subset = inplay[inplay["model"] == model]
        for minute, group in subset.groupby("snapshot_minute"):
            metrics = regression_metrics(group["y_pred"].to_numpy(),
                                         group["y_true"].to_numpy())
            rows.append({"task": "Lr", "series": "in-play", "model": model,
                         "snapshot_minute": int(minute), "n": len(group),
                         **{k: round(v, 5) for k, v in metrics.items()}})

    for model in shared_models:
        frozen = freeze_prematch(inplay[inplay["model"] == model],
                                 prematch[prematch["model"] == model],
                                 ["y_pred"])
        for minute, group in frozen.groupby("snapshot_minute"):
            metrics = regression_metrics(group["y_pred"].to_numpy(),
                                         group["y_true"].to_numpy())
            rows.append({"task": "Lr", "series": "frozen pre-match",
                         "model": model, "snapshot_minute": int(minute),
                         "n": len(group),
                         **{k: round(v, 5) for k, v in metrics.items()}})
    return shared_models


def calibration_by_phase():
    inplay = _read_predictions("Lc")
    inplay = inplay.assign(phase=inplay["snapshot_minute"].map(phase_of))
    rows = []
    for (model, phase), group in inplay.groupby(["model", "phase"]):
        proba = group[PROBABILITY_COLUMNS].to_numpy()
        y_true = group["y_true"].to_numpy()
        confidence = proba.max(axis=1)
        predicted = np.array([CLASS_ORDER[i] for i in proba.argmax(axis=1)])
        rows.append({
            "model": model, "phase": phase, "n": len(group),
            "ece": round(expected_calibration_error(proba, y_true), 5),
            "mean_confidence": round(float(confidence.mean()), 5),
            "accuracy": round(float((predicted == y_true).mean()), 5),
            "over_confidence": round(
                float(confidence.mean() - (predicted == y_true).mean()), 5),
            **{k: round(v, 5)
               for k, v in classification_metrics(proba, y_true).items()},
        })
    frame = pd.DataFrame(rows)
    frame["phase"] = pd.Categorical(frame["phase"], PHASE_LABELS, ordered=True)
    return frame.sort_values(["model", "phase"]).reset_index(drop=True)


def plot(curves, phases, output_path):
    figure = make_subplots(
        rows=2, cols=2,
        subplot_titles=("Task L classification - RPS by minute",
                        "Task L regression - MAE by minute",
                        "Task L classification - ECE by minute",
                        "Calibration gap by game phase (confidence - accuracy)"))

    def add(frame, metric, row, col, show_legend):
        for (model, series), group in frame.groupby(["model", "series"]):
            group = group.sort_values("snapshot_minute")
            frozen = series == "frozen pre-match"
            figure.add_trace(go.Scatter(
                x=group["snapshot_minute"], y=group[metric],
                mode="lines+markers",
                name=f"{model} ({series})",
                legendgroup=f"{model}|{series}",
                showlegend=show_legend,
                line=dict(dash="dash" if frozen else "solid",
                          width=2 if frozen else 1.6)),
                row=row, col=col)

    add(curves[curves["task"] == "Lc"], "rps", 1, 1, True)
    add(curves[curves["task"] == "Lr"], "mae", 1, 2, False)
    add(curves[curves["task"] == "Lc"], "ece", 2, 1, False)

    for model, group in phases.groupby("model"):
        figure.add_trace(go.Bar(x=group["phase"].astype(str),
                                y=group["over_confidence"], name=str(model),
                                legendgroup=f"{model}|phase",
                                showlegend=False), row=2, col=2)

    figure.update_xaxes(title_text="snapshot minute", row=1, col=1)
    figure.update_xaxes(title_text="snapshot minute", row=1, col=2)
    figure.update_xaxes(title_text="snapshot minute", row=2, col=1)
    figure.update_layout(
        height=900, template="plotly_white", barmode="group",
        title_text="Model 3 vs. the frozen pre-match reference "
                   "(dashed = frozen pre-match prediction)")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    figure.write_html(str(output_path), include_plotlyjs="cdn")


def main():
    rows = []
    classification_curves(rows)
    regression_curves(rows)
    curves = pd.DataFrame(rows)
    phases = calibration_by_phase()

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    curves.to_csv(RESULTS_DIR / "inplay_metric_by_minute.csv", index=False,
                  encoding="utf-8")
    phases.to_csv(RESULTS_DIR / "inplay_calibration_by_phase.csv", index=False,
                  encoding="utf-8")
    plot(curves, phases, VIZ_DIR / "inplay_curves.html")

    frozen = curves[(curves["series"] == "frozen pre-match")
                    & (curves["task"] == "Lc")]
    for model, group in frozen.groupby("model"):
        spread = group["rps"].max() - group["rps"].min()
        assert spread < 1e-9, \
            f"frozen pre-match reference for {model} varies by minute ({spread})"

    print("Task L classification, RPS by minute (in-play vs frozen):")
    pivot = (curves[curves["task"] == "Lc"]
             .pivot_table(index="snapshot_minute", columns="series",
                          values="rps", aggfunc="min"))
    print(pivot.round(5).to_string())
    print("\nCalibration gap by phase (mean over models):")
    print(phases.groupby("phase", observed=True)[
        ["ece", "mean_confidence", "accuracy", "over_confidence"]]
        .mean().round(4).to_string())
    print(f"\nWrote -> {RESULTS_DIR / 'inplay_metric_by_minute.csv'}")
    print(f"Wrote -> {RESULTS_DIR / 'inplay_calibration_by_phase.csv'}")
    print(f"Wrote -> {VIZ_DIR / 'inplay_curves.html'}")


if __name__ == "__main__":
    main()
