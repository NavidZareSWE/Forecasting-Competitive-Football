"""Imbalance study on Task C: vanilla, smote, borderline_smote, adasyn,
class_weight and P1 (G-SMOTENC) on identical splits. Pre-match table only.

    python src/models/resampling_study.py
"""

import pandas as pd

from modeling_common import task_frame, RESULTS_DIR
from model_zoo import classifier_zoo
from run_models import PER_CLASS_COLUMNS
from train import evaluate_classification
from tuning import load_best_params


TASK = "C"

# class_weight reweights the loss instead of inventing rows.
ARMS = [("vanilla", "none", None),
        ("smote", "smote", None),
        ("borderline_smote", "borderline_smote", None),
        ("adasyn", "adasyn", None),
        ("class_weight", "none", "balanced"),
        ("p1_gsmotenc", "p1", None)]

# Models accepting class_weight, so every arm runs on the same learners.
STUDY_MODELS = ["random_forest", "gbm", "kernel_svm", "lightgbm"]


def main():
    df, continuous, nominal, target, task_type = task_frame(TASK)
    data = (df, continuous, nominal, target, task_type)
    tuned = load_best_params().get(TASK, {})

    counts = pd.Series(df[df["split"] == "train"][target]).value_counts()
    print(f"Training class balance: {counts.to_dict()}  "
          f"(draw share {counts.get('D', 0) / counts.sum():.3f})")

    rows = []
    for arm_name, resampling, class_weight in ARMS:
        zoo = classifier_zoo(0, task="C", tuned=tuned)
        print(f"\n--- arm: {arm_name} ---")
        for model_name in STUDY_MODELS:
            if model_name not in zoo:
                print(f"  {model_name}: unavailable, skipped")
                continue
            params = dict(tuned.get(model_name, {}))
            if class_weight is not None:
                params["class_weight"] = class_weight
            factory = classifier_zoo(
                0, task="C", tuned={model_name: params})[model_name]
            result, _ = evaluate_classification(model_name, factory, data,
                                                resampling)
            result["arm"] = arm_name
            result["task"] = TASK
            rows.append(result)
            print(f"  {model_name:16s} rps={result['rps']:.5f} "
                  f"recall_D={result['recall_D']:.3f} "
                  f"f1_D={result['f1_D']:.3f} "
                  f"ece={result['ece_before']:.4f}->{result['ece_after']:.4f}")

    results = pd.DataFrame(rows)
    order = ["task", "arm", "model", "n_train", "calibration",
             "rps", "rps_before", "log_loss", "brier",
             "ece_before", "ece_after", *PER_CLASS_COLUMNS]
    results = results.reindex(columns=[c for c in order if c in results.columns])

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    output_path = RESULTS_DIR / "resampling_study.csv"
    results.to_csv(output_path, index=False, encoding="utf-8")

    summary = (results.groupby("arm")[["rps", "recall_D", "f1_D", "ece_after"]]
               .mean().round(5).sort_values("rps"))
    print("\nMean over models, per arm (RPS lower is better):")
    print(summary.to_string())
    print(f"\nWrote {len(results)} rows -> {output_path}")


if __name__ == "__main__":
    main()
