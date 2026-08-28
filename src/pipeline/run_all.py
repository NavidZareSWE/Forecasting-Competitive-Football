"""Run from the repository root with:

    python src/pipeline/run_all.py                  # everything
    SKIP_TUNING=1 python src/pipeline/run_all.py    # reuse best_params.json
    SKIP_DOWNLOAD=1 python src/pipeline/run_all.py  # trust cached src/data/
"""

import os
import subprocess
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent          # src/pipeline
SRC = HERE.parent                               # src
VIZ_DIR = SRC / "viz"
FEATURE_DIR = SRC / "features"
MODEL_DIR = SRC / "models"
PAPER_DIR = SRC / "papers"
ANALYSIS_DIR = SRC / "analysis"
REPORT_DIR = SRC / "report"

SKIP_TUNING = os.environ.get("SKIP_TUNING", "") not in {"", "0"}
SKIP_DOWNLOAD = os.environ.get("SKIP_DOWNLOAD", "") not in {"", "0"}

STEPS = [
    # --- Phases 1-4: raw JSON to the relational store ---
    ("Relational store (matches, labels, lineups, events, cleaning, splits)",
     HERE, "build_relational_store.py"),
    ("Market baseline (odds tagging + de-vig)",
     HERE, "build_market_baseline.py"),

    # --- Extended data layer: 2008-2025, 11 leagues, ratings, FIFA squads ---
    ("Extended raw data (ESD sqlite, football-data seasons, FIFA ratings)",
     HERE, "download_extended_data.py"),
    ("Team registry (canonical ids + alias maps across sources)",
     HERE, "build_team_registry.py"),
    ("Extended match store (era-partitioned, 52k matches)",
     HERE, "build_extended_match_store.py"),
    ("Extended splits (season-boundary, in-play holdout excluded)",
     HERE, "build_extended_splits.py"),
    ("Extended market baseline (three odds sources, de-vig)",
     HERE, "build_extended_market_baseline.py"),
    ("Team rating tests (Elo + pi hand-computed)",
     HERE, "test_team_ratings.py"),
    ("Team ratings (Elo + pi-ratings, leak-free sequential)",
     HERE, "build_team_ratings.py"),
    ("Player rating snapshots (ESD attributes + sofifa updates)",
     HERE, "build_player_ratings.py"),
    ("FIFA-style XI and squad rating features",
     FEATURE_DIR, "build_rating_features.py"),
    ("Extended pre-match feature tests (hand-computed)",
     FEATURE_DIR, "test_extended_prematch_features.py"),
    ("Extended pre-match features (form + ratings + squads)",
     FEATURE_DIR, "build_extended_prematch_features.py"),

    # --- Paper reimplementations: tests are part of the run, not optional ---
    ("P1 tests (G-SMOTENC)", PAPER_DIR, "test_g_smotenc.py"),
    ("P2 tests (Hierarchical Shrinkage)", PAPER_DIR,
     "test_hierarchical_shrinkage.py"),

    # --- Phase 5: the two feature pipelines ---
    ("Leakage tests (time-t cut)", FEATURE_DIR, "test_inplay_cut.py"),
    ("Pre-match feature tests (hand-computed)", FEATURE_DIR,
     "test_prematch_features.py"),
    ("Pre-match features", FEATURE_DIR, "build_prematch_features.py"),
    ("In-play snapshot features", FEATURE_DIR, "build_inplay_features.py"),

    # --- Phase 6: modeling ---
    ("Hyperparameter search (equal budget, CV inside train)",
     MODEL_DIR, "tuning.py"),
    ("Model sweep (all tasks, tuned configurations)",
     MODEL_DIR, "run_models.py"),
    ("Persist serving models (joblib bundles + manifest)",
     MODEL_DIR, "train_final.py"),
    ("Imbalance study (six resampling arms, Task C)",
     MODEL_DIR, "resampling_study.py"),

    # --- Phase 7: the graded analyses that read the sweep's predictions ---
    ("Market comparison (Task C vs de-vigged odds, tagged test matches)",
     MODEL_DIR, "market_comparison.py"),
    ("In-play curves (metric vs minute, frozen reference, per-phase ECE)",
     MODEL_DIR, "inplay_curves.py"),
    ("Kernel scaling (O(n^2) demonstrated)", MODEL_DIR, "kernel_scaling.py"),
    ("Compute and memory comparison", MODEL_DIR, "compute_profile.py"),
    ("Margin-to-probability conversion (Model 2 -> Model 1)",
     MODEL_DIR, "margin_to_probability.py"),
    ("Statistical testing across repeated experiments",
     MODEL_DIR, "significance.py"),

    # --- Phase 8: error analysis and SHAP ---
    ("SHAP and worst-prediction post-mortem", ANALYSIS_DIR,
     "shap_analysis.py"),

    # --- Phase 9: ablation, scored on validation only ---
    ("Ablation (feature groups, snapshot frequency, balancing, P1)",
     MODEL_DIR, "ablation.py"),

    # --- Visualisations ---
    ("Reliability diagrams", VIZ_DIR, "plot_calibration.py"),
    ("Visualize relational store", VIZ_DIR, "visualize_store.py"),
    ("Visualize market baseline", VIZ_DIR, "visualize_market_baseline.py"),
    ("Visualize raw match outcomes", VIZ_DIR, "viz_raw_matches.py"),

    # --- Phase 10: the report, built from the measured results ---
    ("Final report PDF and content export", REPORT_DIR, "build_report.py"),
]


def run_step(name: str, directory: Path, script: str) -> None:
    path = directory / script
    if not path.exists():
        raise SystemExit(f"{name}: script not found at {path}")
    print(f"\n===== {name} ({script}) =====")
    started = time.time()
    result = subprocess.run([sys.executable, str(path)])
    if result.returncode != 0:
        raise SystemExit(f"{name} failed (exit {result.returncode})")
    print(f"----- {name} done in {time.time() - started:.1f}s -----")


if __name__ == "__main__":
    pipeline_started = time.time()
    for name, directory, script in STEPS:
        if script == "tuning.py" and SKIP_TUNING:
            print(f"\n===== SKIPPED: {name} (SKIP_TUNING set) =====")
            continue
        if script == "download_extended_data.py" and SKIP_DOWNLOAD:
            print(f"\n===== SKIPPED: {name} (SKIP_DOWNLOAD set) =====")
            continue
        run_step(name, directory, script)
    print(f"\nFull pipeline complete in {time.time() - pipeline_started:.1f}s.")
    print("Relational store  -> src/reports/processed/")
    print("Feature tables    -> src/reports/features/")
    print("Model results     -> src/reports/*.csv, best_params.json")
    print("SHAP artefacts    -> src/reports/visualizations/shap/")
    print("Final report      -> src/reports/final_report.pdf "
          "and final_report.docx")
    print("Visualizations    -> src/reports/visualizations/")
