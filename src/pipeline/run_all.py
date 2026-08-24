"""End-to-end run: raw JSON -> relational store -> features -> models -> analysis.

A hard dependency chain; each stage reads the previous stage's CSV. Stages are
subprocesses because src/ has no package structure.

    python src/pipeline/run_all.py                  # everything
    SKIP_TUNING=1 python src/pipeline/run_all.py    # reuse best_params.json
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

# The search is the longest stage and best_params.json is stable across reruns.
SKIP_TUNING = os.environ.get("SKIP_TUNING", "") not in {"", "0"}

# (step name, directory the script lives in, script filename)
STEPS = [
    # --- Phases 1-4: raw JSON to the relational store ---
    ("Relational store (matches, labels, lineups, events, cleaning, splits)",
     HERE, "build_relational_store.py"),
    ("Market baseline (odds tagging + de-vig)",
     HERE, "build_market_baseline.py"),

    # --- Paper reimplementations: tests are part of the run, not optional ---
    ("P1 tests (G-SMOTENC)", PAPER_DIR, "test_g_smotenc.py"),
    ("P2 tests (Hierarchical Shrinkage)", PAPER_DIR,
     "test_hierarchical_shrinkage.py"),

    # --- Phase 5: the two feature pipelines ---
    ("Leakage tests (time-t cut)", FEATURE_DIR, "test_inplay_cut.py"),
    ("Pre-match feature tests (hand-computed)", FEATURE_DIR,
     "test_prematch_features.py"),
    ("Pre-match features", FEATURE_DIR, "build_prematch_features.py"),

    # --- Phase 6: modeling ---
    ("Hyperparameter search (equal budget, CV inside train)",
     MODEL_DIR, "tuning.py"),
    ("Model sweep (all tasks, tuned configurations)",
     MODEL_DIR, "run_models.py"),
    ("Imbalance study (six resampling arms, Task C)",
     MODEL_DIR, "resampling_study.py"),

    # --- Phase 7: the graded analyses that read the sweep's predictions ---
    ("Market comparison (Task C vs de-vigged odds, tagged test matches)",
     MODEL_DIR, "market_comparison.py"),
    ("In-play curves (metric vs minute, frozen reference, per-phase ECE)",
     MODEL_DIR, "inplay_curves.py"),
    ("Kernel scaling (O(n^2) demonstrated)", MODEL_DIR, "kernel_scaling.py"),
    ("Ablation (feature groups + snapshot frequency, validation only)",
     MODEL_DIR, "ablation.py"),

    # --- Visualisations ---
    ("Reliability diagrams", VIZ_DIR, "plot_calibration.py"),
    ("Visualize relational store", VIZ_DIR, "visualize_store.py"),
    ("Visualize market baseline", VIZ_DIR, "visualize_market_baseline.py"),
    ("Visualize raw match outcomes", VIZ_DIR, "viz_raw_matches.py"),
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
        run_step(name, directory, script)
    print(
        f"\nFull pipeline complete in {time.time() - pipeline_started:.1f}s.")
    print("Relational store  -> src/reports/processed/")
    print("Feature tables    -> src/reports/features/")
    print("Model results     -> src/reports/*.csv, best_params.json")
    print("Visualizations    -> src/reports/visualizations/")
