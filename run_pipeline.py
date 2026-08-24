"""Run the whole remaining pipeline and capture every console output.

    python run_pipeline.py

Everything printed by every step is written to console-outputs/, one log per
step, plus a summary and a zip you can share.

By default this runs EVERYTHING from the raw data forward. Stages, in order:

  1. tests        Paper and leakage tests on fixtures. No data needed. If these
                  fail, nothing after them is trustworthy, so the run stops.
  2. data         StatsBomb JSON -> match, label, lineup and event stores;
                  cleaning; chronological splits; odds tagging and de-vig;
                  competition audit. Downloads ~1500 match files the first
                  time and caches them under data/statsbomb_open_data/.
  3. features     Per-team event aggregates, then the pre-match and in-play
                  feature tables.
  4. tuning       Hyperparameter search. Writes best_params.json.
  5. models       Model sweep + the six-arm imbalance study.
  6. experiments  Market, curves, scaling, compute, conversion, significance,
                  ablation, SHAP.
  7. viz          Reliability diagrams and store visualisations.
  8. report       PDF, then DOCX if node is available.

Useful flags:
    --only STAGE      run one stage: tests, data, features, tuning, models,
                      experiments, viz, report
    --from STAGE      start at a stage and run everything after it
    --skip-tuning     reuse an existing best_params.json
    --tasks Lr        restrict the model sweep, e.g. to resume a killed run
    --seeds N         seed repetitions in significance.py (default 3)
    --continue        keep going after a failing step instead of stopping
    --list            print the plan and exit without running anything

Nothing here touches the test split except the final evaluation inside
run_models.py, which is the one place that is allowed to.
"""

from datetime import datetime
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path
import argparse
import os
import shutil
import subprocess
import sys

from capture_console import (ARCHIVE_NAME, OUTPUT_DIR, collect_result_files,
                             make_archive, prepare_output_dir, run_step,
                             write_summary)

PROJECT = Path(__file__).resolve().parent
PY = sys.executable

SRC = PROJECT / "src"
PROCESSED = SRC / "reports" / "processed"
EVENTS = PROCESSED / "clean_events.csv"
BEST_PARAMS = SRC / "reports" / "best_params.json"
ODDS_DIR = Path(os.environ.get("FOOTBALL_DATA",
                               SRC / "data" / "Football_Data"))
STATSBOMB_DIR = Path(os.environ.get("STATSBOMB_LOCAL",
                                    SRC / "data" / "statsbomb_open_data"
                                    / "data"))

STAGE_ORDER = ["tests", "data", "features", "tuning", "models",
               "experiments", "viz", "report"]


def resolve_folder(path):
    """Return `path`, or a sibling whose name matches case-insensitively.

    Windows filesystems are case-insensitive, so a folder named FOOTBALL_DATA
    satisfies a lookup for Football_Data there but not on Linux or macOS. This
    keeps the same layout working on every platform.
    """
    path = Path(path)
    if path.exists():
        return path
    parent = path.parent
    if parent.is_dir():
        for candidate in parent.iterdir():
            if candidate.is_dir() and candidate.name.lower() == path.name.lower():
                return candidate
    return path


# Files worth shipping back with the logs. Kept small on purpose.
RESULT_PATTERNS = [
    "src/reports/*.csv",
    "src/reports/best_params.json",
]

TEST_STEPS = [
    ("02_test_prematch_features", "src/features/test_prematch_features.py"),
    ("03_test_inplay_cut", "src/features/test_inplay_cut.py"),
    ("04_test_g_smotenc", "src/papers/test_g_smotenc.py"),
    ("05_test_hierarchical_shrinkage",
     "src/papers/test_hierarchical_shrinkage.py"),
]

DATA_STEPS = [
    ("06_build_relational_store", "src/pipeline/build_relational_store.py"),
    ("07_build_market_baseline", "src/pipeline/build_market_baseline.py"),
    ("08_competition_audit", "src/audit/competition_audit.py"),
]

FEATURE_STEPS = [
    ("11_build_prematch_features", "src/features/build_prematch_features.py"),
    ("12_build_inplay_features", "src/features/build_inplay_features.py"),
]

MODEL_STEPS = [
    ("21_run_models", "src/models/run_models.py"),
    ("22_resampling_study", "src/models/resampling_study.py"),
]

VIZ_STEPS = [
    ("50_plot_calibration", "src/viz/plot_calibration.py"),
    ("51_visualize_store", "src/viz/visualize_store.py"),
    ("52_visualize_market_baseline", "src/viz/visualize_market_baseline.py"),
    ("53_viz_raw_matches", "src/viz/viz_raw_matches.py"),
]

EXPERIMENT_STEPS = [
    ("30_market_comparison", "src/models/market_comparison.py"),
    ("31_inplay_curves", "src/models/inplay_curves.py"),
    ("32_kernel_scaling", "src/models/kernel_scaling.py"),
    ("33_compute_profile", "src/models/compute_profile.py"),
    ("34_margin_to_probability", "src/models/margin_to_probability.py"),
    ("35_significance", "src/models/significance.py"),
    ("36_ablation", "src/models/ablation.py"),
    ("37_shap_analysis", "src/analysis/shap_analysis.py"),
]


def script(path):
    return f'"{PY}" -u {path}'


def preflight(stages):
    """Fail early and clearly rather than three stages in."""
    problems, warnings = [], []
    for module in ["pandas", "numpy", "sklearn", "scipy", "matplotlib",
                   "shap", "imblearn", "reportlab"]:
        try:
            __import__(module)
        except ImportError:
            problems.append(f"Missing Python package: {module}. "
                            f"Run: pip install -r requirements.txt")

    if "data" in stages:
        odds = resolve_folder(ODDS_DIR)
        if not odds.exists() or not any(odds.glob("*.csv")):
            problems.append(
                f"No Football-Data odds CSVs found in "
                f"{_relative(ODDS_DIR)}.\n"
                f"    Note: paths are rooted at src/, not the repository "
                f"root.\n"
                f"    Put the Season_20152016_*.csv files there, or set "
                f"FOOTBALL_DATA to the folder that holds them.")
        events_dir = STATSBOMB_DIR / "events"
        cached = (len(list(events_dir.glob("*.json")))
                  if events_dir.is_dir() else 0)
        if cached:
            warnings.append(
                f"Found {cached} cached StatsBomb event files in "
                f"{_relative(events_dir)}; the data stage will reuse them "
                f"instead of downloading.")
        else:
            warnings.append(
                f"No cached StatsBomb events in {_relative(STATSBOMB_DIR)}. "
                f"The data stage will download about 1500 match files on this "
                f"first run and cache them there for later runs.")
    needs_store = ({"features", "tuning", "models", "experiments", "viz",
                    "report"} & set(stages)) and "data" not in stages
    if needs_store:
        required = {"clean_events.csv": "the event store",
                    "match_store.csv": "the match store",
                    "temporal_match_splits.csv": "the chronological splits"}
        for filename, description in required.items():
            if not (PROCESSED / filename).exists():
                problems.append(
                    f"Missing {_relative(PROCESSED / filename)} ({description}).\n"
                    f"    Either run the data stage, or put the file there.")
    return problems, warnings


def record_environment():
    """Write the exact resolved versions, so unpinned installs stay traceable."""
    import platform

    lines = [f"python  {platform.python_version()} ({platform.python_implementation()})",
             f"system  {platform.system()} {platform.release()} "
             f"({platform.machine()})",
             f"run at  {datetime.now():%Y-%m-%d %H:%M:%S}",
             ""]
    packages = ["numpy", "pandas", "scipy", "scikit-learn", "xgboost",
                "lightgbm", "imbalanced-learn", "shap", "matplotlib",
                "pillow", "plotly", "psutil", "reportlab", "fastapi",
                "uvicorn", "httpx"]
    lines.append("declared dependencies, as resolved:")
    for package in packages:
        try:
            lines.append(f"  {package:<20} {version(package)}")
        except PackageNotFoundError:
            lines.append(f"  {package:<20} NOT INSTALLED")

    node = shutil.which("node")
    lines.append("")
    lines.append(f"node    {'not found' if not node else node}")

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    (OUTPUT_DIR / "00_environment.txt").write_text(
        "\n".join(lines) + "\n", encoding="utf-8")

    frozen = subprocess.run([PY, "-m", "pip", "freeze"],
                            capture_output=True, text=True, timeout=180)
    if frozen.returncode == 0:
        header = (f"# Exact versions resolved on "
                  f"{datetime.now():%Y-%m-%d %H:%M:%S}, Python "
                  f"{platform.python_version()}.\n"
                  f"# requirements.txt carries lower bounds and is what you "
                  f"should normally install.\n"
                  f"# Use this file only to reproduce a specific set of "
                  f"results exactly:\n"
                  f"#     pip install -r requirements.lock.txt\n")
        (PROJECT / "requirements.lock.txt").write_text(
            header + frozen.stdout, encoding="utf-8")
    return "\n".join(lines)


def _relative(path):
    try:
        return path.relative_to(PROJECT)
    except ValueError:
        return path


def selected_stages(arguments):
    if arguments.only:
        return [arguments.only]
    stages = list(STAGE_ORDER)
    if arguments.start_from:
        stages = stages[STAGE_ORDER.index(arguments.start_from):]
    return stages


def build_plan(arguments, stages):
    """(stage, name, command) for every step this run should execute."""
    plan = []

    if "tests" in stages:
        for name, path in TEST_STEPS:
            plan.append(("tests", name, script(path)))

    if "data" in stages:
        for name, path in DATA_STEPS:
            plan.append(("data", name, script(path)))

    if "features" in stages:
        for name, path in FEATURE_STEPS:
            plan.append(("features", name, script(path)))

    if "tuning" in stages:
        if arguments.skip_tuning and BEST_PARAMS.exists():
            print(f"Skipping tuning; reusing {BEST_PARAMS.name}")
        else:
            plan.append(("tuning", "20_tuning",
                         script("src/models/tuning.py")))

    if "models" in stages:
        for name, path in MODEL_STEPS:
            command = script(path)
            if name.endswith("run_models") and arguments.tasks:
                command = f"TASKS={arguments.tasks} {command}"
            plan.append(("models", name, command))

    if "experiments" in stages:
        for name, path in EXPERIMENT_STEPS:
            command = script(path)
            if name.endswith("significance"):
                command = f"N_SEEDS={arguments.seeds} {command}"
            plan.append(("experiments", name, command))

    if "viz" in stages:
        for name, path in VIZ_STEPS:
            plan.append(("viz", name, script(path)))

    if "report" in stages:
        plan.append(("report", "40_build_report",
                     script("src/report/build_report.py")))
        if shutil.which("node"):
            plan.append(("report", "41_render_docx",
                         "node src/report/render_docx.js"))
        else:
            print("node not found; skipping the DOCX build. "
                  "The PDF is unaffected.")

    return plan


def main():
    parser = argparse.ArgumentParser(
        description="Run the pipeline and capture all console output.")
    parser.add_argument("--only", choices=STAGE_ORDER,
                        help="run a single stage")
    parser.add_argument("--from", dest="start_from", choices=STAGE_ORDER,
                        help="start at this stage and run everything after it")
    parser.add_argument("--skip-tuning", action="store_true",
                        help="reuse an existing best_params.json")
    parser.add_argument("--list", action="store_true",
                        help="print the plan and exit without running")
    parser.add_argument("--tasks", default="",
                        help="restrict run_models.py, e.g. --tasks Lr")
    parser.add_argument("--seeds", type=int, default=3,
                        help="seed repetitions in significance.py")
    parser.add_argument("--continue", dest="keep_going", action="store_true",
                        help="keep going after a failing step")
    arguments = parser.parse_args()

    if arguments.only and arguments.start_from:
        parser.error("use --only or --from, not both")

    stages = selected_stages(arguments)
    plan = build_plan(arguments, stages)
    if not plan:
        print("Nothing to run.")
        return 0

    if arguments.list:
        print(f"Stages: {', '.join(stages)}\n")
        for position, (stage, name, command) in enumerate(plan, start=1):
            print(f"{position:>3}. [{stage:<11}] {name:<32} {command}")
        return 0

    problems, warnings = preflight(stages)
    if problems:
        print("Cannot start:\n")
        for problem in problems:
            print(f"  - {problem}")
        return 2
    for warning in warnings:
        print(f"Note: {warning}\n")

    prepare_output_dir(fresh=True)
    environment = record_environment()
    print(environment)
    print(f"\nRunning {len(plan)} steps. Logs -> "
          f"{OUTPUT_DIR.relative_to(PROJECT)}/\n")

    results, notes = [], []
    for position, (stage, name, command) in enumerate(plan, start=1):
        print(f"\n{'=' * 70}\n[{position}/{len(plan)}] {stage}: {name}\n"
              f"{'=' * 70}")
        result = run_step(name, command)
        result["stage"] = stage
        results.append(result)

        if result["exit_code"] == 0:
            continue

        fatal = stage == "tests" or not arguments.keep_going
        message = (f"Step {name} exited {result['exit_code']}. "
                   f"See console-outputs/{result['log']}.")
        notes.append(message)
        print(f"\n!! {message}")
        if stage == "tests":
            notes.append("Tests failed, so later stages were not run: their "
                         "results would not be trustworthy.")
            print("!! Tests failed; stopping. Send me that log.")
            break
        if fatal:
            notes.append("Stopped at the first failure. Re-run with "
                         "--continue to attempt the remaining steps.")
            print("!! Stopping. Re-run with --continue to push past this.")
            break

    copied = collect_result_files(RESULT_PATTERNS)
    notes.insert(0, f"Stages run: {', '.join(stages)}")
    notes.append(
        f"Copied {copied} result files into console-outputs/results/.")
    if not BEST_PARAMS.exists():
        notes.append("best_params.json was NOT produced, so the models ran on "
                     "library defaults. The report says so on its front page.")

    summary = write_summary(results, notes)
    archive = make_archive()

    print(f"\n{'=' * 70}\n{summary}")
    print(f"Logs    -> {OUTPUT_DIR}")
    print(f"Share   -> {archive}")
    return 0 if all(r["exit_code"] == 0 for r in results) else 1


if __name__ == "__main__":
    raise SystemExit(main())
