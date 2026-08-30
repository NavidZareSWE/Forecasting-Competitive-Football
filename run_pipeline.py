"""Run from the repository root with:

    python run_pipeline.py

Useful flags:
    --only STAGE      run one stage: tests, data, features, tuning, models,
                      experiments, viz, service, report
    --from STAGE      start at a stage and run everything after it
    --skip-tuning     reuse an existing best_params.json
    --skip-download   trust the cached extended raw data in src/data/
    --tasks Lr        restrict the model sweep, e.g. to resume a killed run
    --seeds N         seed repetitions in significance.py (default 3)
    --continue        keep going after a failing step instead of stopping
    --list            print the plan and exit without running anything

Every executable script in the repository is a step below, in dependency
order, with two deliberate exceptions listed under NOT_STEPS at the bottom of
this module.
"""

from datetime import datetime
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path
import argparse
import os
import shutil
import subprocess
import sys

sys.path.insert(0, str(Path(__file__).resolve().parent))

from capture_console import (ARCHIVE_NAME, LOG_ENCODING, LOG_ERRORS,
                             OUTPUT_DIR, collect_result_files,
                             force_ascii_console, make_archive,
                             prepare_output_dir, run_step, write_summary)

# Every step, every log and every line this runner prints is plain ASCII.
# See the note at the top of capture_console.py for why.
force_ascii_console()

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
ESD_PATH = SRC / "data" / "european_soccer_db" / "database.sqlite"

STAGE_ORDER = ["tests", "data", "features", "tuning", "models",
               "experiments", "viz", "service", "report"]

SUPPRESSED_SKLEARN_PARALLEL_WARNING = ("ignore:`sklearn.utils.parallel.delayed` should be used "
                  "with:UserWarning")


def base_environment():
    existing = os.environ.get("PYTHONWARNINGS", "").strip()
    if not existing:
        return {"PYTHONWARNINGS": SUPPRESSED_SKLEARN_PARALLEL_WARNING}
    if SUPPRESSED_SKLEARN_PARALLEL_WARNING in existing:
        return {}
    return {"PYTHONWARNINGS":
            f"{SUPPRESSED_SKLEARN_PARALLEL_WARNING},{existing}"}


def resolve_folder(path):
    path = Path(path)
    if path.exists():
        return path
    parent = path.parent
    if parent.is_dir():
        for candidate in parent.iterdir():
            if candidate.is_dir() and candidate.name.lower() == path.name.lower():
                return candidate
    return path


RESULT_PATTERNS = [
    "src/reports/*.csv",
    "src/reports/*.json",
    "src/reports/models/manifest.json",
    "src/reports/processed/data_quality_log.csv",
    "src/reports/processed/cleaning_drops.csv",
]

# --- tests -----------------------------------------------------------------
# Every one of these builds its own fixtures in memory, so the whole block runs
# before any data exists and a failure stops the run.
TEST_STEPS = [
    ("01_test_event_aggregates", "src/features/test_event_aggregates.py"),
    ("02_test_prematch_features", "src/features/test_prematch_features.py"),
    ("03_test_extended_prematch_features",
     "src/features/test_extended_prematch_features.py"),
    ("04_test_inplay_cut", "src/features/test_inplay_cut.py"),
    ("05_test_team_ratings", "src/pipeline/test_team_ratings.py"),
    ("06_test_stacking", "src/models/test_stacking.py"),
    ("07_test_frozen_reference", "src/models/test_frozen_reference.py"),
    ("08_test_g_smotenc", "src/papers/test_g_smotenc.py"),
    ("09_test_hierarchical_shrinkage",
     "src/papers/test_hierarchical_shrinkage.py"),
]

# --- data ------------------------------------------------------------------
# The six store builders were previously hidden inside
# build_relational_store.py and shared one log. They are expanded here so
# each writes its own log and --from can resume at any of them.
DATA_STEPS = [
    ("10_build_match_store", "src/pipeline/build_match_store.py"),
    ("11_build_label_store", "src/pipeline/build_label_store.py"),
    ("12_build_lineup_store", "src/pipeline/build_lineup_store.py"),
    ("13_build_event_store", "src/pipeline/build_event_store.py"),
    ("14_clean_store", "src/pipeline/clean_store.py"),
    ("15_build_temporal_splits", "src/pipeline/build_temporal_splits.py"),
    ("16_build_market_baseline", "src/pipeline/build_market_baseline.py"),
    ("17_competition_audit", "src/audit/competition_audit.py"),
]

# The extended layer: 2008-2025, eleven leagues, ratings and FIFA squads.
# 18 is the only step that reaches the network, so it is separately skippable.
DOWNLOAD_STEP = ("18_download_extended_data",
                 "src/pipeline/download_extended_data.py")

EXTENDED_DATA_STEPS = [
    ("19_build_team_registry", "src/pipeline/build_team_registry.py"),
    ("20_build_extended_match_store",
     "src/pipeline/build_extended_match_store.py"),
    ("21_build_extended_splits", "src/pipeline/build_extended_splits.py"),
    ("22_build_extended_market_baseline",
     "src/pipeline/build_extended_market_baseline.py"),
    ("23_tune_ratings", "src/pipeline/tune_ratings.py"),
    ("24_build_team_ratings", "src/pipeline/build_team_ratings.py"),
    ("25_build_player_ratings", "src/pipeline/build_player_ratings.py"),
]

# --- features --------------------------------------------------------------
FEATURE_STEPS = [
    ("30_build_team_match_aggregates",
     "src/features/build_team_match_aggregates.py"),
    ("31_build_stat_form_features",
     "src/features/build_stat_form_features.py"),
    ("32_build_rating_features", "src/features/build_rating_features.py"),
    ("33_build_extended_prematch_features",
     "src/features/build_extended_prematch_features.py"),
    ("34_build_prematch_features", "src/features/build_prematch_features.py"),
    ("35_build_inplay_features", "src/features/build_inplay_features.py"),
]

# --- models ----------------------------------------------------------------
# train_ensemble merges its rows into model_results.csv and train_final reads
# that file to choose what to serialise, so the order here is load-bearing.
MODEL_STEPS = [
    ("41_run_models", "src/models/run_models.py"),
    ("42_train_ensemble", "src/models/train_ensemble.py"),
    ("43_train_final", "src/models/train_final.py"),
    ("44_train_market_blend", "src/models/train_market_blend.py"),
    ("45_resampling_study", "src/models/resampling_study.py"),
]

EXPERIMENT_STEPS = [
    ("50_market_comparison", "src/models/market_comparison.py"),
    ("51_inplay_curves", "src/models/inplay_curves.py"),
    ("52_kernel_scaling", "src/models/kernel_scaling.py"),
    ("53_compute_profile", "src/models/compute_profile.py"),
    ("54_margin_to_probability", "src/models/margin_to_probability.py"),
    ("55_significance", "src/models/significance.py"),
    ("56_ablation", "src/models/ablation.py"),
    ("57_shap_analysis", "src/analysis/shap_analysis.py"),
]

VIZ_STEPS = [
    ("60_plot_calibration", "src/viz/plot_calibration.py"),
    ("61_visualize_store", "src/viz/visualize_store.py"),
    ("62_visualize_market_baseline", "src/viz/visualize_market_baseline.py"),
    ("63_viz_raw_matches", "src/viz/viz_raw_matches.py"),
]

# --- service ---------------------------------------------------------------
# measure_latency drives the FastAPI app in process through TestClient, so it
# needs no server. It writes api_latency.csv, which report section 9 reads, and
# it loads the joblib bundles written by 43_train_final.
SERVICE_STEPS = [
    ("70_measure_latency", "src/service/measure_latency.py"),
]

REPORT_STEPS = [
    ("80_build_report", "src/report/build_report.py"),
]

# pack_report copies the finished artefacts out of the gitignored src/reports/
# tree and stamps each with the stage that produced it, so it runs last.
PACK_STEP = ("82_pack_report", "src/audit/pack_report.py")

# Executable scripts deliberately left out of the plan, so that "why is this
# missing" has an answer in the file rather than in someone's memory:
#
#   src/pipeline/run_all.py          the other orchestrator; running it from
#                                    here would nest the whole pipeline
#   src/pipeline/build_relational_store.py
#                                    expanded into steps 10-15 above
#   src/models/train_prematch.py     thin wrappers that re-run two of the four
#   src/models/train_inplay.py       tasks 41_run_models already covers, into
#                                    separate CSVs nothing downstream reads
#   src/papers/g_smotenc.py          its __main__ is a self-check demo; the
#                                    real assertions are in 08_test_g_smotenc
#   src/service/app.py               a long-running server, not a batch step
#   src/service/replay_driver.py     needs app.py listening on a port
#   capture_console.py               this runner's own logging helper
#   check_progress.py                read-only, run in a second terminal
NOT_STEPS = None


def script(path):
    return f'"{PY}" -u {path}'


def preflight(stages):
    problems, warnings = [], []
    required_modules = ["pandas", "numpy", "sklearn", "scipy", "matplotlib",
                        "shap", "imblearn", "plotly", "joblib", "reportlab"]
    if "service" in stages:
        required_modules += ["fastapi", "httpx"]
    for module in required_modules:
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
    downstream = {"features", "tuning", "models", "experiments", "viz",
                  "service", "report"}
    needs_store = (downstream & set(stages)) and "data" not in stages
    if needs_store:
        required = {"clean_events.csv": "the event store",
                    "match_store.csv": "the match store",
                    "temporal_match_splits.csv": "the chronological splits"}
        if "features" in stages:
            required.update({
                "extended_match_store.csv": "the extended match store",
                "temporal_match_splits_extended.csv":
                    "the extended chronological splits",
                "team_ratings.csv": "the Elo and pi ratings",
                "player_ratings.csv": "the player rating snapshots"})
        for filename, description in required.items():
            if not (PROCESSED / filename).exists():
                problems.append(
                    f"Missing {_relative(PROCESSED / filename)} ({description}).\n"
                    f"    Either run the data stage, or put the file there.")
    return problems, warnings


def record_environment():
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
        "\n".join(lines) + "\n", encoding=LOG_ENCODING,
        errors=LOG_ERRORS)

    frozen = subprocess.run([PY, "-m", "pip", "freeze"],
                            capture_output=True, text=True,
                            encoding=LOG_ENCODING, errors=LOG_ERRORS,
                            timeout=180)
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
            header + frozen.stdout, encoding=LOG_ENCODING,
            errors=LOG_ERRORS)
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
    plan = []

    if "tests" in stages:
        for name, path in TEST_STEPS:
            plan.append(("tests", name, script(path), {}))

    if "data" in stages:
        for name, path in DATA_STEPS:
            plan.append(("data", name, script(path), {}))

        name, path = DOWNLOAD_STEP
        if arguments.skip_download:
            print(f"Skipping {name}; trusting the cached extended raw data "
                  f"in {_relative(SRC / 'data')}")
        else:
            plan.append(("data", name, script(path), {}))

        for name, path in EXTENDED_DATA_STEPS:
            plan.append(("data", name, script(path), {}))

    if "features" in stages:
        for name, path in FEATURE_STEPS:
            plan.append(("features", name, script(path), {}))

    if "tuning" in stages:
        if arguments.skip_tuning and BEST_PARAMS.exists():
            print(f"Skipping tuning; reusing {BEST_PARAMS.name}")
        else:
            plan.append(("tuning", "40_tuning",
                         script("src/models/tuning.py"), {}))

    if "models" in stages:
        for name, path in MODEL_STEPS:
            step_env = {}
            if name.endswith("run_models") and arguments.tasks:
                step_env["TASKS"] = arguments.tasks
            plan.append(("models", name, script(path), step_env))

    if "experiments" in stages:
        for name, path in EXPERIMENT_STEPS:
            step_env = {}
            if name.endswith("significance"):
                step_env["N_SEEDS"] = arguments.seeds
            plan.append(("experiments", name, script(path), step_env))

    if "viz" in stages:
        for name, path in VIZ_STEPS:
            plan.append(("viz", name, script(path), {}))

    if "service" in stages:
        for name, path in SERVICE_STEPS:
            plan.append(("service", name, script(path), {}))

    if "report" in stages:
        for name, path in REPORT_STEPS:
            plan.append(("report", name, script(path), {}))
        if shutil.which("node"):
            plan.append(("report", "81_render_docx",
                         "node src/report/render_docx.js", {}))
        else:
            print("node not found; skipping the DOCX build. "
                  "The PDF is unaffected.")
        name, path = PACK_STEP
        plan.append(("report", name, script(path), {}))

    shared = base_environment()
    return [(stage, name, command, {**shared, **step_env})
            for stage, name, command, step_env in plan]


def main():
    parser = argparse.ArgumentParser(
        description="Run the pipeline and capture all console output.")
    parser.add_argument("--only", choices=STAGE_ORDER,
                        help="run a single stage")
    parser.add_argument("--from", dest="start_from", choices=STAGE_ORDER,
                        help="start at this stage and run everything after it")
    parser.add_argument("--skip-tuning", action="store_true",
                        help="reuse an existing best_params.json")
    parser.add_argument("--skip-download", action="store_true",
                        help="trust the cached extended raw data in src/data/")
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

    missing = [(name, command) for _, name, command, _ in plan
               if command.startswith('"') and not
               (PROJECT / command.split(" -u ", 1)[1]).exists()]
    if missing:
        print("Cannot start; the plan names scripts that are not here:\n")
        for name, command in missing:
            print(f"  - {name}: {command.split(' -u ', 1)[1]}")
        return 2

    if arguments.list:
        print(f"Stages: {', '.join(stages)}\n")
        for position, (stage, name, command, step_env) in enumerate(plan,
                                                                    start=1):
            shown = " ".join(f"{key}={value}"
                             for key, value in sorted(step_env.items())
                             if key != "PYTHONWARNINGS")
            print(f"{position:>3}. [{stage:<11}] {name:<36} "
                  f"{shown + ' ' if shown else ''}{command}")
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
    for position, (stage, name, command, step_env) in enumerate(plan, start=1):
        print(f"\n{'=' * 70}\n[{position}/{len(plan)}] {stage}: {name}\n"
              f"{'=' * 70}")
        result = run_step(name, command, env=step_env)
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
    notes.append(f"Copied {copied} result files into console-outputs/results/.")
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
