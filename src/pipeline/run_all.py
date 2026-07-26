import subprocess
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent      # src/pipeline
VIZ_DIR = HERE.parent / "viz"                # src/viz

# (step name, directory the script lives in, script filename)
STEPS = [
    ("Relational store (matches, labels, lineups, events, cleaning, splits)",
     HERE, "build_relational_store.py"),
    ("Market baseline (odds tagging + de-vig)",
     HERE, "build_market_baseline.py"),
    ("Visualize relational store",
     VIZ_DIR, "visualize_store.py"),
    ("Visualize market baseline",
     VIZ_DIR, "visualize_market_baseline.py"),
    ("Visualize raw match outcomes",
     VIZ_DIR, "viz_raw_matches.py"),
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
        run_step(name, directory, script)
    print(
        f"\nFull pipeline complete in {time.time() - pipeline_started:.1f}s.")
    print("Relational store + market baseline written to src/reports/processed/")
    print("Visualizations written to src/reports/visualizations/")
