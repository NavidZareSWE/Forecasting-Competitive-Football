import subprocess
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent

STEPS = [
    ("Match store", "build_match_store.py"),
    ("Lineup store", "build_lineup_store.py"),
    ("Event store", "build_event_store.py"),
]


def run_step(name: str, script: str) -> None:
    print(f"\n===== {name} ({script}) =====")
    result = subprocess.run([sys.executable, str(HERE / script)])
    if result.returncode != 0:
        raise SystemExit(f"{name} failed (exit {result.returncode})")


if __name__ == "__main__":
    for name, script in STEPS:
        run_step(name, script)
    print("\nRelational store complete: matches + lineups + events (360 absent for these leagues).")
