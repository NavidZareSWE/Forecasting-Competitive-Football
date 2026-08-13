"""Is the pipeline still working, or is it stuck?

Run this in a SECOND terminal while run_pipeline.py keeps going. It only reads,
never writes, so it cannot disturb the run.

    python check_progress.py            # sample twice, 30s apart
    python check_progress.py --wait 90  # longer gap for slow stages

It reports what has been produced so far, then samples again after a pause and
tells you whether anything changed. Growth means it is working. No growth for a
few minutes on a stage that should be producing files means it is worth
investigating.
"""

from datetime import datetime
from pathlib import Path
import argparse
import time

PROJECT = Path(__file__).resolve().parent
SRC = PROJECT / "src"
PROCESSED = SRC / "reports" / "processed"
FEATURES = SRC / "reports" / "features"
REPORTS = SRC / "reports"
STATSBOMB = SRC / "data" / "statsbomb_open_data" / "data"
LOGS = PROJECT / "console-outputs"

# What each stage produces, in the order the pipeline produces it. Used to say
# which stage the run has most likely reached.
MILESTONES = [
    ("data: match store", PROCESSED / "match_store.csv"),
    ("data: label store", PROCESSED / "model_targets.csv"),
    ("data: event store", PROCESSED / "events_index.csv"),
    ("data: cleaned events", PROCESSED / "clean_events.csv"),
    ("data: chronological splits", PROCESSED / "temporal_match_splits.csv"),
    ("data: market baseline", PROCESSED / "market_baseline.csv"),
    ("features: team aggregates", PROCESSED / "team_match_aggregates.csv"),
    ("features: pre-match table", FEATURES / "prematch_features.csv"),
    ("features: in-play table", FEATURES / "inplay_features.csv"),
    ("tuning: best parameters", REPORTS / "best_params.json"),
    ("models: results", REPORTS / "model_results.csv"),
    ("experiments: significance", REPORTS / "significance_bootstrap.csv"),
    ("report: PDF", REPORTS / "final_report.pdf"),
]


def human_size(count):
    value = float(count)
    for unit in ["B", "KB", "MB", "GB"]:
        if value < 1024 or unit == "GB":
            return f"{value:,.1f} {unit}" if unit != "B" else f"{int(value)} B"
        value /= 1024
    return f"{value:.1f} GB"


def age(path):
    seconds = time.time() - path.stat().st_mtime
    if seconds < 90:
        return f"{int(seconds)}s ago"
    if seconds < 5400:
        return f"{seconds / 60:.0f} min ago"
    return f"{seconds / 3600:.1f} h ago"


def sample():
    """A cheap snapshot: counts and total bytes of everything that grows."""
    snapshot = {}
    events = STATSBOMB / "events"
    files = list(events.glob("*.json")) if events.is_dir() else []
    snapshot["statsbomb_events"] = (len(files),
                                    sum(f.stat().st_size for f in files))
    for label, folder in [("processed", PROCESSED), ("features", FEATURES),
                          ("logs", LOGS)]:
        found = [f for f in folder.rglob("*") if f.is_file()] \
            if folder.is_dir() else []
        snapshot[label] = (len(found), sum(f.stat().st_size for f in found))
    return snapshot


def newest_file():
    newest, newest_time = None, 0.0
    for folder in [PROCESSED, FEATURES, STATSBOMB, LOGS, REPORTS]:
        if not folder.is_dir():
            continue
        for path in folder.rglob("*"):
            if path.is_file() and path.stat().st_mtime > newest_time:
                newest, newest_time = path, path.stat().st_mtime
    return newest


def report_state():
    print(f"Checked at {datetime.now():%H:%M:%S}\n")

    events = STATSBOMB / "events"
    count = len(list(events.glob("*.json"))) if events.is_dir() else 0
    if count:
        size = sum(f.stat().st_size for f in events.glob("*.json"))
        print(f"StatsBomb event files cached: {count}  ({human_size(size)})")
        print("  A full download is roughly 1500 files and several GB.\n")
    else:
        print("StatsBomb event files cached: none yet\n")

    print("Milestones:")
    reached = None
    for label, path in MILESTONES:
        if path.exists():
            reached = label
            print(f"  [x] {label:<34} {human_size(path.stat().st_size):>10}"
                  f"   {age(path)}")
        else:
            print(f"  [ ] {label}")
    print(f"\nFurthest milestone reached: {reached or 'none yet'}")

    if LOGS.is_dir():
        logs = sorted(LOGS.glob("*.txt"))
        if logs:
            print(f"\nStep logs written so far ({len(logs)}):")
            for path in logs[-6:]:
                print(f"  {path.name:<38} {age(path)}")
            running = LOGS / "_running.txt"
            if running.exists():
                print(f"\nCurrently running step, last output "
                      f"{age(running)}:")
                tail = running.read_text(encoding="utf-8",
                                         errors="replace").splitlines()
                for line in tail[-8:]:
                    print(f"  | {line}")

    newest = newest_file()
    if newest:
        print(f"\nMost recently touched file: {newest.name}  ({age(newest)})")


def current_step():
    """The step in flight, from the log the runner streams as it goes."""
    running = LOGS / "_running.txt"
    if not running.exists():
        return None, None
    lines = [l for l in running.read_text(encoding="utf-8",
                                          errors="replace").splitlines()
             if l.strip()]
    command = next((l[2:] for l in lines if l.startswith("$ ")), "")
    name = command.split("/")[-1].split("\\")[-1] if command else "unknown"
    last = lines[-1] if lines else ""
    return name, last


def watch(interval):
    """One compact line per tick, each compared with the tick before it."""
    print(f"Watching every {interval}s. Ctrl-C to stop.\n")
    print(f"{'time':<10}{'step':<34}{'done':>5}{'sb files':>10}"
          f"{'change since last tick':>26}")
    print("-" * 85)
    previous = None
    try:
        while True:
            snapshot = sample()
            name, last = current_step()
            done = sum(1 for _, path in MILESTONES if path.exists())
            events = snapshot["statsbomb_events"][0]

            if previous is None:
                change = "first sample"
            else:
                deltas = []
                for key in snapshot:
                    added = snapshot[key][0] - previous[key][0]
                    grew = snapshot[key][1] - previous[key][1]
                    if added:
                        deltas.append(f"{key.split('_')[0]} +{added}")
                    elif grew > 0:
                        deltas.append(f"{key.split('_')[0]} +"
                                      f"{human_size(grew)}")
                change = ", ".join(deltas) if deltas else "nothing changed"

            print(f"{datetime.now():%H:%M:%S}  {str(name)[:32]:<34}"
                  f"{done:>4}/{len(MILESTONES)}{events:>10}"
                  f"{change:>26}")
            if last and not last.startswith("$"):
                print(f"          | {last[:110]}")
            previous = snapshot
            time.sleep(interval)
    except KeyboardInterrupt:
        print("\nStopped watching. The pipeline itself is unaffected.")


def main():
    parser = argparse.ArgumentParser(
        description="Check whether the pipeline is still making progress.")
    parser.add_argument("--wait", type=int, default=30,
                        help="seconds between the two samples (default 30)")
    parser.add_argument("--watch", type=int, metavar="SECONDS",
                        help="keep checking every SECONDS until Ctrl-C, one "
                             "compact line per tick")
    arguments = parser.parse_args()

    if arguments.watch:
        report_state()
        print()
        watch(arguments.watch)
        return

    report_state()

    print(f"\nSampling again in {arguments.wait}s to see if anything "
          f"changes...")
    before = sample()
    time.sleep(arguments.wait)
    after = sample()

    print()
    moved = False
    for key in before:
        count_before, bytes_before = before[key]
        count_after, bytes_after = after[key]
        if count_after != count_before or bytes_after != bytes_before:
            moved = True
            print(f"  {key}: +{count_after - count_before} files, "
                  f"+{human_size(max(0, bytes_after - bytes_before))}")
    if moved:
        print("\nStill working. Leave it running.")
    else:
        print("  nothing changed in either sample")
        print("\nNo growth in that window. That is not proof it is stuck:")
        print("  - Downloads write one file at a time; a slow connection can")
        print("    take longer than the sampling window per file.")
        print("  - Model fitting produces nothing until the step finishes. A")
        print("    single fit on the in-play table can run for many minutes.")
        print(f"  Re-run with a longer window, e.g. --wait 180, and check the")
        print(f"  terminal running the pipeline for its most recent output.")


if __name__ == "__main__":
    main()
