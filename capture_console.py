"""Run commands and capture everything they print into console-outputs/.

Each step writes its own log file, stdout and stderr interleaved in the order
they were produced, so a traceback appears where it actually happened. A
summary file records the exit code and duration of every step, and the whole
folder is zipped at the end so it can be shared as one file.

Used by run_pipeline.py. Can also be run directly on an arbitrary command:

    python capture_console.py "python src/models/tuning.py"
"""

from datetime import datetime
from pathlib import Path
import os
import shutil
import subprocess
import sys
import time
import zipfile

PROJECT = Path(__file__).resolve().parent
OUTPUT_DIR = PROJECT / "console-outputs"
SUMMARY_NAME = "_summary.txt"
# Live output of the step currently running, so progress is
# visible from another terminal while a long step is in flight.
RUNNING_NAME = "_running.txt"
ARCHIVE_NAME = "console-outputs.zip"

# Anything longer than this is almost certainly a hung process rather than a
# slow one. Tuning on a wide feature table is genuinely slow, so it is high.
DEFAULT_TIMEOUT_SECONDS = 6 * 60 * 60


PREVIOUS_DIR = PROJECT / "console-outputs-previous"


def prepare_output_dir(fresh=True):
    """Start a clean log folder, keeping the last one.

    A resumed run only re-runs the stages it was asked for, so deleting the
    folder outright would throw away the logs of everything before it. The
    tuning log alone represents four hours of compute and is the only record of
    how best_params.json was chosen, so it is moved aside rather than removed.
    """
    if fresh and OUTPUT_DIR.exists():
        if PREVIOUS_DIR.exists():
            shutil.rmtree(PREVIOUS_DIR)
        shutil.move(str(OUTPUT_DIR), str(PREVIOUS_DIR))
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    return OUTPUT_DIR


def _log_path(name):
    safe = "".join(c if c.isalnum() or c in "-_." else "_" for c in name)
    return OUTPUT_DIR / f"{safe}.txt"


def run_step(name, command, timeout=DEFAULT_TIMEOUT_SECONDS, echo=True,
             env=None):
    """Run one command, tee its output to the console and to a log file.

    `env` is a mapping of extra variables merged over the current environment
    and passed to the child. It exists because `VAR=value command` is POSIX
    shell syntax: cmd.exe reads `VAR=value` as the name of the program to run
    and fails with "is not recognized as an internal or external command".
    Passing the variables through the environment works on every platform, and
    the child processes a step spawns (joblib workers, for instance) inherit
    them too.
    """
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    log_path = _log_path(name)
    started_at = datetime.now()
    started = time.perf_counter()

    child_env = None
    if env:
        child_env = dict(os.environ)
        child_env.update({key: str(value) for key, value in env.items()})

    shown = " ".join(f"{key}={value}" for key, value in sorted(env.items())) \
        if env else ""
    header = (f"$ {shown + ' ' if shown else ''}{command}\n"
              f"# started {started_at:%Y-%m-%d %H:%M:%S}\n"
              f"{'-' * 70}\n")
    if echo:
        print(header, end="", flush=True)

    lines = [header]
    timed_out = False
    # Written as the step runs, not after it: a step that takes an hour used to
    # leave no log at all until it finished, which made a long run
    # indistinguishable from a hung one. This file is truncated and replaced by
    # the step's own log when the step ends.
    running_path = OUTPUT_DIR / RUNNING_NAME
    try:
        running = running_path.open("w", encoding="utf-8", errors="replace")
    except OSError:
        running = None
    try:
        process = subprocess.Popen(
            command, shell=True, cwd=str(PROJECT), env=child_env,
            stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
            text=True, bufsize=1, errors="replace")
        deadline = time.perf_counter() + timeout
        for line in process.stdout:
            lines.append(line)
            if echo:
                print(line, end="", flush=True)
            if running is not None:
                running.write(line)
                running.flush()
            if time.perf_counter() > deadline:
                process.kill()
                timed_out = True
                break
        process.wait()
        exit_code = process.returncode
    except KeyboardInterrupt:
        lines.append("\n# interrupted by the user (Ctrl-C)\n")
        exit_code = 130
    except Exception as error:                       # noqa: BLE001
        lines.append(f"\n# the runner itself failed: {error!r}\n")
        exit_code = -1

    seconds = time.perf_counter() - started
    if timed_out:
        lines.append(f"\n# TIMED OUT after {timeout}s and was killed\n")
        exit_code = -9
    footer = (f"{'-' * 70}\n"
              f"# exit code {exit_code} after {seconds:.1f}s\n")
    lines.append(footer)
    if echo:
        print(footer, end="", flush=True)

    if running is not None:
        running.close()
    body = "".join(lines)
    warning_count = sum(1 for line in lines if "Warning:" in line)
    log_path.write_text(body, encoding="utf-8")
    # The heartbeat belongs to whichever step is running now, so clear it.
    if running_path.exists():
        try:
            running_path.unlink()
        except OSError:
            pass
    return {"name": name, "command": command, "exit_code": exit_code,
            "seconds": round(seconds, 1), "log": log_path.name,
            "warnings": warning_count,
            "started_at": started_at.strftime("%Y-%m-%d %H:%M:%S")}


def write_summary(results, extra_notes=None):
    lines = [f"Console capture summary",
             f"Written {datetime.now():%Y-%m-%d %H:%M:%S}",
             f"Python  {sys.version.split()[0]}",
             f"Project {PROJECT}",
             "",
             f"{'step':<44}{'exit':>6}{'seconds':>10}{'warns':>7}  log",
             "-" * 99]
    for result in results:
        lines.append(f"{result['name']:<44}{result['exit_code']:>6}"
                     f"{result['seconds']:>10.1f}"
                     f"{result.get('warnings', 0):>7}  {result['log']}")
    failed = [r for r in results if r["exit_code"] != 0]
    lines.append("-" * 92)
    lines.append(f"{len(results) - len(failed)}/{len(results)} steps exited 0")
    noisy = [r for r in results if r.get("warnings")]
    if noisy:
        total = sum(r["warnings"] for r in noisy)
        lines.append(f"{total} warning lines across {len(noisy)} steps: "
                     + ", ".join(f"{r['name']} ({r['warnings']})"
                                 for r in noisy))
    else:
        lines.append("No warnings emitted by any step.")
    if failed:
        lines.append("")
        lines.append("FAILED STEPS - open these logs first:")
        for result in failed:
            lines.append(f"  {result['log']}  (exit {result['exit_code']})")
    if extra_notes:
        lines.append("")
        lines.extend(extra_notes)
    text = "\n".join(lines) + "\n"
    (OUTPUT_DIR / SUMMARY_NAME).write_text(text, encoding="utf-8")
    return text


def collect_result_files(patterns):
    """Copy small result files next to the logs so one zip has everything."""
    collected = OUTPUT_DIR / "results"
    collected.mkdir(parents=True, exist_ok=True)
    copied = 0
    for pattern in patterns:
        for path in sorted(PROJECT.glob(pattern)):
            if not path.is_file():
                continue
            # Skip anything large; the point is a shareable archive.
            if path.stat().st_size > 25 * 1024 * 1024:
                continue
            destination = collected / path.name
            try:
                shutil.copy2(path, destination)
                copied += 1
            except OSError:
                pass
    return copied


def make_archive():
    archive_path = PROJECT / ARCHIVE_NAME
    if archive_path.exists():
        archive_path.unlink()
    with zipfile.ZipFile(archive_path, "w", zipfile.ZIP_DEFLATED) as archive:
        for path in sorted(OUTPUT_DIR.rglob("*")):
            if path.is_file():
                archive.write(path, path.relative_to(OUTPUT_DIR.parent))
    return archive_path


def main():
    if len(sys.argv) < 2:
        print(__doc__)
        return 2
    command = " ".join(sys.argv[1:])
    prepare_output_dir(fresh=False)
    result = run_step(command.split()[-1].replace("/", "_"), command)
    print(f"\nLog -> {OUTPUT_DIR / result['log']}")
    return 0 if result["exit_code"] == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
