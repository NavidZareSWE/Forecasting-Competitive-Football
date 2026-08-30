"""Used by run_pipeline.py. Can also be run directly on an arbitrary command:

    python capture_console.py "python src/models/tuning.py"

Everything this module prints or writes is plain ASCII. Team names in these
data sources carry accents (Malaga, Atletico, Koln), and on Windows the
console defaults to cp1252, where printing one of them raises
UnicodeEncodeError: 'charmap' codec can't encode character. Rather than
rely on the console, every child process is given an ASCII stdout and every
log file is written as ASCII, with anything outside the range escaped as
\\xNN. The escape is reversible, so nothing is lost.
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
LIVE_STEP_LOG_NAME = "_running.txt"
ARCHIVE_NAME = "console-outputs.zip"

DEFAULT_TIMEOUT_SECONDS = 6 * 60 * 60

# Applied to every child process. "backslashreplace" escapes rather than
# drops, so an accented team name survives the log as M\\xe1laga.
ASCII_STDIO = "ascii:backslashreplace"
LOG_ENCODING = "ascii"
LOG_ERRORS = "backslashreplace"


def force_ascii_console():
    """Make this process' own stdout and stderr ASCII-only.

    Call once at start-up. Children are covered separately, through
    PYTHONIOENCODING in run_step.
    """
    for stream in (sys.stdout, sys.stderr):
        if hasattr(stream, "reconfigure"):
            try:
                stream.reconfigure(encoding=LOG_ENCODING,
                                   errors=LOG_ERRORS)
            except (ValueError, OSError):
                pass


PREVIOUS_DIR = PROJECT / "console-outputs-previous"


def prepare_output_dir(fresh=True):
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
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    log_path = _log_path(name)
    started_at = datetime.now()
    started = time.perf_counter()

    child_env = dict(os.environ)
    if env:
        child_env.update({key: str(value) for key, value in env.items()})
    child_env.setdefault("PYTHONIOENCODING", ASCII_STDIO)

    shown = " ".join(f"{key}={value}" for key, value in sorted(env.items())) \
        if env else ""
    header = (f"$ {shown + ' ' if shown else ''}{command}\n"
              f"# started {started_at:%Y-%m-%d %H:%M:%S}\n"
              f"{'-' * 70}\n")
    if echo:
        print(header, end="", flush=True)

    lines = [header]
    timed_out = False
    live_log_path = OUTPUT_DIR / LIVE_STEP_LOG_NAME
    try:
        live_log = live_log_path.open("w", encoding=LOG_ENCODING,
                                      errors=LOG_ERRORS)
    except OSError:
        live_log = None
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
            if live_log is not None:
                live_log.write(line)
                live_log.flush()
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

    if live_log is not None:
        live_log.close()
    body = "".join(lines)
    warning_count = sum(1 for line in lines if "Warning:" in line)
    log_path.write_text(body, encoding=LOG_ENCODING, errors=LOG_ERRORS)
    if live_log_path.exists():
        try:
            live_log_path.unlink()
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
    (OUTPUT_DIR / SUMMARY_NAME).write_text(text, encoding=LOG_ENCODING,
                                           errors=LOG_ERRORS)
    return text


def collect_result_files(patterns):
    collected = OUTPUT_DIR / "results"
    collected.mkdir(parents=True, exist_ok=True)
    copied = 0
    for pattern in patterns:
        for path in sorted(PROJECT.glob(pattern)):
            if not path.is_file():
                continue
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
    force_ascii_console()
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
