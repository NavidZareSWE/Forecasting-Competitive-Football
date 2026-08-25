# RUN.md — what to run, in order

## The short version

```bash
pip install -r requirements.txt
python run_pipeline.py
```

`requirements.txt` declares lower bounds, not pins, so you get current
releases and the install works on Python 3.12 through 3.14. The runner records
the exact versions it resolved into `console-outputs/00_environment.txt` and
writes `requirements.lock.txt`, so a specific set of results stays reproducible
with `pip install -r requirements.lock.txt` without anyone being forced onto
old versions.

That runs **everything from the raw data forward** — downloads the StatsBomb
JSON, builds the stores, cleans, splits, tags odds, builds both feature tables,
tunes, trains, runs every experiment, and writes the report. Console output for
every step goes to `console-outputs/`, plus `_summary.txt` and a
`console-outputs.zip` to send back.

### The one thing you must supply

StatsBomb data downloads automatically. The Football-Data odds files do not.
Put them here before running:

```
data/Football_Data/Season_20152016_*.csv
```

Or point `FOOTBALL_DATA` at the folder that holds them. The runner checks this
before starting rather than failing several stages in.

### Stages

`tests`, `data`, `features`, `tuning`, `models`, `experiments`, `viz`, `report`
— run in that order. `tests` uses hand-written fixtures and needs no data, so
it can always run. If it fails the run stops, because results computed on
feature tables the tests reject are not trustworthy.

See the plan without running anything:

```bash
python run_pipeline.py --list
```

### Flags

| Flag | Effect |
|---|---|
| `--only data` | run one stage |
| `--from features` | start at a stage and run everything after it |
| `--skip-tuning` | reuse an existing `best_params.json` |
| `--tasks Lr` | restrict the model sweep, e.g. to resume a killed run |
| `--seeds 5` | seed repetitions in significance.py (default 3) |
| `--continue` | keep going after a failing step |
| `--list` | print the plan and exit |

### If the machine sleeps or the run dies

Tuning now saves after every task and resumes automatically, so a crash costs
at most the task in flight, not the whole search:

```bash
python run_pipeline.py --from tuning     # picks up where it stopped
```

Force a full re-search with `TUNE_RESUME=0`, or search one task with
`TUNE_TASKS=Lr`.

### Environment variables on Windows

Every `VAR=value command` line below is POSIX shell syntax. `cmd.exe` reads
`VAR=value` as the name of the program to run and answers `'VAR' is not
recognized as an internal or external command`. On cmd.exe, set the variable
first:

```bat
set N_SEEDS=3 && python src/models/significance.py
```

`run_pipeline.py` passes these through the environment rather than the command
line, so its own steps work on either platform without this.

### Is it still working?

Long stages produce no output for a while, which looks identical to a hang.
Two ways to tell, without stopping the run:

```bash
python check_progress.py                # one-off check
python check_progress.py --watch 60     # keep checking every 60s
```

Run it in a SECOND terminal; it only reads and cannot disturb the run.
`--watch` prints one compact line per tick, each compared with the tick before
it, so a stalled stage is obvious at a glance.

It lists which milestones exist, how many StatsBomb files have downloaded, then
samples again 30 seconds later and says whether anything grew. Use
`--wait 180` for the slow model-fitting stages.

You can also watch the live output of the step currently running:

```bash
tail -f console-outputs/_running.txt      # macOS / Linux
Get-Content console-outputs\_running.txt -Wait   # PowerShell
```

That file holds the output of the in-flight step and is replaced by the step's
own log when it finishes.

### Roughly how long

The first `data` run downloads about 1500 match files and is network-bound;
they cache under `data/statsbomb_open_data/` so later runs skip it. Building
the in-play features takes about 50 minutes on one core. Tuning is the longest
stage. If something dies, `--from` lets you resume without repeating what
already succeeded.

The rest of this file explains the same steps individually, if you want to run
them by hand.

---

# RUN.md — step by step

Everything up to and including the feature tables is **already built and shipped
in this zip**. What is missing is the model runs, because this machine has one
CPU and they take hours here.

## 0. Setup

```bash
pip install -r requirements.txt
npm install docx            # only needed for the DOCX report
```

Put your `clean_events.csv` at:

```
src/reports/processed/clean_events.csv
```

## 1. Check what is already done

These files ship in the zip. You do **not** need to rebuild them.

| File | Status |
|---|---|
| `src/reports/processed/team_match_aggregates.csv` | built (3034 rows) |
| `src/reports/features/prematch_features.csv` | built, 1517 x 237 (231 model columns) |
| `src/reports/features/inplay_features.csv` | built, 28823 x 158 (153 model columns) |

Only rebuild them if you change the feature code:

```bash
python src/features/build_team_match_aggregates.py   # ~2 min
python src/features/build_prematch_features.py       # ~1 min
python src/features/build_inplay_features.py         # ~50 min on 1 core
```

## 2. Run the tests first (fast, ~1 min total)

```bash
python src/features/test_event_aggregates.py      # expect 12/12
python src/features/test_prematch_features.py     # expect  9/9
python src/features/test_inplay_cut.py            # expect  7/7
python src/papers/test_g_smotenc.py
python src/papers/test_hierarchical_shrinkage.py  # expect  9/9
```

If any of these fail, stop and send me the output — it means the feature
tables and the code disagree.

## 3. THE IMPORTANT ONE: hyperparameter search

This has **never completed**. Until it does, every model runs on library
defaults and the report says so on its front page.

```bash
python src/models/tuning.py
```

Writes `src/reports/best_params.json`. Expect this to be slow — the P2 model
does an inner cross-validation, so one fit costs several forest fits. On a
multi-core box it should be far faster than here.

**Send me:** the console output, and `src/reports/best_params.json`.

## 4. Model sweep

```bash
python src/models/run_models.py
```

Writes `model_results.csv` and `predictions_<task>.csv`.

If it gets killed part way (it was killed here during task Lr, most likely out
of memory), resume with only the tasks that did not finish:

```bash
TASKS=Lr python src/models/run_models.py          # bash
set TASKS=Lr && python src/models/run_models.py   # cmd.exe
```

Rows carried over from an earlier run are flagged
`carried_from_previous_run=True` and a warning is printed. **Before reporting
anything, do one clean full run** so every row comes from the same feature
tables.

**Send me:** `model_results.csv` and the console output.

## 5. Experiments

```bash
python src/models/market_comparison.py
python src/models/inplay_curves.py
python src/models/kernel_scaling.py
python src/models/compute_profile.py
python src/models/margin_to_probability.py
N_SEEDS=3 python src/models/significance.py    # bash; 5 is the default and is slow
set N_SEEDS=3 && python src/models/significance.py   # cmd.exe
python src/models/ablation.py                  # validation-only, no test access
python src/analysis/shap_analysis.py
```

**Send me:** every CSV these write into `src/reports/`, plus the PNGs in
`src/reports/visualizations/shap/`.

## 6. Report

```bash
python src/report/build_report.py    # PDF + report_content.json
node   src/report/render_docx.js     # DOCX from that JSON
```

Outputs `src/reports/final_report.pdf` and `final_report.docx`.

The report reads every number from the CSVs at build time. If a CSV is missing
it prints a marker naming the script that produces it. **Do not delete those
markers** — regenerate the file instead.

## Or just run everything

```bash
python src/pipeline/run_all.py       # SKIP_TUNING=1 to reuse best_params.json
node   src/report/render_docx.js
```

---

# What I most want back

In priority order:

1. `src/reports/best_params.json` + tuning console output
2. `src/reports/model_results.csv`
3. `src/reports/significance_bootstrap.csv` — this decides whether the
   pre-match models can be claimed to beat the baseline
4. `src/reports/ablation.csv` — which of the 16 new feature groups earn a place
5. Any traceback, in full

Zip `src/reports/` and send it; that covers 1-4 in one go.

---

# Where things stood when I handed over

**Pre-match features helped.** Best RPS moved 0.22104 -> 0.21640 against an
unchanged 0.22812 baseline, so the gap over the prior widened by about 65%.
Five of six learners improved; xgboost got worse, which is what an untuned
boosted model tends to do when the feature count jumps tenfold.

**But the verdict did not change.** Task C is still 0 of 21 significant after
Holm correction. dummy vs random_forest: difference 0.0117, CI
[-0.0066, +0.0280]. 344 test matches cannot resolve an effect that small.

**In-play features did not help.** Every in-play model came out slightly worse
with 142 extra columns (random forest 0.15221 -> 0.15329). The current goal
difference already carried the signal, so the additions were estimation burden.
Task Lc remains strongly significant, 11 of 15.

The short version: widening helped where information was scarce and hurt where
it was already abundant. Tuning is the outstanding variable that could still
move all of this.

**One bug worth knowing about.** The first aggregate run reproduced the correct
scoreline for only 1391 of 1517 matches, always by exactly one goal on the
wrong side. StatsBomb records `Own Goal For` on the team that *benefits*, and
the code was also crediting the opponent's. Fixed; now 1517/1517, and the check
stays in the builder permanently.
