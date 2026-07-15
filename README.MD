# Forecasting Competitive Football

A Machine Learning final-project pipeline that turns raw **StatsBomb Open Data** event streams and **Football-Data.co.uk** bookmaker odds into three forecasting deliverables for professional football matches:

| Model | Task | Question it answers |
|---|---|---|
| **Model 1 (C)** | Pre-match classification | P(Home win / Draw / Away win), before kick-off |
| **Model 2 (R)** | Pre-match regression | Expected signed goal margin (home − away), clipped to `[-5, +5]` |
| **Model 3 (L)** | In-play / live prediction | Both of the above, re-estimated at snapshot times `t` during the match |

The project follows the ten-stage pipeline (ingestion → integration → labeling → cleaning → temporal split → dual feature pipelines → modeling → calibration → evaluation → error analysis / SHAP → ablation) defined in the course brief (see [`Extras/`](./Extras)).

---

## 📁 Repository Structure

```
.
├── .gitignore
├── config.py                  # Central paths / constants for the pipeline
├── requirements.txt           # Pinned Python dependencies
├── README.MD
├── Extras/                    # Course-provided reference material
│   ├── Final_Project_Machine_Learning.pdf
│   └── Predicting a football match — before it starts, and while it happens - Project Guide.html
├── papers/                    # P1 (data) and P2 (model) paper reimplementations
└── src/
    ├── data/                          # Raw source data (not modeling-ready)
    │   ├── Football_Data/             # Football-Data.co.uk season CSVs (odds, results, stats)
    │   └── statsbomb_open_data/
    │       └── data/
    │           ├── competitions.json  # (expected) master competition/season list
    │           ├── matches/           # matches/{competition_id}/{season_id}.json
    │           ├── events/            # events/{match_id}.json (~4,000 rows/match)
    │           ├── lineups/           # lineups/{match_id}.json
    │           └── three-sixty/       # three-sixty/{match_id}.json (selected matches only)
    │
    ├── scripts/                       # Pipeline stage 1: ingestion & integration
    │   ├── build_match_store.py       # Parses matches/*.json → one row per match + labels
    │   ├── build_label_store.py       # Validates final labels → canonical model targets
    │   ├── clean_store.py             # Validates, cleans, and audits relational stores
    │   ├── build_temporal_splits.py   # Chronological parent-match and snapshot splits
    │   ├── build_event_store.py       # Indexes/parses events/{match_id}.json
    │   ├── build_lineup_store.py      # Parses lineups/{match_id}.json
    │   ├── build_relational_store.py  # Joins match/lineup/event tables on competition_id,
    │   │                               #   season_id, match_id into the relational store
    │   ├── build_market_baseline.py   # Tags Football-Data.co.uk odds to match_id, de-vigs
    │   │                               #   the 1X2 odds into the market baseline
    │   ├── run_all.py                 # Orchestrates the full ingestion → store pipeline
    │   ├── visualize_store.py         # Generates store_overview.html
    │   ├── visualize_market_baseline.py # Generates market_baseline_overview.html
    │   └── viz_raw_matches.py         # Generates raw_result_distribution.html
    │
    ├── test/
    │   └── competition_audit.py       # Validates/justifies competition & season selection
    │
    ├── features/                      # (planned) Pre-match & in-play feature pipelines
    ├── models/                        # (planned) Model 1 / 2 / 3 training & evaluation
    │
    └── reports/
        ├── competition_audit.csv / .html   # Competition/season selection audit
        ├── processed/                      # Modeling-ready intermediate tables
        │   ├── match_store.csv             # One row per match (labels, teams, venue, date)
        │   ├── model_targets.csv            # Canonical targets for Models 1 / 2 / 3
        │   ├── clean_events.csv / clean_lineups.csv # Cleaned feature-pipeline inputs
        │   ├── team_identity_map.csv / player_identity_map.csv # Canonical StatsBomb IDs
        │   ├── cleaning_drops.csv / data_quality_log.csv # Cleaning evidence
        │   ├── temporal_match_splits.csv # One chronological split per match
        │   ├── snapshot_split_plan.csv    # Five-minute snapshots inheriting parent split
        │   ├── events_index.csv            # Indexed/validated event stream reference
        │   ├── lineups.csv                 # Squad & player-position table
        │   ├── alias_map.csv               # StatsBomb ↔ Football-Data team-name aliases
        │   ├── market_baseline.csv         # De-vigged 1X2 probabilities per tagged match
        │   ├── odds_coverage.csv           # Per-season odds-tagging coverage rate
        │   └── odds_failures.csv           # Matches that failed odds tagging, with reasons
        └── visualizations/
            ├── competition_audit.html
            ├── store_overview.html
            ├── market_baseline_overview.html
            └── raw_result_distribution.html
```

> **Note:** `src/data/` holds large raw source files (StatsBomb JSON tree, Football-Data CSVs) and is expected to be excluded from version control via `.gitignore`. Populate it locally before running the pipeline (see [Setup](#️-setup)).

---

## 🎯 Project Overview

This repository implements the data-integration and market-baseline stages of the course's final project, whose full specification lives in [`Extras/Final_Project_Machine_Learning.pdf`](./Extras/Final_Project_Machine_Learning.pdf) and the companion [Project Guide HTML](<./Extras/Predicting a football match — before it starts, and while it happens - Project Guide.html>).

Grading emphasis, as defined in the brief:

| Component | Weight |
|---|---|
| Data Integration & Feature Pipelines | 30% |
| Paper Reimplementations (P1 + P2) | 20% |
| Three Models & Comparative Analysis | 20% |
| Written Report | 15% |
| Defences (2 mid + final) | 15% |

---

## 📊 Data Sources

1. **[StatsBomb Open Data](https://github.com/statsbomb/open-data)** — free, JSON, event-level football data (~4,000 events/match), keyed by `competition_id`, `season_id`, `match_id`. Free for research / non-commercial use **with attribution**.
2. **[Football-Data.co.uk](https://www.football-data.co.uk)** — free per-season CSVs with match results, statistics, and pre-match bookmaker 1X2 / totals / Asian Handicap odds. Used to build the de-vigged market baseline that every classification model is judged against.

These two sources have **no shared key**; `build_market_baseline.py` tags odds rows to a `match_id` via `(date, home_team, away_team)` and an alias map (`alias_map.csv`), logging coverage (`odds_coverage.csv`) and failures (`odds_failures.csv`) as required by the project's test-set discipline rules.

---

## ⚙️ Setup

```bash
# 1. Create and activate a virtual environment
python -m venv .venv
source .venv/bin/activate      # Windows: .venv\Scripts\activate

# 2. Install dependencies
pip install -r requirements.txt

# 3. Populate src/data/ locally
#    - Clone/download StatsBomb Open Data into src/data/statsbomb_open_data/data/
#    - Download the matching Football-Data.co.uk season CSVs into src/data/Football_Data/

# 4. Run the ingestion, labeling, cleaning, and split pipeline
python src/scripts/build_relational_store.py
```

Paths and shared constants used across the scripts are centralized in [`config.py`](./config.py).

`build_event_store.py` downloads missing StatsBomb event files concurrently (six workers by default). If your connection is stable, increase throughput with `SB_DOWNLOAD_WORKERS=8 python src/scripts/build_event_store.py`; lower it to `1` if GitHub throttles or your connection is unreliable. Cached JSON files are always reused.

---

## 🧱 Pipeline Status

**Stage 1 — Ingestion & Integration (in progress / current focus):**
- ✅ Parse and store matches, events, and lineups (`build_match_store.py`, `build_event_store.py`, `build_lineup_store.py`)
- ✅ Join the file families into one relational store (`build_relational_store.py`)
- ✅ Tag and de-vig bookmaker odds into the market baseline (`build_market_baseline.py`)
- ✅ Competition/season selection audit (`test/competition_audit.py`)
- ✅ Diagnostic visualizations of the raw and joined data (`visualize_store.py`, `visualize_market_baseline.py`, `viz_raw_matches.py`)

**Planned next (per the course pipeline diagram):**
- ✅ Labeling & formal problem definition for Models 1/2/3 (`build_label_store.py`, `REPORT.md`)
- ✅ Cleaning: event ordering, canonical identities, and audit logs (`clean_store.py`)
- ✅ Temporal train/validation/test split and inherited five-minute snapshot plan (`build_temporal_splits.py`)
- ⬜ Pre-match feature pipeline (`src/features/`)
- ⬜ In-play snapshot feature pipeline with the time-`t` leakage cut assertion (`src/features/`)
- ⬜ Model training suite (`src/models/`) — Dummy, SVM/SVR, Kernel Ridge, Random Forest, GBM, XGBoost, LightGBM
- ⬜ P1 (data) and P2 (model) paper reimplementations (`papers/`)
- ⬜ Calibration, evaluation, SHAP-based error analysis, and ablation

---

## 📈 Reports & Outputs

`src/reports/processed/` holds the modeling-ready intermediate tables produced by the ingestion scripts (match store, event index, lineups, alias map, market baseline, odds coverage/failures). `src/reports/visualizations/` holds standalone HTML diagnostics for inspecting the raw data, the joined store, the market baseline, and the raw result distribution. `competition_audit.csv` / `.html` document and justify which competitions and seasons were selected for training, per the project's data-integrity requirements.

---

## ⚠️ Data Leakage Discipline

This project follows the leakage rules defined in the course brief:

- Pre-match features are computed only from matches that finished **before** kick-off.
- In-play snapshot features use only events with `timestamp <= t`.
- All snapshots of a match stay on one side of the train/val/test split.
- Resampling, scalers, and imputers are fit on training folds only.
- Odds are used **either** as the market baseline **or** as a declared feature — never both in the same experiment.
- The odds-to-`match_id` tagging uses pre-match identity only (date, teams) — never scores or statistics.

See `Extras/` for the full checklist.

---

## 📚 References

- StatsBomb / Hudl StatsBomb. *StatsBomb Open Data*. https://github.com/statsbomb/open-data
- Football-Data.co.uk. *Historical Football Results and Betting Odds Data*. https://www.football-data.co.uk
- Full course specification and TA clarifications: [`Extras/`](./Extras)
