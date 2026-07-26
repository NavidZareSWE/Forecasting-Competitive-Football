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
├── config.py
├── docs
│   ├── course-brief
│   │   ├── Final_Project_Machine_Learning.pdf
│   │   └── Predicting a football match — before it starts, and while it happens - Project      Guide.html
│   └── papers
│       └── Geometric_SMOTE_forimbalanced_datasets_with_nominaland_continuous.pdf
├── LICENSE
├── README.md
├── REPORT.md
├── requirements.txt
└── src
    ├── audit
    │   └── competition_audit.py
    ├── data
    │   ├── FOOTBALL_DATA
    │   └── statsbomb_open_data
    │       └── data
    │           ├── competitions.json
    │           ├── events/
    │           ├── lineups/
    │           ├── matches/
    │           └── three-sixty/
    ├── pipeline
    │   ├── build_event_store.py
    │   ├── build_label_store.py
    │   ├── build_lineup_store.py
    │   ├── build_market_baseline.py
    │   ├── build_match_store.py
    │   ├── build_relational_store.py
    │   ├── build_temporal_splits.py
    │   ├── clean_store.py
    │   ├── run_all.py
    │   └── __pycache__
    │       └── build_event_store.cpython-314.pyc
    ├── reports
    │   ├── processed
    │   │   ├── alias_map.csv
    │   │   ├── cleaning_drops.csv
    │   │   ├── clean_events.csv
    │   │   ├── clean_lineups.csv
    │   │   ├── data_quality_log.csv
    │   │   ├── events_index.csv
    │   │   ├── lineups.csv
    │   │   ├── market_baseline.csv
    │   │   ├── match_store.csv
    │   │   ├── model_targets.csv
    │   │   ├── odds_coverage.csv
    │   │   ├── odds_failures.csv
    │   │   ├── player_identity_map.csv
    │   │   ├── snapshot_split_plan.csv
    │   │   ├── team_identity_map.csv
    │   │   └── temporal_match_splits.csv
    │   └── visualizations
    │       ├── market_baseline_overview.html
    │       ├── raw_result_distribution.html
    │       └── store_overview.html
    └── viz
        ├── visualize_market_baseline.py
        ├── visualize_store.py
        └── viz_raw_matches.py
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
