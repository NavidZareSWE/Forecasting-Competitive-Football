# Forecasting Competitive Football

A Machine Learning final-project pipeline that turns raw **StatsBomb Open Data** event streams and **Football-Data.co.uk** bookmaker odds into three forecasting deliverables for professional football matches:

| Model | Task | Question it answers |
|---|---|---|
| **Model 1 (C)** | Pre-match classification | P(Home win / Draw / Away win), before kick-off |
| **Model 2 (R)** | Pre-match regression | Expected signed goal margin (home − away), clipped to `[-5, +5]` |
| **Model 3 (L)** | In-play / live prediction | Both of the above, re-estimated at snapshot minutes `t = 0, 5, …, 90` |

In-play work is split into two internal task codes: `Lc` (snapshot classification) and `Lr` (snapshot regression). All four are run by the same sweep.

The project follows the ten-stage pipeline (ingestion → integration → labeling → cleaning → temporal split → dual feature pipelines → modeling → calibration → evaluation → error analysis / SHAP → ablation) defined in the course brief (see [`docs/course-brief/`](./docs/course-brief)).

---

## 📁 Repository Structure

```
├── .gitignore
├── config.py
├── docs
│   ├── .nojekyll
│   ├── course-brief
│   │   ├── Final_Project_Machine_Learning.pdf
│   │   └── Predicting a football match — before it starts, and while it happens - Project Guide.html
│   ├── index.html
│   ├── mid_defence_2_slides.html
│   └── papers
│       ├── P1_GeometricSMOTE_ImbalancedData.pdf
│       ├── P2_HierarchicalShrinkage_TreeModels.pdf
│       ├── p2_paper_Appendix A — Hand derivation of the P2 model.md
│       └── p2_paper_selection.md
├── LICENSE
├── README.md
├── REPORT.md
├── requirements.txt
├── src
│   ├── data
│   │   ├── FOOTBALL_DATA
│   │   └── statsbomb_open_data
│   │       └── data
│   │           ├── competitions.json
│   │           ├── events/
│   │           ├── lineups/
│   │           ├── matches/
│   │           └── three-sixty/
│   ├── audit
│   │   └── competition_audit.py
│   ├── features
│   │   ├── build_inplay_features.py
│   │   ├── build_prematch_features.py
│   │   ├── test_inplay_cut.py
│   │   └── __pycache__
│   │       └── build_inplay_features.cpython-314.pyc
│   ├── models
│   │   ├── inplay_curves.py
│   │   ├── kernel_scaling.py
│   │   ├── market_comparison.py
│   │   ├── modeling_common.py
│   │   ├── model_zoo.py
│   │   ├── resampling_study.py
│   │   ├── run_models.py
│   │   ├── train.py
│   │   ├── train_inplay.py
│   │   ├── train_prematch.py
│   │   └── tuning.py
│   ├── papers
│   │   ├── g_smotenc.py
│   │   ├── hierarchical_shrinkage.py
│   │   ├── test_g_smotenc.py
│   │   ├── test_hierarchical_shrinkage.py
│   │   └── __pycache__
│   │       └── g_smotenc.cpython-314.pyc
│   ├── pipeline
│   │   ├── build_event_store.py
│   │   ├── build_label_store.py
│   │   ├── build_lineup_store.py
│   │   ├── build_market_baseline.py
│   │   ├── build_match_store.py
│   │   ├── build_relational_store.py
│   │   ├── build_temporal_splits.py
│   │   ├── clean_store.py
│   │   ├── run_all.py
│   │   └── __pycache__
│   │       └── build_event_store.cpython-314.pyc
│   ├── reports
│   │   ├── features
│   │   │   ├── inplay_features.csv
│   │   │   └── prematch_features.csv
│   │   ├── processed
│   │   │   ├── alias_map.csv
│   │   │   ├── cleaning_drops.csv
│   │   │   ├── clean_events.csv
│   │   │   ├── clean_lineups.csv
│   │   │   ├── data_quality_log.csv
│   │   │   ├── events_index.csv
│   │   │   ├── lineups.csv
│   │   │   ├── market_baseline.csv
│   │   │   ├── match_store.csv
│   │   │   ├── model_targets.csv
│   │   │   ├── odds_coverage.csv
│   │   │   ├── odds_failures.csv
│   │   │   ├── player_identity_map.csv
│   │   │   ├── snapshot_split_plan.csv
│   │   │   ├── team_identity_map.csv
│   │   │   └── temporal_match_splits.csv
│   │   └── visualizations
│   │       ├── market_baseline_overview.html
│   │       ├── raw_result_distribution.html
│   │       └── store_overview.html
│   └── viz
│       ├── plot_calibration.py
│       ├── visualize_market_baseline.py
│       ├── visualize_store.py
│       └── viz_raw_matches.py
└── TODO.md
```

> **Note:** `src/data/` (raw sources) and `src/reports/` (generated outputs) are excluded from version control via `.gitignore`. Populate `src/data/` locally and regenerate `src/reports/` by running the pipeline.

---

## 🎯 Project Overview

The full specification lives in [`docs/course-brief/Final_Project_Machine_Learning.pdf`](./docs/course-brief/Final_Project_Machine_Learning.pdf) and the companion Project Guide HTML in the same folder.

Grading emphasis, as defined in the brief:

| Component | Weight |
|---|---|
| Data Integration & Feature Pipelines | 30% |
| Paper Reimplementations (P1 + P2) | 20% |
| Three Models & Comparative Analysis | 20% |
| Written Report | 15% |
| Defences (2 mid + final) | 15% |

Scope: four male domestic leagues in the 2015/16 season — Premier League `(2, 27)`, Ligue 1 `(7, 27)`, La Liga `(11, 27)`, Serie A `(12, 27)`.

---

## 📊 Data Sources

1. **[StatsBomb Open Data](https://github.com/statsbomb/open-data)** — free, JSON, event-level football data, keyed by `competition_id`, `season_id`, `match_id`. Free for research / non-commercial use **with attribution**.
2. **[Football-Data.co.uk](https://www.football-data.co.uk)** — free per-season CSVs with results, statistics, and pre-match bookmaker 1X2 / totals / Asian Handicap odds. Used to build the de-vigged market baseline that Task C is judged against.

The two sources have **no shared key**; `build_market_baseline.py` tags odds rows to a `match_id` via `(date, home_team, away_team)` and an alias map (`alias_map.csv`), logging coverage (`odds_coverage.csv`) and failures (`odds_failures.csv`).

---

## ⚙️ Setup

```bash
# 1. Create and activate a virtual environment (Python 3.13.7)
python -m venv .venv
source .venv/bin/activate      # Windows: .venv\Scripts\activate

# 2. Install dependencies
pip install -r requirements.txt
#    macOS: xgboost and lightgbm need OpenMP -> brew install libomp
#    model_zoo.py drops either library silently if its import fails.

# 3. Populate src/data/ locally
#    - StatsBomb Open Data  -> src/data/statsbomb_open_data/data/
#    - Football-Data.co.uk  -> src/data/Football_Data/
```

Missing StatsBomb event and lineup files are downloaded on demand from the public raw endpoint and cached; already-cached JSON is always reused.

---

## ▶️ Running the Pipeline

```bash
python src/pipeline/run_all.py                  # everything
SKIP_TUNING=1 python src/pipeline/run_all.py    # reuse an existing best_params.json
```

`run_all.py` executes each stage as a subprocess, in this order:

1. `build_relational_store.py` — match store → labels → lineups → events → cleaning → temporal splits
2. `build_market_baseline.py` — odds tagging and de-vig
3. `test_g_smotenc.py`, `test_hierarchical_shrinkage.py` — P1 and P2 paper tests (part of the run, not optional)
4. `test_inplay_cut.py` — time-`t` leakage tests
5. `build_prematch_features.py`, `build_inplay_features.py`
6. `tuning.py` → `run_models.py` → `resampling_study.py`
7. `market_comparison.py`, `inplay_curves.py`, `kernel_scaling.py`
8. `plot_calibration.py` and the three store/market/raw visualizers

Useful partial entry points: `src/models/train_prematch.py` (C + R only), `src/models/train_inplay.py` (Lc + Lr only), `src/audit/competition_audit.py`.

**Environment overrides**

| Variable | Default | Effect |
|---|---|---|
| `STATSBOMB_LOCAL` | `src/data/statsbomb_open_data/data` | StatsBomb JSON root |
| `FOOTBALL_DATA` | `src/data/Football_Data` | Football-Data CSV folder |
| `SB_DOWNLOAD_WORKERS` | `6` | Concurrent event-file downloads |
| `SB_MAX_MATCHES` | unset | Cap matches ingested (smoke runs) |
| `SKIP_TUNING` | unset | Skip the hyperparameter search |

> `config.py` records the declared leagues, split ratios, snapshot grid, class order and margin clip. It is currently a reference declaration: the pipeline scripts re-derive their own paths from `Path(__file__)`. See `TODO.md`.

---

## 📄 Paper Reimplementations

Both papers are written from scratch against NumPy / scikit-learn primitives; no third-party implementation is imported.

**P1 (data-level) — Geometric SMOTE for Nominal and Continuous features.**
Fonseca & Bação, *Expert Systems with Applications* 234 (2023). `src/papers/g_smotenc.py`, 8 tests. Addresses the Draw minority class. It is a **training-time toggle only** — oversampling is fit inside training rows and never written into a feature file. Task C only; the snapshot table is never oversampled across matches.

**P2 (model-level) — Hierarchical Shrinkage.**
Agarwal, Tan, Ronen, Singh, Yu. *Hierarchical Shrinkage: Improving the accuracy and interpretability of tree-based models.* ICML 2022, PMLR 162:111–135. `src/papers/hierarchical_shrinkage.py`, 8 tests.

$$\hat f_{\mathrm{HS}}(x) = \mu(t_0) + \sum_{l=1}^{L} \frac{\mu(t_l) - \mu(t_{l-1})}{1 + \lambda / N(t_{l-1})}$$

over the root-to-leaf path, with $\mu(t)$ the training mean in node $t$ and $N(t)$ its sample count. One parameter, $\lambda$, chosen by inner CV over `LAMBDA_GRID`. The same algebra serves classification and regression, so P2 covers all three models. Selection evidence: `docs/p2_paper_selection.md`. Hand derivation: `docs/appendix_a_hierarchical_shrinkage.md`.

---

## 🧠 Modeling

**Estimator zoo** (`model_zoo.py`), with the tasks each supports:

| Model | C | R | L |
|---|:-:|:-:|:-:|
| `dummy` (prior / mean) | ✅ | ✅ | ✅ |
| `kernel_svm` (RBF SVC) | ✅ | — | — |
| `kernel_svr` (RBF SVR) | — | ✅ | — |
| `kernel_ridge_exact` | — | ✅ | — |
| `kernel_ridge_nystroem` | — | ✅ | — |
| `random_forest` | ✅ | ✅ | ✅ |
| `gbm` (HistGradientBoosting) | ✅ | ✅ | ✅ |
| `xgboost` | ✅ | ✅ | ✅ |
| `lightgbm` | ✅ | ✅ | ✅ |
| `p2_hier_shrinkage` | ✅ | ✅ | ✅ |

Kernel methods are pre-match only: an exact Gram matrix over the snapshot table is infeasible. `kernel_ridge_exact` therefore fits on a reproducible capped subsample (`EXACT_KERNEL_MAX_TRAIN = 8000`) and reports `n_train` and `subsampled` honestly; `kernel_scaling.py` demonstrates the $O(n^2)$ wall against the Nyström approximation on nested subsamples.

**Tuning** (`tuning.py`): random search over discrete grids, equal candidate budget per model (`N_CANDIDATES = 12`), `K_FOLDS = 3`, folds taken inside training rows only, grouped by `match_id` for snapshot tasks. Writes `tuning_results.csv` (every candidate) and `best_params.json` (selections).

**Calibration** (`modeling_common.calibrate_proba`): `CalibratedClassifierCV` wrapped around a `FrozenEstimator`, fit on the held-out validation split, isotonic first with a sigmoid fallback and an uncalibrated fail-safe. Probabilities are floored at 0.005 and renormalised before log-loss, because isotonic on a few hundred validation rows emits exact zeros. The results table reports `ece_before` and `ece_after` so the effect of the calibrator is visible.

**Metrics.** Classification: ranked probability score, log-loss, multiclass Brier, ECE, and per-class precision / recall / F1 / support for H, D, A. Regression: MAE, RMSE, correlation, with predictions clipped to `[-5, +5]`. Compute cost (`train_seconds`, `peak_memory_mb`) is recorded for every fit.

**Imbalance study** (`resampling_study.py`): six arms — vanilla, SMOTE, Borderline-SMOTE, ADASYN, `class_weight="balanced"`, and P1 G-SMOTENC — on identical splits across `random_forest`, `gbm`, `kernel_svm`, `lightgbm`.

---

## 📈 Generated Outputs

All paths below are under `src/reports/` and are regenerated by the pipeline.

| Folder | Contents |
|---|---|
| `processed/` | `match_store.csv`, `model_targets.csv`, `lineups.csv`, `events_index.csv`, `clean_events.csv`, `clean_lineups.csv`, `team_identity_map.csv`, `player_identity_map.csv`, `cleaning_drops.csv`, `data_quality_log.csv`, `temporal_match_splits.csv`, `snapshot_split_plan.csv`, `market_baseline.csv`, `alias_map.csv`, `odds_coverage.csv`, `odds_failures.csv` |
| `features/` | `prematch_features.csv`, `inplay_features.csv` |
| *(root)* | `model_results.csv`, `model_results_prematch.csv`, `model_results_inplay.csv`, `predictions_{C,R,Lc,Lr}.csv`, `tuning_results.csv`, `best_params.json`, `resampling_study.csv`, `market_comparison.csv`, `market_coverage.csv`, `inplay_metric_by_minute.csv`, `inplay_calibration_by_phase.csv`, `reliability_bins.csv`, `kernel_scaling.csv`, `competition_audit.csv` |
| `visualizations/` | `store_overview.html`, `market_baseline_overview.html`, `raw_result_distribution.html`, `competition_audit.html`, `reliability_diagrams.html`, `inplay_curves.html`, `kernel_scaling.html` |

---

## ⚠️ Data Leakage Discipline

Four barriers, each backed by an assertion in code:

1. **Chronological splits.** `build_temporal_splits.py` assigns whole match *dates* to `train` / `validation` / `test` at 60/20/20, and asserts that train dates end before validation dates, which end before test dates. Every snapshot of a match inherits its parent's split, asserted single-valued per `match_id`.
2. **Prior-match-only form.** `build_prematch_features.py` applies `.shift(1)` before `.rolling(5)`, and asserts that each team's first match carries no rolling form.
3. **The time-`t` prefix cut.** `build_inplay_features.py` sorts events by `(period, index)` — deliberately not by timestamp, since a small number of events carry corrupted `00:00:00` values — takes a running maximum of `minute` as the effective minute, and asserts the cut both ways: no event after `t` is included, and no event at or before `t` is excluded. Covered by `test_inplay_cut.py`.
4. **Train-only transform fitting.** Imputer, scaler, one-hot encoder and every resampler are fit on training rows only inside `prepare_matrices`; validation and test are transformed, never resampled. Requesting any resampling arm on the snapshot table raises.

Additionally: odds are used **either** as the market baseline **or** as a declared feature, never both in the same experiment, and odds-to-`match_id` tagging uses pre-match identity only (date, teams) — never scores or statistics.

---

## 🧱 Status

**Complete**
- ✅ Phases 1–4: ingestion, integration, labeling, cleaning, temporal splits, market baseline
- ✅ Phase 5: both feature pipelines with leakage tests
- ✅ Phase 6: tuning, full model sweep, calibration, imbalance study
- ✅ Phase 7 analyses: market comparison, in-play curves, kernel scaling, reliability diagrams
- ✅ P1 and P2 reimplemented from scratch, with tests and the P2 hand derivation

**Outstanding** (tracked in [`TODO.md`](./TODO.md))
- ⬜ P2 TA sign-off (Hierarchical Shrinkage selected; FIGS is the declared fallback)
- ⬜ SHAP explainability: global beeswarm per model/task, local plots for the ten worst predictions per task, and an in-play SHAP timeline
- ⬜ Model 2 → Model 1 conversion analysis (margin regression to H/D/A probabilities)
- ⬜ Feature-pipeline depth: pressure, passing by zone, carries, set pieces, possession share, defensive actions, head-to-head; in-play event counts and momentum indicators
- ⬜ `REPORT.md` §4–§9 plus Appendix B (reproducibility)

---

## 📚 References

- Agarwal, A., Tan, Y. S., Ronen, O., Singh, C., Yu, B. *Hierarchical Shrinkage: Improving the accuracy and interpretability of tree-based models.* ICML 2022, PMLR 162:111–135. https://proceedings.mlr.press/v162/agarwal22b.html
- Fonseca, J., Bação, F. *Geometric SMOTE for imbalanced datasets with nominal and continuous features.* Expert Systems with Applications 234 (2023).
- StatsBomb / Hudl StatsBomb. *StatsBomb Open Data.* https://github.com/statsbomb/open-data
- Football-Data.co.uk. *Historical Football Results and Betting Odds Data.* https://www.football-data.co.uk
- Full course specification and TA clarifications: [`docs/course-brief/`](./docs/course-brief)

## Building the report

The report exists in two formats built from one source. `src/report/report_content.py`
holds all prose and reads every number from the result CSVs in `src/reports/`.

```bash
python src/report/build_report.py   # writes final_report.pdf + report_content.json
node   src/report/render_docx.js    # writes final_report.docx from that JSON
```

`build_report.py` runs as part of `src/pipeline/run_all.py`; the DOCX step is
run separately because it needs Node. If a result CSV is missing, the report
prints an explicit marker naming the script that produces it rather than
omitting the section or inserting a placeholder number.
