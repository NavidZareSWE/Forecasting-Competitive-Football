# Forecasting Competitive Football — Report

## 1. Problem Framing and Formal Definitions

The dataset contains professional football matches from the selected StatsBomb competition-seasons. `match_id` is the parent key throughout. Final labels are derived once from `home_score` and `away_score` in the StatsBomb match file. `build_label_store.py` validates them and writes the canonical `src/reports/processed/model_targets.csv`; labels must never be recomputed from an odds row or used as a feature.

### Model 1 — Pre-Match Outcome Classification (Task C)

One example is one scheduled match *m*, represented at its kick-off time by a single pre-match feature vector. The classification label is the final result: `H` when `home_score > away_score`, `D` when scores are equal, and `A` otherwise. The model predicts a calibrated distribution \(P(H), P(D), P(A)\).

The target match may contribute only fixture identity known before kick-off: its competition, season, date, kick-off time, and home and away teams. Features may use match, event, and lineup records from matches completed strictly before *m*'s kick-off. In particular, prior-team form and prior event aggregates are permissible. The target match's score, margin, events, cards, post-match lineup fields, and final statistics are forbidden. De-vigged odds are retained as a separate market baseline, not a Model 1 feature in the baseline-comparison experiment.

### Model 2 — Pre-Match Goal-Margin Regression (Task R)

One example and the permitted information are exactly the same as for Model 1: one vector for match *m* at kick-off. Its label is the final signed margin

\[
y_R = \operatorname{clip}(\text{home_score} - \text{away_score}, -5, 5).
\]

The current match store exposes this label as `margin`; `margin_raw` remains available for audit only. The model predicts the expected final signed margin, so a positive estimate favours the home team. Keeping the observation time and feature set identical to Task C makes any difference in performance attributable to the target, not to access to extra information.

### Model 3 — In-Play Prediction (Task L)

One example is a snapshot \((m,t)\) of a single match at clock time *t*. We will create snapshots every five regulation minutes, \(t \in \{0,5,\ldots,90\}\). Each snapshot has the same two final labels as its parent match: the `H`/`D`/`A` result for classification and clipped signed margin for regression. Thus one match yields several correlated examples.

A snapshot may combine the frozen pre-match vector with current-match events whose match clock is at or before *t*. Permitted in-play quantities include current score, cards/man advantage, and counts or rates over prior windows. Event records must be ordered by period, timestamp, and event index; the feature builder must assert that no selected event is later than *t*. Starting-lineup facts known by kick-off may be used, but card counts, substitutions, and other lineup facts observed after *t* may not. StatsBomb 360 data are excluded because the selected four league seasons have no 360 coverage.

### Source and Split Contract

| Source | Permitted contribution |
| --- | --- |
| `matches/{competition_id}/{season_id}.json` | Fixture metadata and final labels; historical matches only for pre-match aggregates. |
| `events/{match_id}.json` | Historical matches for Tasks C/R; only the target match's events at or before *t* for Task L. |
| `lineups/{match_id}.json` | Historical lineups; target starting lineup only when known by the relevant prediction time. |
| Football-Data CSV / `market_baseline.csv` | Identity-only odds join and evaluation baseline; never the source of labels. |

All examples must be split chronologically by parent match before feature fitting. Every snapshot of one `match_id` inherits that match's split; snapshots from the same match must never appear on opposite sides of a train, validation, or test boundary.

## 2. Cleaning and Identity Contract

`clean_store.py` orders every retained event by `(period, timestamp, index)` and preserves its stable StatsBomb `team_id` and `player_id`. It builds canonical team and player maps from all observed names, so later feature code joins entities by ID rather than by spelling. Rows lacking an event identity, event index, period, timestamp, match identifier, team identifier, or player identifier are removed only when those fields are mandatory, and every removal is written to `cleaning_drops.csv` with a reason. Missing optional event fields (for example coordinates, xG, cards, or player identity on match-level events) and missing lineup coverage are retained and counted in `data_quality_log.csv` instead of being silently discarded.

## 3. Temporal Split and Snapshot Contract

`build_temporal_splits.py` sorts parent matches by match date and assigns whole dates—not individual rows—to a 60% train, 20% validation, and 20% test split. This preserves chronology and prevents same-day fixtures from crossing a boundary when kick-off data are incomplete. It writes `temporal_match_splits.csv`, with exactly one split per `match_id`.

For the live task, the fixed schedule is one snapshot at each regulation minute \(t \in \{0,5,10,\ldots,90\}\). `snapshot_split_plan.csv` repeats the parent match's split and final labels for each scheduled snapshot. It is a planning table only: Phase 5 must add features using events at or before *t* and must not use final labels as predictors. The split assertions guarantee that all snapshots from a parent match remain together.
