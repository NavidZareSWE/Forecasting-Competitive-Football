* ✅ Completed
* 🟡 Partially Implemented
* ❌ Missing


I'll start by extracting and examining both archives.Now let me read all the defence files in full.While that runs, let me examine the remaining pipeline files and run the test suites.While the SVM runs, let me examine the explainer HTML and remaining pipeline files.# Implementation Audit — Forecasting Competitive Football (Mid-Defence 2)

> **Standing reminder:** you still need to select the **P2 paper**. It is not implemented, and it is *not* counted below as work you failed to do — it is tracked as an open, deliberately deferred decision. Every P2-related row in this audit reflects that status.

---

## 1. Executive Summary

The repository is in a **more advanced state than its own documentation claims**, but it is **not runnable end-to-end for Mid-Defence 2 as delivered**. Three independent blockers were confirmed by execution, not inference.

`code.zip` contains a complete, working Phase 1–5 data foundation (ingestion, cleaning, labelling, temporal splits, market baseline, both feature pipelines) plus a from-scratch G-SMOTENC reimplementation and a model zoo. `For_Defence.zip` contains exactly the five files that were missing to make Phase 6 executable: `modeling_common.py`, `train.py`, `run_models.py`, `train_prematch.py`, `train_inplay.py`. There is **zero file overlap** between the two archives — nothing in the Defence bundle duplicates or supersedes existing repository code. All five files belong in `src/models/`, alongside the already-present `model_zoo.py`.

**However, copying the Defence files into `src/models/` is not sufficient.** I executed the integrated project and confirmed:

1. **`modeling_common.py` line 22 has a wrong path.** `PAPERS_DIR = PROJECT.parent / "papers"` resolves to `<repo-root>/papers`, which does not exist; the module lives at `src/papers/g_smotenc.py`. Every P1-enabled run raises `ModuleNotFoundError: No module named 'g_smotenc'`. This kills the single highest-weight deliverable in the checklist.
2. **XGBoost crashes on all classification tasks.** `XGBClassifier.fit` rejects the string labels `H`/`D`/`A` with `ValueError: Invalid classes inferred from unique values of 'y'. Expected: [0 1 2], got ['A' 'D' 'H']`. This is currently *masked* — xgboost is not installed in my base image, so the guarded import silently returned `{}`. On any machine where `requirements.txt` is honoured (i.e. the grader's), `run_models.py` aborts partway through Task C.
3. **`test_g_smotenc.py` is broken and untracked.** It calls `_geometric_continuous(..., alpha_trunc=1.0, alpha_def=1.0)` but the implementation signature is `(center, neighbor, truncation, deformation, rng)` → `TypeError`. The runner has no exception handling, so it dies on test 2 of 8 and reports only `1/8`. Worse: `.gitignore` contains the pattern `test*.py`, so **both** test files are absent from version control (`git ls-files` returns 26 files, neither test among them). Your unit-tested cut assertion — an explicitly graded artefact — is not in the repository.

After I applied the one-line path fix, everything else worked. Tasks C and R trained end-to-end across the full zoo; P1 balanced the training set correctly (`Counter({'A': 388, 'D': 388, 'H': 388})` from 896 rows; in-play: 7,372 per class from 17,024); Task Lc/Lr ran through Random Forest and GBM before I hit my tool budget mid-SVM. The 7 in-play cut tests pass, and all 8 G-SMOTENC tests pass once the keyword names are corrected.

Two documentation deliverables named in the Mid-Defence 2 brief are **entirely absent from both archives**: a hyperparameter tuning protocol and a written calibration strategy. `REPORT.md` stops at Section 3 (Temporal Split). There is no `modeling_protocol.md` anywhere — I searched both archives and the git tree.

**Estimated completion: ~72% of Mid-Defence 2**, with roughly 4–6 hours of focused work remaining, most of it writing rather than coding.

---

## 2. File Comparison

### 2.1 Files arriving from `For_Defence.zip`

All five are **New** — no counterpart exists in `code.zip`. Target directory for all five: `src/models/`.

| File | Status | Purpose | Required Action |
|---|---|---|---|
| `modeling_common.py` | **New** | Loads feature CSVs, assembles the four task frames (C/R/Lc/Lr), performs train-only impute→scale→P1→one-hot, defines RPS/Brier/log-loss/ECE, wraps fail-safe calibration | **Fix line 22:** `PROJECT.parent / "papers"` → `PROJECT / "papers"`. Add a label-encoding path for XGBoost. Then place in `src/models/` |
| `train.py` | **New** | Per-model evaluation harness; times the fit, computes pre- and post-calibration metrics, clips regression output to `[-5, +5]` | Place in `src/models/`. No changes strictly required |
| `run_models.py` | **New** | Orchestrator across all four tasks × full zoo × P1 on/off; writes `model_results.csv` | Place in `src/models/`. Add market-baseline rows (see §4) |
| `train_prematch.py` | **New** | Thin entry point for Tasks C and R only | Place in `src/models/`. Note: computes rows but discards them — prints a count, writes nothing |
| `train_inplay.py` | **New** | Thin entry point for Tasks Lc and Lr only | Place in `src/models/`. Same discard behaviour |

### 2.2 Files already in `code.zip`

| File | Status | Purpose | Required Action |
|---|---|---|---|
| `src/models/model_zoo.py` | Already Present | Factory dicts for 6 classifiers / 7 regressors; guarded optional boosters | **Modify** — `dummy` must be excluded from regression P1 loops correctly (it is), but the XGB classifier needs integer labels |
| `src/papers/g_smotenc.py` | Already Present | From-scratch G-SMOTENC (Algorithms 1–3) | Keep as-is. Verified faithful to the paper (§3) |
| `src/papers/test_g_smotenc.py` | **Broken + Untracked** | 8 unit tests for P1 | **Fix** the two keyword args; **un-ignore** in `.gitignore`; commit |
| `src/features/build_prematch_features.py` | Already Present | Phase 5A — one row per match, prior-only rolling form | Keep. Verified: 1,517 unique rows, leakage assertion fires correctly |
| `src/features/build_inplay_features.py` | Already Present | Phase 5B — 19 snapshots/match via effective-minute cut | Keep. Verified: 28,823 rows, 1,517 matches |
| `src/features/test_inplay_cut.py` | **Untracked** | 7 tests proving the cut is a clean prefix | **Un-ignore** in `.gitignore`; commit. Tests themselves pass 7/7 |
| `src/pipeline/*.py` (9 files) | Already Present | Phases 1–4 | Keep |
| `src/pipeline/run_all.py` | **Modified needed** | Top-level orchestrator | **Modify** — currently stops after visualisations; does not invoke Phase 5 or Phase 6 |
| `src/viz/*.py` (3 files) | Already Present | Diagnostic HTML dashboards | Keep |
| `src/audit/competition_audit.py` | Already Present | Competition/season selection justification | Keep |
| `config.py` | **Dead code** | Declares `PROJECT`, `DATA_DIR`, `REPORTS_DIR`, `LEAGUES`, `SNAPSHOT_MINUTES`, `SPLIT_RATIOS`, `CLASS_ORDER`, `MARGIN_CLIP` | **Modify or remove.** `grep` confirms **zero** imports project-wide. Its `REPORTS_DIR = PROJECT / "reports"` is also *wrong* — every module actually uses `src/reports`. Its own commit message admits "(not in use right now)" |
| `README.md` | **Stale** | Repository overview | **Modify** — marks Phase 5, models, and P1 as `⬜` when all are done; references non-existent `Extras/` and `src/scripts/` paths |
| `REPORT.md` | **Incomplete** | Formal problem framing | **Extend** — ends at §3; also contains a contradiction (§5.2) |
| `.gitignore` | **Harmful** | Ignore rules | **Modify** — `test*.py` on line 47 excludes both test suites from git |

### 2.3 Referenced but missing

| Referenced item | Referenced by | Status |
|---|---|---|
| `feature_common.py` | `football_pipeline_explainer.html`, "Files & dependencies" table, marked *"yes"* under Needed | **Does not exist** in either archive. The described responsibilities (shared paths, feature manifest, effective-clock cut) are instead duplicated inline across `build_prematch_features.py` and `build_inplay_features.py`. Either write the module or correct the explainer |
| `reports/modeling_protocol.md` | Mid-Defence 2 deliverable list | **Does not exist** anywhere |
| `market_baseline.csv` consumption | `REPORT.md` §"Source and Split Contract", explainer ("scored against the market") | File **is produced** (1,516 rows, 99.9% coverage) but `grep` confirms **no modeling code reads it** |

---

## 3. Implementation Checklist

### Phase 5A — Pre-match pipeline

| Requirement | Status | Evidence |
|---|---|---|
| One row per match | ✅ | `build_prematch_features.py:112` — `assert prematch["match_id"].is_unique`. Verified: 1,517 rows, 1,517 unique `match_id`, merged with `validate="one_to_one"` at line 107 |
| Features from prior matches only | ✅ | `_prior_window_mean()` (line 44–45) applies `.shift(1).rolling(5, min_periods=1).mean()` — the shift precedes the window, so match *m*'s own outcome can never enter its own features |
| No future leakage | ✅ | Explicit self-check at lines 99–102: every team's chronologically first match must carry `NaN` across all six form columns. Confirmed in the shipped CSV — row 1 (`match_id 3829413`) has empty form fields and `played_prior = 0` |

### Phase 5B — In-play pipeline

| Requirement | Status | Evidence |
|---|---|---|
| Event-level, many rows per match | ✅ | 28,823 rows = 1,517 matches × 19 snapshots at *t* ∈ {0, 5, …, 90}. Verified directly from `inplay_features.csv` |
| Uses only events at or before *t* | ✅ | `effective_minute()` (line 24) takes `np.maximum.accumulate` over `minute` in **index** order, repairing corrupted `00:00:00` timestamps; `prefix_length_at()` uses `searchsorted(..., side="right")` |
| Unit-tested cut assertion | 🟡 | The assertion exists (`assert_prefix`, lines 34–40) and **7/7 tests pass** — including `test_corrupted_minute_not_pulled_early` (asserts `eff[5] == 10`, not 0) and `test_no_future_event_leaks_into_snapshot`. **Downgraded to partial because `test_inplay_cut.py` is excluded from git by `.gitignore`'s `test*.py`** — a grader cloning the repository receives no tests |
| P1 toggled by config flag | 🟡 | A toggle exists — `prepare_matrices(..., use_p1=False)` and the `for use_p1 in [False, True]` loop in `run_models.py:23`. But it is a **function parameter**, not a configuration flag; `config.py` (the natural home) is dead code and contains no P1 setting. Correctly implemented as *training-time only* — P1 never touches the feature CSVs, which is the right leakage discipline |

### Phase 6 — Modeling

| Requirement | Status | Evidence |
|---|---|---|
| **P1 (G-SMOTENC)** | 🟡 | Implementation is complete and faithful. **Blocked at integration** by the `PAPERS_DIR` bug. After my one-line fix: 896 → 1,164 rows, perfectly balanced `{'A': 388, 'D': 388, 'H': 388}`; in-play 17,024 → 22,116, `{'A': 7372, 'D': 7372, 'H': 7372}` in 16.9 s |
| **Dummy** | ✅ | `DummyClassifier(strategy="prior")` / `DummyRegressor(strategy="mean")`. Ran: Task C `rps=0.22812`; Task R `mae=1.48997, corr=0.0` |
| **Random Forest** | ✅ | Ran: C `rps=0.22341` (P1: `0.21990`); R `mae=1.38430, corr=0.36326`; Lc `rps=0.15219` (P1: `0.15151`) |
| **GBM** | ✅ | `HistGradientBoosting*`. Ran: C `rps=0.22297` (P1: `0.22319`); R `mae=1.46952`; Lc `rps=0.16859` (P1: `0.16831`) |
| **XGBoost** | ❌ | **Classification crashes.** `ValueError: Invalid classes inferred from unique values of 'y'. Expected: [0 1 2], got ['A' 'D' 'H']`. Regression works (`mae=1.48274, corr=0.22355`). The guarded `try/except ImportError` hides this entirely when the package is absent |
| **LightGBM** | ✅ | Both paths work. C: `rps=0.28461, ece=0.33418` — *worse than Dummy and severely overconfident*, which is itself evidence for the missing tuning protocol. R: `mae=1.49820, corr=0.26639` |
| **Kernel methods** | ✅ | `SVC(rbf, probability=True)`, `SVR(rbf)`, `KernelRidge(rbf)`. Ran: C `rps=0.22718`; R `kernel_svr mae=1.42528`, `kernel_ridge mae=1.43197`. Cost note: in-play SVM fit alone takes **94.4 s** on 17,024 rows |
| All three tasks train end-to-end | 🟡 | C and R: fully verified. Lc/Lr: verified through Dummy, RF, GBM before my budget expired; task assembly, P1, and matrix shapes all confirmed for both. **Cannot be ✅ while XGBoost aborts the classification loop** |
| **P2 from paper, not library** | ❌ | *Paper not yet selected — deferred by design, see reminder above.* Nothing to audit |
| Appendix A derivation begun | ❌ | Blocked on P2 selection. No `appendix`, `derivation`, or LaTeX artefact in either archive |

### Mid-Defence 2 Deliverables

| Deliverable | Status | Evidence |
|---|---|---|
| Model paper + core mathematics | 🟡 | P1 fully covered: PDF at `docs/papers/`, verified as Fonseca & Bacao, *Expert Systems With Applications* **234** (2023) 121053. Implementation checked line-by-line against Algorithms 1–3 (§5.3). P2 outstanding |
| Complete data pipeline including P1 | 🟡 | Phases 1–5 verified against real outputs: 1,517 matches, **100% score reconstruction** (`events_index.score_ok` is `True` for all 1,517), 896/277/344 chronological split with zero date overlap, odds coverage 1.00/0.99/1.00/1.00. P1 integration blocked by the path bug |
| All models training successfully | 🟡 | 12 of 12 pre-match rows produced; XGBoost classification blocks completeness |
| Hyperparameter tuning protocol | ❌ | **No tuning of any kind exists.** Every model in `model_zoo.py` uses hardcoded constants (`n_estimators=300`, `max_iter=300`, `C=1.0`, `alpha=1.0`). No grid, no random search, no `GridSearchCV`. The validation split is used *only* for calibration — never for model selection. No protocol document |
| Calibration strategy | 🟡 | Implemented well: `calibrate_proba()` wraps `CalibratedClassifierCV(FrozenEstimator(estimator))`, tries isotonic then sigmoid, falls back to `"uncalibrated"` — correctly fit on **validation**, applied to **test**. All 12 runs reported `cal=isotonic`. **Not documented anywhere in prose** |
| Snapshot leakage documentation | 🟡 | Distributed across three places: `REPORT.md` §3, the `README.md` "Data Leakage Discipline" section, and the reasoning comments at `build_inplay_features.py:18–31`. No consolidated document, and `REPORT.md` **contradicts the code** (§5.2) |

---

## 4. Missing Work

### Critical — required for Mid-Defence 2

1. **Fix `modeling_common.py:22`** → `PAPERS_DIR = PROJECT / "papers"`. One line. Without it P1 cannot run at all.
2. **Fix XGBoost label handling.** Two viable approaches: (a) fit a `LabelEncoder` on `y_train` inside `prepare_matrices`, carry `classes_` through `_proba_in_order`; or (b) wrap `XGBClassifier` in a small adapter in `model_zoo.py` that encodes on fit and decodes on `predict_proba`. Option (b) is less invasive and keeps `CLASS_ORDER` semantics intact everywhere else.
3. **Fix `test_g_smotenc.py`** — rename `alpha_trunc=` → `truncation=` and `alpha_def=` → `deformation=` (or add keyword-compatible parameter names to `_geometric_continuous`). Confirmed: **8/8 pass** after this change.
4. **Remove `test*.py` from `.gitignore`** and commit both test files. This is currently the single biggest silent risk in the repository — your graded leakage proof does not exist in the cloned repo.
5. **Write the hyperparameter tuning protocol** and implement at least a minimal version. A time-respecting search on the validation split (no k-fold shuffling) with a stated budget per model family would satisfy the requirement. The LightGBM result (`rps=0.28461` vs Dummy's `0.22812`, `ece=0.33418`) is concrete, defensible evidence in your report that untuned defaults are inadequate.
6. **Write the calibration strategy document** — the code is sound; it just needs prose. Cover: why isotonic first, why validation-fit, why `FrozenEstimator`, and what the `try/except` fallback protects against.
7. **Consolidate leakage documentation** into `reports/modeling_protocol.md`, covering the snapshot table specifically.

### Important

8. **Resolve the `REPORT.md` §3 contradiction** (see §5.2). A TA reading the report and then the code will find a direct conflict.
9. **Wire the market baseline into evaluation.** `market_baseline.csv` exists with de-vigged `p_home/p_draw/p_away` and 99.9% coverage, but no model is ever compared against it. Add a baseline row to `model_results.csv` scored with the same RPS/Brier/log-loss/ECE functions — this is cheap and it is the benchmark your whole project is framed around.
10. **Extend `run_all.py`** to invoke Phase 5 and Phase 6. It currently stops after the visualisations.
11. **Update `README.md`** — the status table marks completed work as `⬜`, and `Extras/` / `src/scripts/` paths do not exist.
12. **Resolve `config.py`.** Either import it everywhere (and fix `REPORTS_DIR` to `PROJECT / "src" / "reports"`) or delete it. Right now it is a trap: a plausible-looking constants file that is wrong and unused.
13. **Make `train_prematch.py` / `train_inplay.py` persist their results**, or document that they are diagnostic entry points only.
14. **Resolve `feature_common.py`** — write it, or correct the explainer's dependency table.

### Optional

15. Reduce the in-play kernel-SVM cost (subsample, or `LinearSVC` + calibration) — 94 s per fit × P1 variants is slow for iteration.
16. `snapshot_minute` is passed to G-SMOTENC as a **continuous** feature, so synthetic rows carry fractional minutes (e.g. 33.7) that cannot occur in real data. Consider declaring it nominal, or note the choice explicitly in your defence.
17. Add a seed-variation run to report metric stability.
18. `_knn_indices` recomputes neighbours inside the generation loop — correct but O(N·n).

---

## 5. Data Leakage Verification

### 5.1 What is correctly prevented

The leakage discipline is the strongest part of this project. Four independent barriers, each verified:

**Temporal split integrity.** `build_temporal_splits.py` assigns *whole dates* to splits, then asserts strict ordering (lines 50–55). Confirmed empirically — train ends 2016-01-31, validation runs 2016-02-01 → 2016-03-18, test runs 2016-03-19 → 2016-05-17. Zero overlap. Assigning by date rather than by row is the right call given incomplete kick-off times, and it is defensible under questioning.

**Prior-only pre-match features.** The `.shift(1)` before `.rolling(5)` is the correct order. The assertion at lines 99–102 proves it holds for every team's first match. Note for your defence: a test match's form window may draw on training-period matches — this is *not* leakage, since those matches concluded before kick-off.

**The in-play cut.** The effective-minute construction is the most defensible piece of engineering here. Because `np.maximum.accumulate` produces a non-decreasing sequence, `searchsorted` is guaranteed to return a **contiguous index-ordered prefix** — the property `assert_prefix` checks from both sides (line 36: nothing after *t* included; line 39: nothing at-or-before *t* excluded). `test_cut_is_prefix_and_nested` additionally verifies prefix lengths are monotone in *t*.

**Train-only fitting.** In `prepare_matrices`, `SimpleImputer` and `StandardScaler` call `fit_transform` on train and plain `transform` on validation/test (lines 83–84 vs 103). G-SMOTENC runs *after* the train-only fit and *only* on the training block. `OneHotEncoder` uses `handle_unknown="ignore"`. Calibration fits on validation, scores on test. All four are correct.

**Snapshot cohesion.** Asserted twice — `build_temporal_splits.py:67` and `build_inplay_features.py:136`. Confirmed: 17,024 / 5,263 / 6,536 in-play rows, no match spanning a boundary.

### 5.2 Where risk remains

**A documentation-vs-code contradiction.** `REPORT.md` §3 states that event records *"must be ordered by period, timestamp, and event index"*. `clean_store.py:52` does exactly that. But `build_inplay_features.py:93` **re-sorts by `index` alone** — deliberately, because ~26 events across 24 matches carry corrupted `00:00:00` timestamps that would otherwise sort to the front of their period. The code is right; the report is wrong. Fix the prose, not the code. A TA who reads both will ask about this, and "the report is stale" is a much better answer than being unable to explain the discrepancy.

**Untested, unversioned proof.** The cut assertion is only as valuable as its evidence. With `test*.py` in `.gitignore`, the proof does not survive a clone. This is a leakage-verification failure in the *process* sense even though the runtime behaviour is correct.

**Not a leak, but worth pre-empting.** At *t* = 90, `inplay_goal_diff` nearly determines `label_margin`. This is the task definition, not leakage — but expect the question, and have the framing ready: the model is asked what the final result will be given everything observed through minute 90, and near-certainty at that horizon is the correct behaviour.

**No leakage found in the P1 path.** G-SMOTENC is invoked only inside `prepare_matrices` on the training block, never written to `prematch_features.csv` or `inplay_features.csv`. Confirmed both by reading and by the fact that the shipped feature CSVs contain exactly 1,517 and 28,823 rows — no synthetic rows.

**One inherited-risk note.** `load_inplay()` merges pre-match features into snapshots after dropping `split`, `label_result`, `label_margin`, and `match_date` (line 40–41). Correct — but it retains `competition_name` from the pre-match frame, which is fine, and no post-match field survives the drop. Verified by inspecting the 35 continuous columns produced for Task Lc.

---

## 6. Pipeline Architecture

**Stage 1 — Ingestion.** `build_match_store.py` parses StatsBomb `matches/{competition_id}/{season_id}.json` for the four configured leagues (2/27 Premier League, 7/27 Ligue 1, 11/27 La Liga, 12/27 Serie A). `build_event_store.py` downloads and parses `events/{match_id}.json` concurrently; `build_lineup_store.py` handles lineups. Output: `match_store.csv` (1,517 matches), `events_index.csv`, `lineups.csv`.

**Stage 2 — Labelling.** `build_label_store.py` derives `label_result` ∈ {H, D, A} and `label_margin` = clip(home − away, −5, +5) once, from the match file only, and writes `model_targets.csv`. The 100% `score_ok` rate in `events_index.csv` proves event-derived goals reconcile with the official score for every match — this is a strong number to lead with in your defence.

**Stage 3 — Cleaning.** `clean_store.py` orders events by `(period, timestamp, index)`, drops rows lacking mandatory identity fields (logged with reasons to `cleaning_drops.csv`), retains missing optional fields (counted in `data_quality_log.csv`), and builds canonical `team_identity_map.csv` / `player_identity_map.csv` by ID rather than spelling. Produces the 903 MB `clean_events.csv`.

**Stage 4 — Market baseline.** `build_market_baseline.py` joins Football-Data.co.uk odds to `match_id` on `(date, home_team, away_team)` using accent-stripping, token similarity, Levenshtein distance, and one documented manual override (`"Ath Madrid"` → Atlético, because no string metric can separate Athletic from Atlético). De-vigs proportionally: `p_i = (1/o_i) / Σ(1/o_j)`, asserting overround > 1 pre-normalisation. Coverage: La Liga 1.00, Ligue 1 0.9895, Premier League 1.00, Serie A 1.00.

**Stage 5 — Splits.** `build_temporal_splits.py` produces `temporal_match_splits.csv` and `snapshot_split_plan.csv` (28,823 planned snapshots).

**Stage 6 — Feature pipelines.** 5A builds a team-match long table, computes lagged rolling form, pivots to one row per match with home/away/diff columns → `prematch_features.csv` (23 continuous + 1 nominal). 5B walks each match's events in index order, applies the effective-minute cut at each *t*, and computes 12 state features → `inplay_features.csv`. Joined with pre-match at model time, Task L sees 35 continuous + 1 nominal.

**Stage 7 — Modeling (the Defence files).** `modeling_common.task_frame(task)` assembles one of four frames. `prepare_matrices` runs impute → scale → *optional P1* → one-hot, all train-fitted. `model_zoo` supplies estimator factories. `train.evaluate_classification` fits, scores raw, calibrates on validation, re-scores. `run_models.main` sweeps all four tasks × zoo × P1 and writes `model_results.csv` to `src/reports/`.

**The seam that breaks:** `_apply_p1` imports `g_smotenc` from `sys.path`, which `modeling_common` populates from `PAPERS_DIR` — and that path is wrong. This is the only structural defect in an otherwise clean architecture.

---

## 7. Final Verdict

**Approximately 72% of Mid-Defence 2 is complete.**

Breaking that down against the checklist's own weighting: Phase 5A is fully done; Phase 5B is functionally done but its proof is unversioned; Phase 6's model suite is 6 of 7 working; P1 is written correctly but disconnected; P2 is an open decision by design; and two of six deliverables (tuning protocol, calibration write-up) do not exist in any form.

**Are the Defence files alone sufficient? No.** They are necessary — they supply the entire training harness that `code.zip` lacks, with no duplication and no conflicts — but three code defects must be fixed before anything runs as intended, and two documents must be written from scratch. Dropping the five files into `src/models/` and running `run_models.py` on a machine with `requirements.txt` installed produces a `ModuleNotFoundError` the moment P1 is enabled, and a `ValueError` the moment XGBoost is reached.

**What remains, in the order I would do it:**

The three code fixes are small — the path is one line, the test keywords are two, and XGBoost label encoding is perhaps fifteen. The `.gitignore` change is one line and is the highest value-per-character edit available to you. That is maybe forty-five minutes of work and it takes you from "does not run" to "runs completely."

The remaining time is writing. `reports/modeling_protocol.md` needs to cover the hyperparameter protocol, the calibration strategy, and the four-property leakage account. You have unusually good raw material for it: real numbers from real files, a genuine data-quality problem (corrupted timestamps) that you solved with a defensible technique, and a concrete demonstration that untuned defaults fail. The LightGBM result in particular — worse than a prior-strategy Dummy, with an ECE of 0.334 — is the kind of honest negative finding that reads as rigour rather than as a mistake, provided you frame it as motivation for the tuning protocol rather than leaving it unexplained.

Wiring in the market baseline is the one substantive addition I would prioritise beyond the fixes. Your entire project framing rests on beating the de-vigged market, the data is already sitting in `market_baseline.csv` at 99.9% coverage, and the metric functions you need are already written. It is a small amount of code for a large amount of defensibility.

And P2 remains open — deliberately so. When you evaluate candidates, the hard filters you have set (non-deep, 2022+, Q1 or CORE A\*/A, applicable to both classification and regression, hand-derivable for Appendix A) are restrictive enough that it is worth confirming a candidate satisfies all five before committing implementation time.