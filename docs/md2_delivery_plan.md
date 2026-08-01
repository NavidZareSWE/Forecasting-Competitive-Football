# Mid Defence 2 — delivery plan (milestones and PRs)

Everything below exists in the working tree and has been run end to end. This
document is the plan for landing it in reviewable pieces, in dependency order.

The branches are **stacked**: each PR targets the previous one as its base, not
`main`. Merge them into `main` in numeric order.

```
main
 ├─ fix/env-pins-and-inplay-cut              PR 1
 └─ docs/p2-paper-selection                  PR 2
      └─ feat/p2-hierarchical-shrinkage      PR 3
           └─ feat/model-training-harness    PR 4
                └─ feat/md2-required-analyses PR 5
                     └─ chore/pipeline-and-defence-pack PR 6
```

Review order matters: **Milestone B is the sign-off gate.** If the defence is
imminent, land B first — A is a prerequisite only for reproducing the numbers,
not for defending the paper.

---

## Milestone A — Blockers cleared

*Goal: the repo runs correctly on this machine and the leakage tests pass.*

### PR 1 — `fix/env-pins-and-inplay-cut`

| | |
|---|---|
| Files | `requirements.txt`, `src/features/build_inplay_features.py` |
| Base | `main` |
| Type | fix |

- Pins every dependency to the versions the reported results were produced with,
  adds `psutil`/`scipy`, and documents `brew install libomp` — without it
  xgboost and lightgbm fail to import and `model_zoo` silently drops them,
  so a sweep quietly runs 4 of 6 classifiers.
- Snapshot events sorted by `(period, index)` with a fallback to `index` alone
  when the frame has no `period` column. The unguarded version raised
  `KeyError: 'period'` on the unit-test fixtures and took the graded leakage
  proof from 7/7 to 0/7.
- Recent-event window computed on the repaired *effective* minute rather than
  the raw `minute` column, which is wrong for exactly the corrupted rows the
  repair exists for.

**Acceptance:** `test_inplay_cut.py` 7/7 · `test_g_smotenc.py` 8/8 ·
`classifier_zoo()` returns 6 · xgboost 3.3.0 and lightgbm 4.6.0 both import.

---

## Milestone B — P2 paper (**TA sign-off gate**)

*Goal: the paper is picked, defensible against all four hard filters, and
reimplemented from scratch with the Appendix A derivation.*

### PR 2 — `docs/p2-paper-selection`

| | |
|---|---|
| Files | `docs/p2_paper_selection.md` |
| Base | `main` |
| Type | docs |

The candidate audit and the decision record. Hierarchical Shrinkage (ICML 2022,
CORE A\*) selected. Records *why the rejects were rejected*, which is what the TA
will probe: DRF (JMLR 2022) and GPBoost (JMLR 2022 / TPAMI 2023) fail the
from-scratch requirement — both are large C++/Rcpp codebases whose approximations
are not specified in the paper text. NGBoost (2020) and PGBM (2021) fail the date
filter.

**Acceptance:** all four hard filters evidenced with a citable source · every
rejected candidate carries a specific reason.

### PR 3 — `feat/p2-hierarchical-shrinkage`

| | |
|---|---|
| Files | `src/papers/hierarchical_shrinkage.py`, `src/papers/test_hierarchical_shrinkage.py`, `docs/appendix_a_hierarchical_shrinkage.md` |
| Base | PR 2 |
| Type | feat |

From-scratch reimplementation against the raw `tree_` arrays — the shrinkage
recursion, the leaf descent and the λ selection are all ours; sklearn only grows
the underlying CART, exactly as the paper does. Classifier and regressor variants,
so one method serves Tasks C, R and L. λ chosen by 3-fold CV **inside training
rows only**, because the validation split is reserved for calibration.

Appendix A derives both the convex-combination form (which proves the classifier
output stays on the simplex, so no clipping is needed) and the ridge objective
whose greedy solution is the paper's rule, with penalty weight
`N(t)/N(parent(t))`.

**Acceptance:** 8/8 tests — λ=0 ≡ plain CART · λ→∞ ≡ root mean · simplex closure
at every λ · an independent per-path transcription of the paper equation matching
the fast recursion · hand-written leaf descent matching `tree_.apply`.

---

## Milestone C — Models train end to end

*Goal: "held once your baseline models train end to end" — the literal
precondition for this defence.*

### PR 4 — `feat/model-training-harness`

| | |
|---|---|
| Files | `src/models/modeling_common.py`, `model_zoo.py`, `train.py`, `tuning.py`, `run_models.py`, `train_prematch.py`, `train_inplay.py` |
| Base | PR 3 |
| Type | feat |

The harness modules import one another, so they land together — reviewing the
zoo without `modeling_common` is meaningless.

**Task assembly and preprocessing.** Train-only imputers, scalers and resampling;
the six resampling arms; hand-rolled RPS / Brier / log-loss / ECE / per-class
P-R-F1; the isotonic→Platt→raw calibration chain fitted on validation and applied
to test; wall-clock and peak-RSS instrumentation for the compute table. Per-row
test predictions are persisted so every downstream analysis joins to them instead
of refitting.

Two guards worth calling out in review:

- `prepare_matrices` **raises** if any resampling arm is requested on a frame
  containing `snapshot_minute` (brief §7.2 forbids oversampling across matches).
- All probabilities are floored at 0.005 and renormalised. Isotonic fitted on a
  ~340-row validation split emits exact zeros; RPS and Brier barely notice but
  log-loss blew up to 1.9–2.8 nats, making every model look far worse than the
  market for a reason that was an artefact of the calibrator. Post-fix: 1.08–1.22.

**Task-aware zoo.** `TASK_SUPPORT` declares which model runs on which task,
straight from the brief's model table, so the sweep cannot silently run a model
where it was not specified. Builders take `(random_state, **params)` so the tuner
and the sweep share one definition of every model. Kernel methods are declared
pre-match only — deliberate, not an omission: an exact Gram on 17,024 in-play
rows is 2.2 GB, and the cost is quantified in PR 5 rather than paid in the sweep.

**Tuning protocol.** Random search, **12 configurations × 3 folds = 36 fits,
identical for every tuned model**, per task. Random rather than grid because a
full grid would silently give LightGBM 729 configs and Kernel SVM 20 — that is
the unfair-budget failure the brief calls out. `dummy` is excluded rather than
given a free pass. CV runs *inside the training split only*; `GroupKFold` on
`match_id` for Task L so minute 40 and minute 45 of the same match never straddle
a fold. Tables over 8,000 rows are tuned on a seeded subsample of **whole
matches**, declared per row in the audit CSV.

**Acceptance:** C → 7 classifiers · R → 9 regressors · L → 6 each · no kernel
method on L · 288 candidate evaluations written and the equal-budget assert
passes · `best_params.json` produced · `model_results.csv` 28 rows ·
`predictions_{C,R,Lc,Lr}.csv` written · resampling on the snapshot table raises.

---

## Milestone D — The required comparisons

*Goal: every graded comparison in brief §5.1 and §7.1 exists as a runnable script
with a CSV artefact.*

### PR 5 — `feat/md2-required-analyses`

| | |
|---|---|
| Files | `src/models/market_comparison.py`, `inplay_curves.py`, `resampling_study.py`, `kernel_scaling.py`, `src/viz/plot_calibration.py` |
| Base | PR 4 |
| Type | feat |

Five independent readers of `predictions_*.csv`. None imports another, so they
review in any order, but they land together because they are one deliverable.

**Market comparison.** Every Task C model re-scored on the **intersection** of the
test set with the odds-tagged matches, reported beside the de-vigged market on
those same rows, with the tagging coverage rate. Scoring the model on all test
matches and the market on the tagged subset would compare two different samples.
Re-asserts the de-vigging contract at the point of use.
→ 343/344 test matches tagged (**coverage 0.9971**). Market RPS 0.19665; best
model Kernel SVM 0.21656, P2/HS 0.21659. **0 of 7 models beat the market.** Honest
negative result; the brief says a well-analysed failure beats an unexplained
good-looking score.

**In-play curves.** Every Task L metric as a function of match minute, with the
**frozen pre-match prediction** as the reference curve, plus per-game-phase
calibration (0–15′ … 75–90′). Asserts the frozen curve is flat — it is one number
per match repeated across snapshots, so any slope would mean the reference leaked
match progress.
→ frozen reference flat at RPS 0.21641; in-play RPS falls 0.21825 → 0.02385.
**Crossover between minute 10 and 15** — before that the live stream adds nothing.
Over-confident by +0.05 through minute 75, then **under**-confident by −0.06 in
75–90′. Aggregate ECE hides both.

**Imbalance study.** The six arms of §7.1 — vanilla, SMOTE, BorderlineSMOTE,
ADASYN, `class_weight='balanced'`, and P1 (G-SMOTENC) — on identical splits,
Task C only, on aggregate *and* per-class metrics.
→ (mean over 4 models) P1 best RPS 0.21907 and best calibration (ECE 0.04519);
SMOTE best draw recall 0.23101 but worst RPS 0.22283. The textbook trade-off
measured rather than assumed.

**Kernel scaling.** The O(n²) claim demonstrated empirically on the real in-play
training matrix with a fitted power law, and asserted.
→ exact kernel ridge **t ~ n^2.24**, exact SVR **t ~ n^1.85**, Nyström
**t ~ n^0.39**. At the full 17,024 rows the exact Gram is 2,211 MB against 130 MB
peak for Nyström.

**Reliability diagrams.** Every probabilistic classifier on Tasks C and Lc, drawn
twice per model — raw and calibrated — so the calibration step's effect is visible
rather than compressed into one ECE number. Standalone by design: `src/` has no
package structure, so cross-directory imports do not resolve.

---

## Milestone E — Wiring and the defence pack

### PR 6 — `chore/pipeline-and-defence-pack`

| | |
|---|---|
| Files | `src/pipeline/run_all.py`, `docs/mid_defence_2.md`, `docs/md2_delivery_plan.md`, `TODO.md` |
| Base | PR 5 |
| Type | chore |

`run_all.py` previously stopped at Phase 4 and the visualisations. Now runs the
full chain: relational store → market baseline → **both paper test suites** →
leakage tests → both feature builders → tuning → sweep → imbalance study →
market comparison → in-play curves → kernel scaling → all visualisations.
`SKIP_TUNING=1` reuses `best_params.json` for fast iteration.

The defence brief: P2 presentation and filters, derivation sketch, model suite
table, tuning protocol, calibration strategy, the four leakage guarantees, and
worked answers to both design-consequence questions the brief names — *snapshots
every minute instead of every five*, and *SMOTE on the snapshot table*.

---

## Two loose ends to decide before landing

- **`forME/`** is a working audit of an older version of the project. It is
  untracked. Either move it to `docs/` as a historical note or leave it out of the
  repo — it should not land as-is, since it describes a tree that no longer exists.
- **`src/reports/` outputs** — the CSVs and HTML are generated artefacts. Confirm
  whether the submission wants them committed (the brief asks for reproducible
  code, and the checklist asks for the metric tables) or regenerated from
  `run_all.py`.

## Deferred to the final defence

SHAP (global beeswarm, local waterfalls for the 10 worst predictions, in-play
timeline), the Model 2 → Model 1 margin-to-probability conversion, the full
`REPORT.md` §1–§9 rewrite, and the FastAPI bonus. None is required at MD2.
