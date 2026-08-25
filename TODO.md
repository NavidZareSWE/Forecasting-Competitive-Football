# TODO — gap list against the course brief

Derived from `docs/course-brief/Final_Project_Machine_Learning.pdf` and a code
review of the current `main`.

| Component | Weight | Status |
|---|---|---|
| Data integration & feature pipelines | 30% | done; feature set still thin (see P2) |
| Paper reimplementations (P1 + P2) | 20% | both done and tested; P2 needs TA sign-off |
| Three models & comparative analysis | 20% | done |
| Written report | 15% | builds from measured results; regenerate after tuning |
| Defences (2 mid + final) | 15% | SHAP timeline works; final defence ready once tuned |

---

## P0 — blocking

- [ ] **Re-run the pipeline on a multi-core machine.** `tuning.py` was crashing
      on a stale keyword and has now been fixed, but `best_params.json` has
      never been produced. Every number currently in `model_results.csv`,
      `market_comparison.csv` and `resampling_study.csv` was produced with
      `tuned=False`, i.e. library defaults. Order:
      `python src/models/tuning.py` then `python src/pipeline/run_all.py`.
      Delete `best_params.json` first if the feature space changed.
- [ ] **P2 TA sign-off.** Hierarchical Shrinkage is selected, reimplemented,
      derived and tested, but the brief requires sign-off *before*
      implementation. FIGS remains the declared fallback.

## P1 — findings that need a decision, not code

- [ ] **The pre-match models are not distinguishable from the prior baseline.**
      The match-clustered bootstrap returns 0 of 21 significant pairs on Task C
      after Holm correction, including dummy versus the best learner. The
      in-play tasks are strongly significant (11 of 15 on Lc, 7 of 15 on Lr).
      Decide whether to present this as the headline honest result or to widen
      the feature set first.
- [ ] **Denser snapshots buy almost nothing.** Training on 10 snapshots per
      match scores the same as 19 on a fixed validation set (0.13394 vs
      0.13401); 7 is slightly worse. The extra rows are highly correlated.
- [ ] **P1 does what it claims, but not to RPS.** G-SMOTENC more than doubles
      draw recall (0.205 -> 0.449 random forest, 0.295 -> 0.410 xgboost) and
      gives the best draw F1 of all six arms, while moving RPS marginally the
      wrong way on one learner and the right way on the other. Report both.

## P2 — feature pipeline depth (30% component)

- [ ] Pre-match features remain the thin set: rolling `gf/ga/xgf/xga/points/win`
      plus `rest_days` and `played_prior`, 22 model columns. The brief also
      names pressure intensity, passing volume and completion by zone, carries
      into the final third, set-piece counts, possession share, defensive
      actions, head-to-head and venue. The ablation shows only `form_xg`
      carries signal, which is partly a statement about how narrow the set is.
- [ ] In-play features carry 11 model columns. The brief asks for event counts
      and rates in the recent window plus momentum indicators.
- [ ] Note: `build_team_match_aggregates.py` is referenced by the console
      capture harness but does not exist in the repository or its git history.

## P3 — report and documentation

- [x] `REPORT.md` §1/§2 event-ordering contradiction corrected against the code.
- [ ] Fold the remaining `REPORT.md` prose into `src/report/build_report.py`, or
      retire `REPORT.md` in favour of the generated PDF.
- [ ] `README.md` is stale in places.

## P4 — housekeeping

- [ ] `config.py` still has zero importers; every script re-derives `PROJECT`.
      Wire it in or delete it.
- [ ] Delete the remote branches ending in deletion commits.
- [ ] Decide whether generated `src/reports/` artefacts belong in the repo.

## P5 — bonus (§11)

- [x] FastAPI service reusing the training feature code, with p50/p95/p99.
- [ ] Live dashboard. Frontend not chosen; Vite + React or Next.js.

---

## Done and verified in this pass

- `tuning.py` stale-keyword crash fixed at source.
- `prepare_matrices` returns `feature_names`, the fitted `transform` and the
  column lists, so SHAP, the ablation and the API stop duplicating assembly.
- `HSForest.shap_base_estimator()` writes shrunk values into `tree_.value`;
  verified to reproduce the model to 1.1e-16, TreeSHAP additive to 1.4e-15.
- `test_prematch_features.py` 9/9, `test_hierarchical_shrinkage.py` 9/9.
- SHAP beeswarms, worst-prediction waterfalls and the full-match in-play SHAP
  timeline all render; the timeline attributes both probability jumps in match
  265944 to `inplay_goal_diff`.
- Margin-to-probability: ordinal link RPS 0.21484 against the direct classifier
  0.21440, with materially better ECE (0.054 vs 0.072).
- Match-clustered bootstrap, seed repetition, compute profile, all four
  ablation axes and the report build all execute.
- Two methodology defects caught during validation and fixed: the seed-wise
  t-test was masquerading as a significance test, and the snapshot-frequency
  ablation was scoring each arm on a different evaluation set.
