# TODO — gap list against the course brief

Derived from `docs/course-brief/Final_Project_Machine_Learning.pdf`, the `forME/`
audit, and a code review of the six-PR stack in `docs/md2_delivery_plan.md`.

Grading weights, for prioritisation:

| Component | Weight | Status |
|---|---|---|
| Data integration & feature pipelines | 30% | done, features still thin |
| Paper reimplementations (P1 + P2) | 20% | P1 done, P2 done pending TA sign-off |
| Three models & comparative analysis | 20% | done |
| Written report | 15% | stops at §3 of 13 |
| Defences (2 mid + final) | 15% | MD2 ready, final blocked on SHAP |

---

## P0 — blocking

- [ ] **P2 TA sign-off.** Hierarchical Shrinkage (ICML 2022, CORE A\*) is selected,
      reimplemented and derived, but the brief requires sign-off *before*
      implementation. Present at Mid Defence 2; FIGS is the declared fallback.
      Evidence: `docs/p2_paper_selection.md`, `docs/appendix_a_hierarchical_shrinkage.md`.

## P1 — mandatory brief requirements not implemented

### SHAP (brief §7.3, §9, final defence)
- [ ] No SHAP code anywhere. Required: global beeswarm per model/task, local
      force/waterfall plots for the worst predictions, and an in-play SHAP timeline
      across one full match. This is the final-defence centrepiece.
- [ ] 10-worst-prediction post-mortem per task, explained with SHAP.

### Model 2 → Model 1 (brief §1)
- [ ] Required analysis of whether the regressed margin can be converted into useful
      H/D/A probabilities. Not attempted.

## P2 — feature pipeline depth (30% component)

- [ ] Pre-match features are thin: rolling `gf/ga/xgf/xga/points/win` + `rest_days` +
      `played_prior`. Brief §2.2 and flowchart 5A additionally name pressure
      intensity, passing volume and completion by zone, carries into the final third,
      set-piece counts, possession share, defensive actions, head-to-head, and venue.
- [ ] In-play features carry only recent xG as a rate. Brief 5B asks for event
      *counts and rates* in the recent window plus momentum indicators.
- [ ] Document the early-season form caveat: `min_periods=1` means a team's second
      match carries a one-match "rolling" average.

## P3 — report and documentation (15% component)

- [ ] `REPORT.md` stops at §3. The brief §8 structure needs §1-§9 plus Appendix A
      (P2 hand derivation, already written) and Appendix B (reproducibility:
      requirements, seeds, git log).
- [ ] `REPORT.md` §3 contradicts the code on event ordering — it claims
      `(period, timestamp, index)`; `build_inplay_features.py` sorts by
      `(period, index)` deliberately, because ~26 events carry corrupted `00:00:00`
      timestamps. The code is right; fix the prose.
- [ ] Consolidated leakage document — the graded proof. Four barriers hold
      (chronological splits, `.shift(1)` before `.rolling(5)`, the effective-minute
      prefix cut with two-sided assertions, train-only transform fitting); the
      summary in `docs/mid_defence_2.md` needs to move into the report.
- [ ] Competition selection justification from `src/audit/competition_audit.py` needs
      to land in report §2.
- [ ] `README.md` is stale — references `Extras/` and `src/scripts/`, neither of which
      exists, and marks completed work as pending.
- [ ] P1 fidelity table (implementation vs paper) and the honest "did the synthetic
      data actually help" result — `resampling_study.csv` now has the numbers.

## P4 — housekeeping

- [ ] `config.py` is populated with correct paths and has **zero importers**; every
      script re-derives its own `PROJECT`. Wire it in or delete it.
- [ ] `origin/feat/match-featrues` and `origin/feat/model-zoo-classifiers-regressors`
      end in deletion commits that would remove `g_smotenc.py` and both feature
      builders. Their content is already in `main`. Delete the remote branches so
      nobody merges a tip.
- [ ] `.gitignore` no longer contains `test*.py`; both test files are tracked. Do not
      let it come back.
- [ ] Decide whether `forME/` and the generated `src/reports/` artefacts belong in
      the repo — see `docs/md2_delivery_plan.md`.
- [ ] `prepare_matrices` re-fits the imputer, scaler and sampler once per model.
      Correct, but on the snapshot table it is the dominant cost.

## P5 — bonus (§11, optional credit)

- [ ] FastAPI service wrapping Model 3 (plus Models 1-2 for the pre-kickoff state),
      under 200 ms end to end, reusing the training feature code rather than
      reimplementing it. Report p50 / p95 / p99 under repeated calls.
- [ ] Live dashboard replaying a held-out match: outcome probabilities over match
      time, expected final margin, goal/card markers, top SHAP attributions.

---

## Done and verified

- Phases 1-4 relational store, labels, cleaning, temporal splits, market baseline.
- Odds join on pre-match identity only, bijective alias map, 99.9% coverage.
- G-SMOTENC (P1) from scratch — `python src/papers/test_g_smotenc.py` → 8/8.
- Hierarchical Shrinkage (P2) from scratch, with the Appendix A derivation —
  `python src/papers/test_hierarchical_shrinkage.py` → 8/8.
- In-play time-t cut with two-sided prefix assertions —
  `python src/features/test_inplay_cut.py` → 7/7.
- Task-aware model zoo, training harness, calibration, four task runners.
- Equal-budget random search with grouped CV — 288 evaluations, assert passes.
- Market comparison on the odds-tagged test subset — coverage 0.9971, 0 of 7
  models beat the de-vigged market.
- Task L metric-vs-minute curves against a frozen pre-match reference, and
  per-phase calibration.
- Six-arm imbalance study with per-class metrics.
- Kernel scaling exponents measured and asserted.
- Reliability diagrams before and after calibration.
- `run_all.py` runs the whole chain, Phase 1 through the analyses.
- `libomp` installed; xgboost and lightgbm both load, zoo returns the full set.
