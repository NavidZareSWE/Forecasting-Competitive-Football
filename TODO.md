# TODO — remaining work

Verified 2026-08-24 against the working tree, `git log`, and
`docs/course-brief/Final_Project_Machine_Learning.pdf`. Updated the same day
after the feature-depth / ablation / service work landed and the full
pipeline re-ran end to end on the widened tables.

Grading weights, for prioritisation:

| Component | Weight | Status |
|---|---|---|
| Data integration & feature pipelines | 30% | done — 71 pre-match / 26 in-play model columns, brief §2.2/5A/5B covered |
| Paper reimplementations (P1 + P2) | 20% | done; P2 pending TA sign-off |
| Three models & comparative analysis | 20% | done, incl. ablation + P1 comparison |
| Written report | 15% | §1–§11 + App A/B built (PDF + DOCX), zero "Result not available" markers |
| Defences (2 mid + final) | 15% | MD2 pack ready; final-defence SHAP + live service/dashboard all runnable |

---

## P0 — process, not code

- [ ] **P2 TA sign-off** for Hierarchical Shrinkage at Mid Defence 2. FIGS is the
      declared fallback. Only remaining blocker anywhere.

## P1 — remaining polish

- [ ] Rehearse the final-defence live case: `python src/service/app.py`, open
      `http://127.0.0.1:8100/`, replay a TA-selected held-out match, narrate the
      SHAP panel. `replay_driver.py` is the terminal fallback.
- [ ] Optional: regenerate `console-outputs.zip` after any further re-run so the
      shipped logs match the shipped numbers.

## P3 — housekeeping (unchanged)

- [ ] **Two orchestrators.** Root `run_pipeline.py` (preflight, resume,
      env-snapshot, logging) is the superset and ran the full chain;
      `src/pipeline/run_all.py` now includes ablation but still misses
      `build_inplay_features`, `competition_audit`, `compute_profile`,
      `margin_to_probability`, `significance`, `shap_analysis`, `build_report`.
      Bless one as canonical (or make `run_all.py` delegate).
- [ ] `config.py` is populated and correct but still has **zero importers**.
      Wire it in or delete it.
- [ ] Delete stale remote branches whose tips end in deletion commits —
      `origin/feat/match-featrues`, `origin/feat/model-zoo-classifiers-regressors`
      (content already in `main`); prune the other merged branches too.
- [ ] Decide whether `forME/` and generated `src/reports/` artefacts stay in
      the repo.
- [ ] `CLAUDE.md` "Current state" section is stale (predates SHAP, the report
      generator, `run_pipeline.py`, the service, and the widened features).
- [ ] `REPORT.md` at the repo root still stops at §3; superseded by the
      generated `src/reports/final_report.pdf`. Either delete it or leave a
      pointer to the generated report.

---

## Done and verified (2026-08-24 session)

- **Feature depth (brief §2.2 / 5A / 5B).** `load_events` extended with
  `pass_outcome`, `pass_type`, `end_x`, `end_y`; `clean_events.csv`
  regenerated. Pre-match: 22 → 71 model columns (shots, SOT, pressures,
  possession share by pass share, passing volume by pitch third + completion,
  carries into the final third, corners/free kicks/throw-ins, defensive
  actions, head-to-head expanding means) — all via the same
  shift-then-roll; `test_prematch_features.py` 12/12 incl. extended barrier
  (perturbing a match's own result *or events* moves nothing). In-play:
  12 → 26 columns (prefix + recent-window counts, per-minute rates, xG
  momentum, recent event share); `test_inplay_cut.py` 11/11, prefix
  assertions cover every new feature.
- **Full re-tune + sweep on the widened tables.** `TUNE_RESUME=0
  run_pipeline.py --from tuning`: 16/17 steps green first pass (DOCX needed
  `npm install docx`), tuning 41 min, all analyses regenerated. Market still
  unbeaten (RPS 0.197 market vs 0.221 best model) — the honest-failure
  narrative holds.
- **Ablation** — `src/models/ablation.py`: feature-group axis (partition
  asserted, 6 groups pre-match / 9 in-play), snapshot-frequency axis
  (training thinned to every 5/15/45, validation at full density), plus
  `p1_comparison.csv` (P1 raises draw recall on all six learners, RPS
  slightly worse on all six). All validation-only. Wired into both
  orchestrators; report §8 renders from it.
- **Bonus service (brief §11)** — `src/service/app.py`: FastAPI, models +
  calibrators + TreeSHAP explainer fitted once at startup, test-split-only
  serving, `/predict`, `/replay/{id}`, `/matches`, `/health`, dashboard at
  `/`. `measure_latency.py`: in-process p50/p95/p99 (predict p99 = 67 ms,
  budget 200), parity assert vs the sweep's stored predictions — passes.
  `dashboard.html` (probability lines, expected margin, goal/card markers,
  top-SHAP bars, self-paced replay) + `replay_driver.py`.
- **Report** — `p1_comparison.csv` + `competition_audit.csv` gaps filled
  (competitions.json + 76 match files fetched to the cache), §3 prose
  rewritten to describe the implemented feature set exactly, §8 column-count
  and snapshot-frequency prose corrected to the measured result. PDF + DOCX
  rebuilt: zero "Result not available" markers.
- **Bug fix** — `run_pipeline.py` preflight `needs_store` set-in-set
  `TypeError`.
- Everything from before: relational store, odds join (0.9971), G-SMOTENC
  8/8, Hierarchical Shrinkage 8/8, model zoo, equal-budget search, market
  comparison, in-play curves, six-arm imbalance study, kernel scaling,
  reliability diagrams, SHAP suite, margin→probability, significance,
  compute profile.
