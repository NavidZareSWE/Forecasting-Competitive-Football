# Session handover

Written at the end of a long chat. Covers only what a new session cannot get
from the project instructions or memory: current state, what changed, what to
wait for.

---

## Waiting on

The user is re-running the full pipeline and will send back
`console-outputs.zip` (one log per step, `_summary.txt` with exit codes and
durations, `00_environment.txt`, and copies of the result CSVs).

Read in this order:

1. `best_params.json` + the `20_tuning` log — **has never completed**
2. `model_results.csv`
3. `significance_bootstrap.csv` — decides whether pre-match models can be
   claimed to beat the baseline
4. `ablation.csv` — first run on the widened features
5. Any traceback, in full

---

## State that differs from memory

**P2 is decided and built.** Hierarchical Shrinkage (Agarwal et al., ICML 2022,
PMLR 162:111-135). Implemented, derived in Appendix A, 9/9 tests.
**TA sign-off still outstanding.**

**Feature tables were widened mid-session** at the user's request:

| Table | Was | Now |
|---|---|---|
| Pre-match | 22 model columns | **231** |
| In-play | 11 model columns | **153** |

New shared module `src/features/event_aggregates.py` computes 29 per-team
quantities from any frame of events. The pre-match builder passes a whole match
and rolls it into prior form; the in-play builder passes the prefix up to minute
*t*. One definition per quantity, so the two tables cannot drift apart. Also
added venue-split form, head-to-head, and for in-play three views of every
quantity: match total, recent 10-minute window, per-minute rate.

**Deliberately not built:** pass completion and carries *into* the final third.
Both need fields absent from the current `clean_events.csv`.
`build_event_store.py` now extracts `pass_outcome` and `carry_end_x` so a future
re-extraction gets them. Absent, not proxied.

**New entry point.** `run_pipeline.py` at the project root runs everything —
8 stages, 28 steps, tests through report — with `capture_console.py` logging
each step and `check_progress.py` as a read-only progress checker.
`--from <stage>` resumes.

**Environment:** Windows 11, Python 3.14.0, pandas 2.3.3, scikit-learn 1.8.0.
Requirements are now lower bounds, not pins, with `requirements.lock.txt`
written per run for reproducibility.

---

## Bugs fixed this session

**Blocking**

1. `tuning.py` called `prepare_matrices(..., use_p1=False)`; the parameter is
   `resampling=`. Crashed on the first task in 1.4s, so `best_params.json` was
   never written and **every result to date is on library defaults**.
2. Tuning wrote `best_params.json` only after all four tasks finished. The
   user's machine stopped several times, discarding hours each time. Now saves
   after each task and resumes. `TUNE_RESUME=0`, `TUNE_TASKS=C,R` override.

**Correctness**

3. Own-goal attribution was backwards. StatsBomb records `Own Goal For` on the
   team that **benefits**; the code also credited the opponent's. Scoreline
   agreement 1391/1517 -> **1517/1517**. Permanent guard in the builder.
4. A seed-repetition paired t-test was reported as a significance test. Only the
   random state varies, so a deterministic model has zero spread and the test
   returns p~0 for any gap; it declared `dummy` different from `gbm` while the
   correct test found nothing. Renamed to stability diagnostics. The
   **match-clustered bootstrap is the only test licensing a superiority claim**.
5. Snapshot-frequency ablation thinned validation as well as training, so arms
   were scored on different evaluation rows. Training only now.
6. Nystroem requested more landmarks than fold rows; sklearn silently reduced
   the count, so candidates were **scored under a different configuration than
   the one recorded**. `ClampedNystroem` clamps to fold size.
7. TreeSHAP would have explained the *unshrunk* forest: P2 stores shrunk values
   outside the tree structure. `HSForest.shap_base_estimator()` writes them into
   `tree_.value`. Verified to 1.1e-16; SHAP additive to 1.4e-15.
8. `REPORT.md` event-ordering: §1 was wrong, §2 was right. Worth knowing that
   the first attempted fix would have introduced a false claim into the correct
   sentence, caught only by reading the source. `clean_store.py` sorts by
   `(period, event_time, index)`; `build_inplay_features.py` deliberately
   re-sorts by `(period, index)` because ~26 events carry corrupted `00:00:00`
   timestamps.

**Environment and usability**

9. Pinned requirements broke installation on Python 3.14 — pandas 2.3.2 has no
   cp314 wheel, so pip attempted a source build and failed for want of MSVC.
10. 148 pandas FutureWarnings from `.where(notna(), False)`, which still
    downcasts on pandas 2.3.3. Missed twice because the assistant's container
    runs pandas 3.0.2 where it does not fire. Now fills through numpy.
11. 5,229 LightGBM/sklearn UserWarnings — lightgbm 4.6 against sklearn 1.8.
    Floor raised to 4.7.
12. Step logs were written only after a step finished, so a long step looked
    identical to a hang. Output now streams to `console-outputs/_running.txt`.
13. Runner preflight used the repo root, but pipeline scripts root their paths
    at `src/`; it also demanded the store that the `data` stage creates, making
    a from-scratch run impossible. Odds folder now resolves case-insensitively.
14. SHAP computed on the full test split (6,536 x 386 x 5 models). Global view
    now uses a 2,000-row sample; worst-10 and the match timeline stay exact.
    Task Lc: hours -> 36s.

---

## Measured results

**All from untuned defaults.** Provisional until tuning lands.

### Pre-match (Task C) — widening helped

| Model | Narrow | Wide |
|---|---|---|
| dummy | 0.22812 | 0.22812 |
| random_forest | 0.22323 | **0.21640** |
| p2_hier_shrinkage | 0.22212 | 0.21676 |
| gbm | 0.22296 | 0.21941 |
| lightgbm | 0.22623 | 0.22256 |
| kernel_svm | 0.22716 | 0.22441 |
| xgboost | 0.22104 | 0.22464 |

Dummy identical to 5dp confirms the test set is unchanged. Gap over baseline
widened ~65%.

**The verdict did not change: Task C is still 0 of 21 significant** after Holm.
dummy vs random_forest: difference 0.0117, CI **[-0.0066, +0.0280]**. 344 test
matches cannot resolve an effect that small.

### In-play (Task Lc) — widening did *not* help

random_forest 0.15221 -> 0.15329; p2 0.15191 -> 0.15332; xgboost 0.15310 ->
0.15899. Every model slightly worse with 142 extra columns: goal difference
already carried the signal. **Lc remains strongly significant, 11 of 15**;
dummy vs random_forest CI [0.060, 0.089].

Headline: widening helped where information was scarce and hurt where it was
already abundant.

### Other

- **Margin -> probability** on wide features: ordinal conversion now beats the
  direct classifier on **both** metrics — RPS 0.20780 vs 0.21284, ECE 0.04490 vs
  0.05455. Previously it won only on calibration.
- **P1**: draw recall 0.205 -> 0.449 (RF), 0.295 -> 0.410 (XGB), best draw F1 of
  six arms; RPS effect small and mixed.
- **Feature-group ablation** (narrow only, needs re-running): only `form_xg`
  carried signal, +0.0158 RPS when dropped.
- **Snapshot density**: 10 per match ~= 19 per match on a fixed evaluation set.
- Tests: aggregates 12/12, prematch 9/9, inplay cut 7/7, G-SMOTENC 8/8, HS 9/9.

---

## Risks in the pending run

- `run_models.py` task Lr was killed in the assistant's container, likely
  memory. Resume with `TASKS=Lr python src/models/run_models.py`. Rows carried
  from an earlier run are flagged `carried_from_previous_run=True` — **one clean
  full run before reporting anything**, or feature versions get mixed.
- Tuning is slow: `[Lc] gbm` alone took 13,878s (3.9 h) in the user's last run.
- Report is 16 body pages of 25 allowed, and regenerates from the CSVs, so it
  updates itself once results land. Missing CSVs print a marker naming the
  producing script — those must not be papered over.
