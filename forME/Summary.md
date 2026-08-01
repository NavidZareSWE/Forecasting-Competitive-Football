# Audit Summary

## Verdict
**~72% of Mid-Defence 2 is complete.** The Defence files are *necessary but not sufficient* — dropping them into `src/models/` and running produces crashes on any properly-provisioned machine.

## The two archives
- `For_Defence.zip` contains 5 files (`modeling_common.py`, `train.py`, `run_models.py`, `train_prematch.py`, `train_inplay.py`) with **zero overlap** with `code.zip`. All belong in `src/models/` next to the existing `model_zoo.py`.
- `code.zip` already has a complete, verified Phase 1–5 foundation plus the from-scratch G-SMOTENC.

## Code defects confirmed by execution

| # | Defect | Effect |
|---|---|---|
| 1 | `modeling_common.py:22` — `PROJECT.parent / "papers"` should be `PROJECT / "papers"` | `ModuleNotFoundError: No module named 'g_smotenc'` — **every P1 run fails** |
| 2 | `XGBClassifier` rejects string labels `H`/`D`/`A` | `ValueError: Expected: [0 1 2], got ['A' 'D' 'H']` — aborts Task C. Currently masked because the guarded import silently returns `{}` when xgboost isn't installed |
| 3 | `test_g_smotenc.py` calls `alpha_trunc=` / `alpha_def=`; implementation uses `truncation` / `deformation` | `TypeError` kills the runner at test 2 of 8 (reports "1/8") |
| 4 | `.gitignore` line 47 contains `test*.py` | **Both test files are untracked** — `git ls-files` returns 26 files, neither test among them. Your graded leakage proof doesn't exist in a fresh clone |
| 5 | **(found after the main report)** `KernelRidge` on Task Lr | `MemoryError` — 17,024 training rows needs a 2.32 GB dense kernel plus Cholesky workspace. SVR is fine by contrast (~14 s extrapolated) |

After fixing #1, P1 balanced correctly: 896 → 1,164 rows, `{'A': 388, 'D': 388, 'H': 388}`; in-play 17,024 → 22,116.

## Missing documentation (nothing exists in either archive)
- **Hyperparameter tuning protocol** — no tuning of any kind. All hyperparameters hardcoded; validation is used only for calibration. Evidence it matters: LightGBM scored `rps=0.28461, ece=0.33418` — *worse than Dummy* (`0.22812`).
- **Calibration strategy write-up** — the code is sound (isotonic → sigmoid → fallback, fit on validation, applied to test; all 12 runs reported `cal=isotonic`), it just has no prose.
- **Consolidated snapshot-leakage document.** `REPORT.md` stops at §3.

## Important, non-blocking
- **`REPORT.md` §3 contradicts the code** — says events are ordered by `(period, timestamp, index)`, but `build_inplay_features.py:93` sorts by `index` alone, deliberately, to survive corrupted timestamps. The code is right; fix the prose.
- **`market_baseline.csv` is built but never consumed** — 99.9% coverage, de-vigged probabilities ready, and no model is scored against it despite that being the project's framing.
- `run_all.py` stops after visualisations (no Phase 5/6); `config.py` is dead code with a wrong `REPORTS_DIR`; `README.md` marks completed work as `⬜`; `feature_common.py` is listed as required in the explainer but doesn't exist.

## Leakage
Correctly prevented, verified on four independent barriers: strict chronological date splits (train ends 2016-01-31, test starts 2016-03-19, zero overlap), `.shift(1)` before `.rolling(5)`, the effective-minute prefix cut with two-sided assertions, and train-only fitting of imputer/scaler/P1 with calibration on validation. The only real risk is procedural — the proof isn't in version control.

## Reminder
**P2 paper still needs selecting.** Tracked as an open, deliberately deferred decision, not as missed work.

---

Say the word and I'll turn this into the `.md` file you asked for — I'd suggest a `FIXES.md` with the five code defects as a checklist and the documentation gaps as a separate section, so it doubles as your working list between now and the defence.