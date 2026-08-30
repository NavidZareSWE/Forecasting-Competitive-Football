# Final Defence — Speaker Script

Deck: `docs/final_defence_slides.html` · ~15:30 spoken + Q&A fills a ~30-minute slot.
Cut bracketed lines if running long. Backup slides are for Q&A only — never present them.

---

## Cue card — the whole talk at a glance

One line per slide. If you remember only this table, the talk still works.

| # | Slide | Say this one thing |
|---|---|---|
| 01 | Title | Two moments a forecast is worth money — pre-match and in-play. We built both. |
| 02 | Problem | Three models, RPS because outcomes are ordered, judged against the de-vigged market. |
| 03 | Data | 80 audited → 4 kept; odds joined on identity only — leakage control; 343/344 covered. |
| 04 | Build | Standalone idempotent stages; labels from match file only, re-derived + asserted; assertions ARE the spec. |
| 05 | Leakage | Four barriers in code; scoreline-rewrite test; running-max repair errs conservative only. |
| 06 | Features | One shared definition per quantity → in-play and form can't drift apart. 231 / 158 cols. |
| 07 | Papers | P1 fixes our real imbalance; P2 is one closed-form transform — checkable line by line; deviations declared. |
| 08 | Protocol | Equal budget asserted; calibration frozen on validation; match-clustered bootstrap + Holm. |
| 09 | Task C | Market wins: 0.19665 vs 0.21878. 0/21 survive Holm — leaderboard isn't a ranking. We close ~30% of the gap. |
| 10 | Task R | MAE 1.37 vs 1.49, corr 0.43; survivors only vs baseline. |
| 11 | In-play | THE finding: 0.219 → 0.023. Evidence = gap to frozen line, not the fall. Draw recall 0.18 → 0.41, same imbalance. |
| 12 | Calibration | Global calibrator over-confident early, under-confident late — measured, drives next steps. |
| 13 | SHAP | Goals appear as steps, form as flat offset — pipeline correct end to end. Right features ≠ good model. |
| 14 | P1 study | Recall ×4, RPS last — detection and probability quality are different objectives. Product decision. |
| 15 | Problems | Three defects, no symptoms — caught by cross-check and mechanism tests, never by metrics. |
| 16 | Hardening | Five more; each fix became a permanent guard. Dangerous bugs hand you plausible numbers. |
| 17 | Conclusions | Constraint is information, not capacity — tells the client where to spend. |
| — | Closing | In-play: yes, demonstrably. Pre-match: market wins and we know exactly why. |

**Q&A half of the slot:** likely deep-dives → leakage (slide 5), why-not-beat-market (slide 9 reframe), P2 equation (formula on slide 7), kernels/compute (backup slides).

---

## 01 · Title — 0:30

> A client wants a football forecast they can act on. There are two moments where that forecast is worth money — before kick-off, and during the match — and they're different problems built on different pipelines. This is 1,517 matches, four leagues, one season. I'll show you how we built both forecasts, and how we tested whether they're good enough to use.

## 02 · The problem — 0:50

> Three models. Model 1: at kick-off, a probability distribution over home, draw, away. Model 2: same input, the goal margin. Model 3: both answers re-computed every five minutes during the match — 19 snapshots, 28,823 rows.
>
> Headline metric is RPS, because the outcomes are ordered — calling a home win and getting a draw is a smaller mistake than getting an away win. Accuracy can't express that, log loss treats both as equal.
>
> And one standard: we don't mark our own homework. Every classification number is judged against the de-vigged bookmaker market on identical matches.

## 03 · The data — 0:45

> Competition selection was an audit, not intuition: a script examined all 80 competition-season entries and recorded a decision for each. 76 excluded — women's competitions and tournaments with no odds source, partial seasons that only contain Barcelona or Real Madrid. Four complete league seasons survived.
>
> The odds come from a second source with no shared key, so we join on identity only — competition, date, team names — never on scores. That's a leakage control, because an odds row carries the final result. Coverage: 343 of 344 test matches, 99.7%.

## 04 · The build — 1:00

> The whole system is a chain of standalone, idempotent scripts. Each stage reads the previous stage's CSV, and any stage can be re-run alone. A fresh clone works — missing StatsBomb JSON downloads and caches on first use.
>
> Two design decisions matter most. First, labels are derived once, from the match file only — never from odds, never from events — and a second store re-derives them independently and asserts agreement. Second, assertions are the specification: every builder ends with self-checks — unique keys, splits strictly ordered by date, the alias map a bijection, de-vigged probabilities summing to one.
>
> Prose can't fail. Assertions can.

## 05 · Leakage — 1:30

> The label is derivable from the same event stream the features come from, so leakage rules here are code, not prose. Four barriers.
>
> One: chronological split, 896 training, 277 validation, 344 test — every snapshot inherits its match's side.
>
> Two: pre-match form uses a one-match lag before the rolling window. The strong test: we rewrite a match's own scoreline to a different result and assert its feature row is byte-identical — that catches a lag missing in some rows, which no assertion on form values would find.
>
> Three: the time-t cut. About 26 events carry corrupted zero timestamps, so sorting by timestamp silently corrupts the prefix. We sort by period and index, and repair the minute column with a running maximum. A running maximum never decreases — so the repair can only make the cut more conservative, never less. Guarded on both sides.
>
> Four: imputers, scalers, resamplers fit on training rows only — the pipeline raises if you try to resample snapshots.

## 06 · Features — 0:45

> One shared module computes every per-team quantity once, and both pipelines consume it. The whole-match version feeds shift-then-roll into pre-match form; the prefix-up-to-t version feeds the in-play state. That guarantees the in-play value of a quantity and the rolling form of the same quantity mean the same thing — two definitions would drift apart silently.
>
> Result: 231 pre-match columns, 158 in-play columns joined to the frozen pre-match vector, with three views of the live state — totals, a momentum window, and per-minute rates so minute 10 compares with minute 85.

## 07 · The papers — 1:15

> Two papers, both reimplemented from scratch.
>
> P1, G-SMOTENC, targets the actual imperfection in our data: draws are the minority class and the table mixes nominal and continuous. It generates synthetic minority points inside a geometric region and takes the mode over neighbours for categoricals — never a fractional one-hot that's no real league.
>
> P2, Hierarchical Shrinkage, rewrites a fitted tree's node values without touching its shape: each split's increment is damped by how little data supported it, one parameter lambda.
>
> We initially ranked FIGS first and switched — HS is one closed-form transformation, no iterative optimiser, no unstated constants, which is what made the hand derivation and line-by-line fidelity tests possible. And our deviations from both papers are declared on the slide, not hidden.

## 08 · The protocol — 0:50

> Fairness is enforced, not described. Every model gets the same search budget — twelve candidates, three-fold CV inside training only, and the equality is asserted at the end of the run. Calibration is isotonic on validation with the base model frozen.
>
> Significance uses a match-clustered bootstrap with Holm correction — nineteen snapshots of one match are not nineteen observations; a row-level bootstrap would understate variance by the cluster size.
>
> [And the budget was enough: boosted models show large best-versus-median gaps, low-variance models were already near their best.]

## 09 · Task C — 1:30

> The pre-match result, honestly: the market wins. Market 0.19665, our best 0.21878, gap 0.022. Zero of seven models beat it — and zero of 21 pairwise comparisons survive Holm correction, so the leaderboard isn't even a ranking.
>
> Sharpest detail: the market never once makes the draw its top pick in 343 matches — draw recall exactly zero — and still posts the best probabilistic score. Ranking the minority class and stating probabilities are different skills.
>
> Why we trail: an information gap — team news, line-ups, money — plus a distribution shift of our own making: trained on a 43% home-win rate, tested on 49%, because the chronological split puts the season's end in test. We close about thirty per cent of the distance from ignorance to the market.

## 10 · Task R — 0:40

> Margin is modestly predictable: best MAE 1.37 goals against 1.49 constant — minus 8.3% — and correlation 0.43 where the baseline is zero by construction. Six comparisons survive Holm, all against the constant baseline — no learner separates from another.
>
> [And exact versus Nyström kernel ridge differ by 0.004 MAE, which is what licenses the approximation at snapshot scale — backup slide has the measurements.]

## 11 · In-play — 1:30

> This is the central finding. RPS falls from 0.219 at kick-off to 0.023 at minute 90 — minus 33% overall.
>
> But the falling curve alone isn't the evidence — any model shows that, because the outcome itself becomes certain. The evidence is the vertical gap to the dashed line: the same fixture, frozen at its pre-match forecast, on the same matches. That gap opens around minute 15 and widens monotonically.
>
> Draw recall goes 0.18 to 0.41 with the same class imbalance — proof that draws were an information problem, not a balance problem. A goalless match at minute 80 is a state a model can recognise; pre-match, no such state exists.
>
> And here the statistics finally bite: 6 of 15 and 9 of 15 comparisons survive Holm, against zero pre-match. Same machinery — what changed is the effect size.

## 12 · Calibration — 0:40

> One global calibrator, fitted on all snapshots pooled, and we measured what that costs: every learner is over-confident in the opening phase and under-confident at the end — plus 9 points early, minus 8 late. It averages across regimes.
>
> That's a measured mis-specification, and it's the direct justification for the per-phase calibrator on the next-steps slide — not a hunch.

## 13 · Explanation — 0:50

> TreeSHAP, exact for tree ensembles, additive, checkable. The timeline follows one match — chosen deterministically, not for a good story — through all 19 snapshots: goal-difference contributions jump exactly at the goals, pre-match form stays a flat offset all match. That's the pipeline behaving correctly end to end.
>
> Ablation agrees independently: deleting the current-score group costs an order of magnitude more than any other group.
>
> One caveat I'd raise before you do: SHAP shows the model uses the right features — it does not show the model is good. Pre-match proves you can use the right features and still not beat the base rate.

## 14 · The imbalance study — 0:45

> Six arms on the pre-match table. The two rankings disagree, and that's the finding: P1 delivers the largest draw-recall gain of any arm — recall quadruples on the random forest — and sits last on RPS.
>
> Exactly what the definitions predict: a proper scoring rule is minimised by the true probabilities, and rebalancing pushes the predicted draw rate above them. So it's a product decision: calibrated probabilities, leave it off; flagging likely draws, P1 wins.
>
> Snapshots are excluded by design — the pipeline raises — because interpolating across matches builds impossible game states.

## 15 · Problems — 1:15

> Three defects that changed results — none found by looking at a score.
>
> Corrupted timestamps: no symptom at all, the naive cut looked fine.
>
> Own goals: StatsBomb records them twice, we double-counted — the clue was errors exactly symmetric, never a wrong total; after the fix, 1,517 of 1,517 scorelines reconstruct from events, and that guard is permanent.
>
> TreeSHAP: our shrunk values lived outside the tree structure, so the explainer read the unshrunk forest — plausible attributions, no warning, entirely wrong.
>
> Two of the three produced no error and no visible symptom. They were caught by cross-checking an independent source of truth and by testing the mechanism. Aggregate metrics cannot find this class of defect.

## 16 · Hardening — 1:00

> Five quieter ones, each of which left a permanent guard behind.
>
> The worst: tuning passed a wrong keyword argument — it crashed in 1.4 seconds, which means every earlier number was library defaults; we re-ran the entire search, and tuning now checkpoints per task with automatic resume.
>
> My favourite: xG joined on team name instead of ID gave Marseille zero xG — caught because one away goal with zero xG is impossible, not just unlikely. The others are on the slide.
>
> The lesson repeats: the dangerous version of a bug runs to completion and hands you plausible numbers.

## 17 · Conclusions — 1:00

> What we know: live information beats every modelling choice, and that result is statistically supported; the pre-match leaderboard is not.
>
> What we tested and refuted — twice: thinning the snapshot grid 63% doesn't hurt, and the overfitting suspicion on 231 columns was specific and wrong.
>
> Limitations, plainly: one season, a real home-win drift induced by our own split, thin validation for isotonic.
>
> Next steps each tied to a measurement: squad-strength features from the already-built lineup store attack the measured market gap; a per-phase calibrator fixes what slide 12 measured; more seasons cure the drift. What I would not propose: more model. The constraint is information, not capacity — and that tells the client exactly where to spend.

## Closing — 0:15

> Does it solve the client's problem? In-play — yes, demonstrably. Pre-match — the market still wins, and we can say precisely why and what would close the gap. Every number on these slides is copied from a result CSV in the repository. Thank you — happy to take questions.

*(If demo prepped: "The models are also live behind a FastAPI service replaying a held-out match — I can show it.")*

---

## Q&A — backup slides

- **Kernel scaling** → "measured, not assumed": exact Gram at 17k rows is 2,211 MB; Nyström is 280× faster for 0.004 MAE.
- **Compute ledger** → "since no pre-match learner is statistically separable, compute is a legitimate tie-breaker — and it favours xgboost decisively."

## Five numbers to know cold

| number | meaning |
|---|---|
| 0.19665 | market RPS — the target |
| 0.21862 → 0.02266 | in-play arc, kick-off → minute 90 |
| +0.02213 | gap to market, best model |
| 0.4137 | in-play draw recall (from 0.1772) |
| 896 / 277 / 344 | train / validation / test |

## Demo — run before the talk

```bash
source /Users/parsabordbar/venev/ml/bin/activate
python src/service/app.py                        # terminal 1 — wait for uvicorn "running"
open http://127.0.0.1:5500/                      # dashboard tab, keep in background
python src/service/replay_driver.py 3825848 0.5  # terminal 2 — snappy replay for Q&A
```
