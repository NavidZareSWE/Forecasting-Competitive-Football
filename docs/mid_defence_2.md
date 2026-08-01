# Mid Defence 2 — P2 Paper & Predictive Modeling

Defence brief. Section 10 of the course brief asks for two things at this
milestone: the P2 pick (presented, filtered, core maths, hand-derivation sketch —
**sign-off happens here**) and the modeling approach (suite configuration, tuning
budget and protocol, calibration strategy, and how Model 3 trains on the snapshot
table without leaking). Design-consequence questions are answered in §5.

---

## 1. P2 pick — Hierarchical Shrinkage

**Citation.** Agarwal, Tan, Ronen, Singh, Yu. *Hierarchical Shrinkage: Improving
the accuracy and interpretability of tree-based models.* ICML 2022,
PMLR 162:111–135. <https://proceedings.mlr.press/v162/agarwal22b.html>

**The four hard filters.**

| Filter | Evidence |
|---|---|
| Published 2022 or later | ICML 2022, PMLR v162 — publication date, not a preprint date |
| Q1 venue / CORE A\* or A | ICML is CORE **A\*** |
| Directly applicable to this dataset and tasks | Regularises CART/RF, which is exactly what both feature tables are — mixed-type tabular with small effective sample size |
| Not deep learning | No neural component anywhere; it is a closed-form transform of tree leaf values |

**Why this paper and not the alternatives.** P2 must serve *both* classification
and regression so one method covers Models 1, 2 and 3. HS is defined on the node
mean μ(t), which is a scalar for regression and a class-distribution vector for
classification — identical algebra, no per-task adaptation hack. Two candidates
that also cleared date and venue were rejected on the from-scratch requirement:
Distributional Random Forests (JMLR 2022) and GPBoost (JMLR 2022 / TPAMI 2023) are
both large C++/Rcpp codebases whose fast approximations are not fully specified in
the paper text. NGBoost and PGBM were rejected on the date filter (2020, 2021).

**The one-sentence method.** Grow a tree exactly as CART does, then replace each
leaf's prediction by a weighted average of the means of *every node on its
root-to-leaf path*, damping each split's contribution in proportion to how little
data supported it.

**Core equation.**

$$\hat f_{\mathrm{HS}}(x) = \mu(t_0) + \sum_{l=1}^{L}
\frac{\mu(t_l) - \mu(t_{l-1})}{1 + \lambda / N(t_{l-1})}$$

with $t_0 \supset \dots \supset t_L$ the root-to-leaf path, $\mu(t)$ the training
mean in node $t$, $N(t)$ its sample count, and one parameter $\lambda \ge 0$.
Note the damping uses the **parent's** count: a split taken off a large node is
trusted, the same split taken off a near-empty node is nearly ignored.

**Derivation sketch (full version: `docs/appendix_a_hierarchical_shrinkage.md`).**
Two results to be able to put on the board.

1. *It is a convex combination.* Set $w_l = N(t_{l-1})/(N(t_{l-1})+\lambda)$,
   $w_{L+1}=0$. Re-indexing the sum gives
   $\hat f_{\mathrm{HS}}(x) = \sum_{l=0}^{L} c_l \mu(t_l)$ with $c_0 = 1-w_1$,
   $c_l = w_l - w_{l+1}$, $c_L = w_L$. These telescope to 1, and are non-negative
   because the path is nested ($N(t_{l-1}) \ge N(t_l)$) and $u\mapsto u/(u+\lambda)$
   increases. **Consequence:** for classification the output is automatically a
   valid probability vector for any $\lambda$ — no clipping. $\lambda=0$ gives
   plain CART; $\lambda\to\infty$ gives the root mean, i.e. the Dummy prior.
2. *It is ridge in the increment basis.* Writing the tree as
   $f = \beta_0 + \sum_{t\neq t_0}\beta_t\mathbf 1\{x\in t\}$ makes the coefficients
   *the increments*. Solving, for one node $t$ with parent $p$,
   $\min_\delta \sum_{i\in p}(y_i - \mu(p) - \delta\mathbf 1\{x_i\in t\})^2 + \lambda_t\delta^2$
   gives $\hat\delta = (\mu(t)-\mu(p))\,N(t)/(N(t)+\lambda_t)$, which reproduces the
   boxed equation exactly when $\lambda_t = \lambda\,N(t)/N(p)$. So HS solves
   $$\min_\beta \|y - f\|^2 + \lambda \sum_{t \neq t_0} \frac{N(t)}{N(\mathrm{parent}(t))}\beta_t^2,$$
   penalising each split by the fraction of its parent it kept.

**Implementation.** `src/papers/hierarchical_shrinkage.py`, written against the raw
`tree_` arrays — the shrinkage recursion, the leaf descent and the $\lambda$
selection are all ours; sklearn only grows the underlying CART, exactly as in the
paper. `src/papers/test_hierarchical_shrinkage.py`, **8/8 passing**, including an
independent per-path transcription of the paper equation checked against the fast
recursion, the $\lambda=0$ and $\lambda\to\infty$ limits, simplex closure, and
agreement of our hand-written descent with `tree_.apply`.

**Honest limitation to volunteer.** HS cannot repair a badly grown tree. It only
redistributes weight along paths CART already chose; if the split criterion picked
the wrong variable, HS averages over a wrong path. It is a variance intervention,
not a bias one.

---

## 2. Model suite configuration, per deliverable

Task support follows the brief's model table and is declared in one place,
`model_zoo.TASK_SUPPORT`, so the sweep cannot silently run a model where it was
not specified.

| Model | C (Model 1) | R (Model 2) | L (Model 3) | Role |
|---|:--:|:--:|:--:|---|
| Dummy / prior | ✓ | ✓ | ✓ | sanity floor; for L also the *frozen pre-match* reference |
| Kernel SVM (SVC) | ✓ | — | — | first kernel method; drives the O(n²) discussion |
| Kernel SVR | — | ✓ | — | regression counterpart |
| Kernel Ridge (exact) | — | ✓ | — | exact Gram, subsampled at 8,000 rows and **flagged** as such |
| Kernel Ridge (Nyström) | — | ✓ | — | approximate kernel, the exact-vs-approximate comparison |
| Random Forest | ✓ | ✓ | ✓ | low-tuning variance-reduction reference |
| GBM (Hist) | ✓ | ✓ | ✓ | textbook boosting; the speed gap to XGB/LGBM is the compute story |
| XGBoost | ✓ | ✓ | ✓ | regularised boosting |
| LightGBM | ✓ | ✓ | ✓ | histogram boosting; the 28k-row snapshot table is where it should win |
| **P2 — Hierarchical Shrinkage** | ✓ | ✓ | ✓ | paper method, one learner across all three |
| De-vigged market | ✓ | — | — | **the** baseline, §4 |

Kernel methods are deliberately **not** run on Task L. That is not an omission: an
exact Gram on 17k in-play training rows is 2.3 GB and O(n³) to solve, and the cost
is quantified properly in the kernel-scaling deliverable (`kernel_scaling.py`)
rather than paid inside the main sweep.

**Answer if asked "why is your P2 a forest and not a single tree?"** Both exist
(`HSTreeClassifier`, `HSForestClassifier`). The single tree is the paper's
interpretability argument and is the cleaner object to demonstrate; the forest is
what goes in the results table, because the comparison set is ensembles and a
lone depth-limited CART would lose for reasons unrelated to HS.

---

## 3. Hyperparameter budget and search protocol

Implemented in `src/models/tuning.py`; audit trail in
`src/reports/tuning_results.csv`, selections in `src/reports/best_params.json`.

| Choice | What we do | Why |
|---|---|---|
| Search | **Random** search over a discrete per-model grid | A full grid would silently give LightGBM 729 configs and Kernel SVM 20. Random search fixes the number of *fits*, not the shape of the space — that is what "comparable budget" has to mean. |
| Budget | **12 configurations × 3 folds = 36 fits**, identical for every tuned model, per task | Asserted at the end of the run, not just claimed in prose: the script fails if any model on a task saw a different candidate count. |
| Excluded | `dummy` (nothing to tune) | Given no budget rather than a free pass. |
| Evaluation | K-fold CV **inside the training split only** | The validation split is reserved for calibration. Selecting hyperparameters on it would make the calibration set a second training set and break the split contract. |
| Grouping | `GroupKFold` on `match_id` for Task L | Ungrouped folds would put minute 40 of a match in train and minute 45 of the same match in validation. |
| Objective | log-loss (classification), MAE (regression) | Probability scores, matching the RPS/Brier/ECE actually reported. Selecting on accuracy then reporting calibration would be selecting on the wrong thing. |
| Budget cap | Tables > 8,000 rows tuned on a seeded subsample, **whole matches only** | Declared per row in the audit CSV, never silent. The winner is then refitted on the full training split. |
| P2's own λ | Chosen by the paper's inner 3-fold CV; the outer search tunes only the forest | Giving HS an outer λ grid *as well* would hand it a larger budget than every other model. |

---

## 4. Calibration strategy

- **Method.** Every probabilistic classifier is calibrated with
  `CalibratedClassifierCV` over a `FrozenEstimator`: isotonic first, Platt/sigmoid
  as fallback, uncalibrated as a last resort — and which one was used is recorded
  per row in the `calibration` column, never hidden.
- **Where it is fitted.** On the **validation** split, applied to **test**. Never
  on training rows (the model is already fitted there, so the calibrator would see
  in-sample probabilities) and never on test.
- **What is reported.** ECE before and after, per model, plus reliability diagrams
  drawn twice per model — raw and calibrated — in
  `src/reports/visualizations/reliability_diagrams.html`. Aggregate RPS, log-loss
  and Brier are reported alongside, because a model can improve ECE and lose RPS.
- **Per game phase, for Model 3.** ECE, mean confidence, accuracy and the signed
  confidence gap for 0–15′, 15–30′, …, 75–90′
  (`inplay_calibration_by_phase.csv`). This is where the interesting failure lives:
  late snapshots with a two-goal lead are the easy ones, so aggregate ECE flatters
  a model that is over-confident early.
- **Interaction with HS.** Shrinkage pulls probabilities toward the base rate,
  which is itself a calibration intervention. Reporting ECE before *and* after the
  post-hoc step separates "HS already fixed the over-confidence" from "the isotonic
  step did" — that separation is the point of including both columns.
- **Interaction with resampling.** The imbalance study reports per-class recall
  and F1 next to ECE precisely because aggressive oversampling usually buys
  minority recall with calibration. That trade-off is the deliverable; there is no
  winner to crown.

---

## 5. Model 3: training on the snapshot table without leaking

Four separate guarantees, each enforced in code rather than by convention.

**5.1 The time-t cut.** One snapshot per 5 regulation minutes,
t ∈ {0, 5, …, 90}. Events are ordered by `(period, index)` and the snapshot uses
the prefix of events with *effective minute* ≤ t, where effective minute is
`np.maximum.accumulate(minute)`.

*The invariant:* `maximum.accumulate` only ever raises a value, so
eff[i] ≥ minute[i] for every event. An event enters the snapshot only when
eff[i] ≤ t, which implies minute[i] ≤ t. **No event later than t can ever be
included, regardless of how corrupted the timestamps are.** The cut errs toward
excluding, never including. `build_inplay_features.assert_prefix` checks the
selected slice is a clean prefix at every snapshot. `test_inplay_cut.py`: 7/7.

*Why the repair exists:* ~26 events carry corrupted `00:00:00` timestamps. This is
also why in-play code sorts by index rather than timestamp — `REPORT.md` §3 still
describes the older `(period, timestamp, index)` ordering and is out of date on
this point.

*Volunteer the half-time consequence before being asked.* StatsBomb's `minute` is
not globally monotonic: period 1 runs into stoppage (46′, 47′) while period 2
restarts at 45′. The running maximum therefore pins the opening events of the
second half behind the first half's last minute, so they are absent from t=45 and
reappear at t=50. This is deliberate — under "t=45 means half-time" it is the
correct cut, and the alternative (capping period 1 at 45′) would pull first-half
stoppage events *into* the t=45 snapshot, which is the direction that actually
leaks.

**5.2 The match-level split.** Splits are assigned per `match_id` and every
snapshot inherits its match's split, so all 19 snapshots of a match are always on
one side of every boundary. Train dates < validation dates < test dates, strictly,
asserted in `build_temporal_splits.py`.

**5.3 Train-only preprocessing.** Imputer, scaler and encoder are fitted on
training rows only and applied frozen to validation and test. Tuning folds are
grouped by match. Calibration is fitted on validation only.

**5.4 No resampling of the snapshot table, ever.** `prepare_matrices` raises
`ValueError` if any resampling arm is requested on a frame containing
`snapshot_minute`. See §6.2 for why.

---

## 6. Design-consequence questions

### 6.1 "What would change if snapshots were taken every minute instead of every five?"

**Rows: 19 → 91 per match, 28,823 → ~138,000. Information: almost none.**

The state variables move only on discrete events — roughly 2.7 goals and 4 cards
per match. Between events, consecutive one-minute snapshots differ only in the
rolling recent-event-rate features. So nominal *n* rises 4.8× while effective
sample size barely moves; the autocorrelation between adjacent rows approaches 1.

Five concrete consequences:

1. **Within-match correlation gets worse, so the grouping matters more.** Any
   ungrouped CV becomes more optimistic, not less. Our `GroupKFold` on `match_id`
   is the thing standing between us and a badly inflated score.
2. **Confidence intervals would be wrong in a specific, quantifiable way.** A
   standard error computed as if rows were independent is optimistic by roughly
   √(1 + (m−1)ρ) with m snapshots per match; going 19 → 91 inflates that factor,
   not the precision.
3. **Compute rises ~5× for trees and ~23× for anything with a Gram matrix.** The
   exact kernel, already excluded from Task L, becomes indefensible rather than
   merely expensive.
4. **Per-phase calibration gets finer and more reliable** — more snapshots per
   phase bin — which is the one genuine benefit.
5. **The leakage machinery is unchanged.** `prefix_length_at` is defined for any
   t. But the effective-minute repair matters more, because more snapshot
   boundaries fall near the corrupted-timestamp events.

**The answer we would defend:** the cost/information trade favours the 5-minute
grid, and if we wanted more resolution the right move is not a finer uniform grid
but **event-triggered snapshots** — sample at each goal, red card and substitution,
where the state actually changes.

### 6.2 "What would change if SMOTE were applied to the snapshot table?"

**It would break the task in three distinct ways, which is why the code refuses.**

SMOTE interpolates between a minority row and one of its k nearest neighbours.
Neighbours are selected globally, so the two parents are almost always snapshots
from *different matches*.

1. **The synthetic rows are not game states.** A convex combination of
   (minute 40, 1–0, 11 v 11) and (minute 75, 0–2, 10 v 11) yields something like
   minute 57.5, goal difference −0.4, 10.6 players. Goal difference and
   man-advantage are integers; cumulative xG is monotone in minute; every one of
   those constraints is violated. The model then fits a decision surface over a
   region of feature space with no real support, and we have no way to say whether
   what it learned there is right.
2. **It moves match-level label information across the match boundary.** The label
   is a property of the *match*, replicated across all 19 snapshots. A synthetic
   row inherits the minority parent's label (match A's eventual result) while half
   its feature mass comes from match B. The brief names this explicitly in §7.2.
3. **It inflates n while shrinking effective sample size.** One match already
   contributes 19 correlated rows; SMOTE lets a single match's minority snapshots
   be re-synthesised dozens of times, so a handful of matches come to dominate
   training. Every internal CV score gets more optimistic while the model gets
   worse.

There is also a fourth, more mundane problem: `snapshot_minute` is a **feature**,
so interpolating produces minutes that are not on the grid, which makes the
per-phase calibration analysis incoherent.

**What we do instead.** Task L trains on the raw distribution and relies on
post-hoc calibration; the imbalance study is confined to the one-row-per-match
pre-match table, where a row *is* an independent example. Enforced structurally,
not by comment: `prepare_matrices` raises on any resampling arm when
`snapshot_minute` is present, and `run_models.run_classification_task` refuses the
arm before it gets that far.

**If forced to rebalance the snapshot table anyway**, the defensible options are
(a) `class_weight='balanced'` — reweight the loss, synthesise nothing;
(b) resampling *within* a match or within a phase, never across a match boundary;
(c) resampling whole matches, carrying all 19 snapshots together.

### 6.3 Other questions to expect

- **"Why is Task L's Dummy the frozen pre-match prediction and not the prior?"**
  Both are reported. The prior is the sanity floor; the frozen pre-match curve is
  the bar that justifies the in-play model existing at all. A Model 3 that merely
  matches the frozen curve has learned nothing from the live stream, however good
  its absolute RPS looks.
- **"Your kernel ridge subsampled — isn't that cheating?"** It is capped at 8,000
  training rows and the results table carries `subsampled` and the actual
  `n_train` used, so the compute row cannot be read as a full-data fit. The full
  cost curve is the kernel-scaling deliverable.
- **"Why tune on CV inside train rather than on the validation split?"** Because
  validation is the calibration set. Tuning there would fit hyperparameters and
  the calibrator on the same rows, and the reported ECE would be optimistic.
- **"Odds as a feature?"** No. Odds are used as the market baseline only. The
  brief permits either/or, never both in one experiment, and we chose baseline.
- **"Coverage?"** Every Task C model is re-scored on the intersection of the test
  set with the odds-tagged matches, and the tagging coverage rate is reported
  beside it (`market_comparison.csv`, `market_coverage.csv`). Scoring the model on
  all test matches and the market on the tagged subset would compare two different
  samples.

---

## 7. What is not in this defence

SHAP (global, local, in-play timeline), the 10-worst-prediction post-mortem, the
live online case and the bonus service belong to the **final** defence and are not
built yet. The written report is drafted to §3 and needs extending; `README.md` is
stale and is being rewritten alongside the report.
