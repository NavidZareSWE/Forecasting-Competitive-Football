# Appendix A — Hand derivation of the P2 model

**P2 paper.** Abhineet Agarwal, Yan Shuo Tan, Omer Ronen, Chandan Singh, Bin Yu.
*Hierarchical Shrinkage: Improving the accuracy and interpretability of tree-based models.*
Proceedings of the 39th International Conference on Machine Learning (ICML), 2022,
PMLR 162:111–135. <https://proceedings.mlr.press/v162/agarwal22b.html>

**Hard filters.** Published 2022 (≥ 2022 ✓). ICML is CORE A\* ✓. Non-deep — it
regularises CART trees and forests, no neural components ✓. Serves classification
and regression from one method, so a single learner covers Model 1 (Task C),
Model 2 (Task R) and Model 3 (Task L) ✓.

**Implementation.** `src/papers/hierarchical_shrinkage.py`, from scratch against the
raw `tree_` arrays. Tests: `src/papers/test_hierarchical_shrinkage.py`, 8/8.

---

## A.1 Setup and notation

Grow a CART tree by any standard criterion; HS does not touch the growing step.

For a point $x$, let its root-to-leaf path be the nested sequence of nodes

$$t_0 \supset t_1 \supset \dots \supset t_L, \qquad t_0 = \text{root}, \quad t_L = \text{leaf}(x).$$

Write

* $N(t)$ — number of training samples falling in node $t$;
* $\mu(t) = \frac{1}{N(t)}\sum_{i:\,x_i \in t} y_i$ — training mean of the response in $t$.

For classification $y_i$ is the one-hot encoding of the label, so $\mu(t) \in \Delta^{k-1}$
is the class distribution in $t$; every line below holds coordinate-wise and nothing
in the derivation depends on $y$ being scalar.

Because the nodes are nested, $N(t_0) \ge N(t_1) \ge \dots \ge N(t_L)$. This single
inequality is what makes §A.3 work.

Standard CART predicts the leaf mean, $\hat f(x) = \mu(t_L)$.

## A.2 The telescoping identity and the HS estimator

The leaf mean is exactly the root mean plus the increments picked up along the path:

$$\mu(t_L) \;=\; \mu(t_0) \;+\; \sum_{l=1}^{L}\bigl[\mu(t_l) - \mu(t_{l-1})\bigr]. \tag{A.1}$$

This is an identity — the sum telescopes — so it adds no assumption. Every term
$\mu(t_l) - \mu(t_{l-1})$ is the amount by which the $l$-th split moved the prediction.

The deeper a split sits, the fewer samples supported the decision to make it, so the
later increments are the noisy ones. HS damps each increment by the size of the node
it was estimated *from*:

$$\boxed{\;\hat f_{\mathrm{HS}}(x) \;=\; \mu(t_0) \;+\; \sum_{l=1}^{L}
\frac{\mu(t_l) - \mu(t_{l-1})}{1 + \lambda / N(t_{l-1})}\;} \tag{A.2}$$

with a single regularisation parameter $\lambda \ge 0$. Note the divisor uses
$N(t_{l-1})$, the **parent**: a split taken off a large node is trusted almost fully,
the same split taken off a nearly-empty node is almost ignored.

Setting $\lambda = 0$ in (A.2) recovers (A.1), i.e. the unshrunk tree.

## A.3 Derivation: HS is a convex combination of the ancestor means

Define the damping factors

$$w_l \;=\; \frac{1}{1 + \lambda/N(t_{l-1})} \;=\; \frac{N(t_{l-1})}{N(t_{l-1}) + \lambda},
\qquad l = 1,\dots,L, \qquad w_{L+1} := 0 .$$

Substituting into (A.2) and splitting the sum:

$$\hat f_{\mathrm{HS}}(x) = \mu(t_0) + \sum_{l=1}^{L} w_l\,\mu(t_l) - \sum_{l=1}^{L} w_l\,\mu(t_{l-1}).$$

Re-index the second sum with $l \mapsto l+1$:

$$\sum_{l=1}^{L} w_l\,\mu(t_{l-1}) \;=\; \sum_{l=0}^{L-1} w_{l+1}\,\mu(t_l).$$

Collecting the coefficient of each $\mu(t_l)$:

$$\hat f_{\mathrm{HS}}(x) \;=\; \underbrace{(1-w_1)}_{c_0}\,\mu(t_0)
\;+\; \sum_{l=1}^{L-1}\underbrace{(w_l - w_{l+1})}_{c_l}\,\mu(t_l)
\;+\; \underbrace{w_L}_{c_L}\,\mu(t_L),$$

that is

$$\boxed{\;\hat f_{\mathrm{HS}}(x) \;=\; \sum_{l=0}^{L} c_l \,\mu(t_l),
\qquad c_0 = 1-w_1,\quad c_l = w_l - w_{l+1},\quad c_L = w_L. \;} \tag{A.3}$$

**The $c_l$ are a probability vector.**

*They sum to one* — the sum telescopes:

$$\sum_{l=0}^{L} c_l = (1-w_1) + (w_1-w_2) + \dots + (w_{L-1}-w_L) + w_L = 1 .$$

*They are non-negative* — $u \mapsto u/(u+\lambda)$ is increasing in $u$ for $\lambda \ge 0$,
and the path is nested so $N(t_{l-1}) \ge N(t_l)$, hence

$$w_l = \frac{N(t_{l-1})}{N(t_{l-1})+\lambda} \;\ge\; \frac{N(t_l)}{N(t_l)+\lambda} = w_{l+1}
\;\Longrightarrow\; c_l \ge 0 .$$

So **HS returns a weighted average of the means of every node on the path**, with weight
sliding from the leaf toward the root as $\lambda$ grows. Three consequences:

1. **Classification is safe for free.** Each $\mu(t_l)$ lies on the simplex and (A.3) is a
   convex combination, so $\hat f_{\mathrm{HS}}(x)$ is a valid probability vector for
   *any* $\lambda$. No clipping, no renormalisation.
   (Verified: `test_classifier_output_is_a_valid_probability_vector`.)
2. **$\lambda = 0$:** all $w_l = 1$, so $c_L = 1$ and every other $c_l = 0$ — plain CART.
   (`test_lambda_zero_is_the_unshrunk_tree`.)
3. **$\lambda \to \infty$:** all $w_l \to 0$, so $c_0 \to 1$ — every prediction collapses
   onto the root mean, i.e. the base rate. For Task C that limit is the Dummy prior,
   which makes HS a continuous interpolation between the sanity floor and the full tree.
   (`test_large_lambda_collapses_to_the_root_mean`.)

## A.4 Derivation: the objective HS solves

(A.2) is not an ad-hoc formula; it is the closed-form minimiser of a ridge penalty
applied in the *increment basis* of the tree.

For every non-root node $t$ let $\psi_t(x) = \mathbf{1}\{x \in t\}$. Any tree function
can be written exactly in this over-complete basis as

$$f(x) \;=\; \beta_0 + \sum_{t \neq t_0} \beta_t\, \psi_t(x),
\qquad \beta_0 = \mu(t_0), \quad \beta_t = \mu(t) - \mu(\mathrm{parent}(t)),$$

because for a given $x$ only its ancestors have $\psi_t(x) = 1$, and the resulting sum is
precisely the telescoping identity (A.1). The coefficients are therefore *the increments*,
which is exactly what we want to penalise.

Take one node $t$ with parent $p$, and estimate its increment $\delta = \beta_t$ by ridge
regression restricted to the samples in $p$, holding the parent's fit $\mu(p)$ fixed:

$$J(\delta) \;=\; \sum_{i:\,x_i \in p} \bigl(y_i - \mu(p) - \delta\,\mathbf{1}\{x_i \in t\}\bigr)^2
\;+\; \lambda_t\,\delta^2 .$$

Differentiate and set to zero — only the $N(t)$ samples inside $t$ contribute to the first term:

$$\frac{\partial J}{\partial \delta} = -2\!\!\sum_{i:\,x_i \in t}\!\bigl(y_i - \mu(p) - \delta\bigr) + 2\lambda_t \delta = 0 .$$

$$\sum_{i:\,x_i \in t}\bigl(y_i - \mu(p)\bigr) \;=\; \bigl(N(t) + \lambda_t\bigr)\,\delta .$$

The left-hand side is $N(t)\bigl(\mu(t) - \mu(p)\bigr)$ by definition of $\mu(t)$, so

$$\hat\delta \;=\; \bigl(\mu(t) - \mu(p)\bigr)\,\frac{N(t)}{N(t) + \lambda_t}. \tag{A.4}$$

Now choose the penalty weight that makes (A.4) reproduce (A.2). We need

$$\frac{N(t)}{N(t)+\lambda_t} \;=\; \frac{N(p)}{N(p)+\lambda}
\quad\Longleftrightarrow\quad
\boxed{\;\lambda_t \;=\; \lambda\,\frac{N(t)}{N(p)}\;.} \tag{A.5}$$

So HS is the greedy, top-down solution of

$$\min_{\beta}\;\sum_{i=1}^{n}\Bigl(y_i - \beta_0 - \sum_{t \neq t_0}\beta_t \psi_t(x_i)\Bigr)^{2}
\;+\; \lambda \sum_{t \neq t_0} \frac{N(t)}{N(\mathrm{parent}(t))}\,\beta_t^{2}. \tag{A.6}$$

The penalty weight $N(t)/N(\mathrm{parent}(t))$ is the **fraction of its parent's samples
that the split sent into $t$**. Reading (A.6) back out: a split that cleanly halves a large
node is barely penalised, while a split that peels off a handful of points from an already
small node is penalised hard. That is the entire method — the tree is grown greedily as
usual, then its increments are ridge-shrunk in proportion to how little data supported them.

## A.5 Why this is the right P2 for this project

* **One method, three deliverables.** (A.2) is defined on $\mu(t)$, which is a mean for
  regression and a class distribution for classification. `HSForestRegressor` serves
  Tasks R and Lr; `HSForestClassifier` serves Tasks C and Lc. No per-task adaptation hack.
* **It targets the project's actual failure mode.** The snapshot table has 28,823 rows but
  only 1,517 independent matches, and deep leaves in the in-play forest are dominated by
  a handful of correlated snapshots from the same match. (A.5) penalises exactly those
  leaves.
* **It composes with the calibration study.** HS shrinks probabilities toward the base
  rate, which is a *calibration* intervention, not an accuracy one. Reporting ECE before
  and after Platt/isotonic on top of HS separates "shrinkage already fixed the
  over-confidence" from "the post-hoc calibrator did".
* **Honest limitation to state at the defence.** HS cannot fix a badly grown tree; it only
  redistributes weight along paths that CART already chose. If the split criterion picked
  the wrong variable, (A.3) averages over a wrong path.

## A.6 Correspondence to the implementation

| Equation | Code |
|---|---|
| (A.2) telescoping form, evaluated top-down | `_shrink_tree` in `hierarchical_shrinkage.py` |
| (A.3) convex form — justifies no clipping | asserted by `test_classifier_output_is_a_valid_probability_vector` |
| (A.1) literal per-path transcription, as an independent check | `test_shrinkage_matches_the_paper_equation_on_every_path` |
| $\lambda$ chosen by K-fold CV on **training rows only** | `_HSBase._select_lambda` |
| leaf lookup (own level-wise descent, not `tree_.apply`) | `_leaf_indices` |

$\lambda$ is selected from $\{0, 0.1, 1, 10, 25, 50, 100\}$ by 3-fold cross-validation
*inside the training split*. The project's validation split is reserved for probability
calibration; tuning $\lambda$ on it would turn the calibration set into a second training
set and break the split contract.
