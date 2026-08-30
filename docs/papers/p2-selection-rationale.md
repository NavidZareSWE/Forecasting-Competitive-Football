# P2 Model Candidates for "Forecasting Competitive Football" — Filter & Reproducibility Audit

## Decision

**Selected: Hierarchical Shrinkage (HS)** — Agarwal, Tan, Ronen, Singh, Yu, ICML 2022,
PMLR 162:111–135. Chosen over FIGS for the simpler closed-form Appendix A derivation.

- Implementation: `src/papers/hierarchical_shrinkage.py` (from scratch, 8/8 tests)
- Hand derivation: `docs/appendix_a_hierarchical_shrinkage.md`
- Serves Tasks C, R and L from one method — `HSForestClassifier` / `HSForestRegressor`
- **TA sign-off: pending, due at Mid Defence 2**

FIGS retained as the fallback if the TA rejects HS. The audit below is the evidence.

## TL;DR
- Two candidates clear every mandatory hard filter AND the new reproducibility bar at ≥8.5: **Hierarchical Shrinkage** (ICML 2022, CORE A*; confidence 9.5) and **Fast Interpretable Greedy-Tree Sums / FIGS** (PNAS 2025, Q1 Multidisciplinary; confidence 9.0). Both are non-deep tree methods, serve classification and regression, have a clean central equation to hand-derive, and ship an official Python implementation in the `imodels` package (co-authored by the papers' authors).
- Of the four already-known candidates, only Hierarchical Shrinkage and FIGS survive. **GPBoost and Distributional Random Forests are rejected on reproducibility** (large C++ codebases with approximations not fully specified in the paper — not realistically reimplementable in Python from the paper alone), and **NGBoost is rejected on the date filter** (published at ICML 2020, PMLR 119:2690–2700, before the 2022 cutoff).
- A strong new ordinal candidate — **Frequency-Adjusted Borders Ordinal Forest (fabOF)**, British Journal of Mathematical and Statistical Psychology 2025, Q1 — is presented as a borderline case scored 8.0 (just below threshold). It elegantly bridges classification and regression for the naturally-ordinal Home/Draw/Away target, but I cannot honestly certify it ≥8.5 because exact reproduction requires reading its R source, and its "novel" part is a thin post-processing layer over a library random forest.

## Key Findings
- Applying the new from-scratch-reproducibility requirement strictly is what does most of the filtering work here. Two of the four already-surfaced methods (HS, FIGS) are excellent; two (DRF, GPBoost) are correctly dated and Q1/top-tier but are not realistically reimplementable in Python from the paper text alone.
- The single most valuable new direction the PDF flagged — ordinal models that bridge classification and regression — does yield a genuinely relevant, recent, Q1 method (fabOF), but it lands just under the confidence threshold rather than above it.
- Several attractive-sounding probabilistic/boosting methods (NGBoost, PGBM, Sigrist's Gradient/Newton Boosting) are disqualified purely on the 2022+ publication-date rule. This is flagged explicitly so the student does not waste time re-finding them.
- Being conservative is the right call: it is far better to hand the student two rock-solid methods (HS, FIGS) that provably satisfy every filter than to pad the list with methods whose reproducibility depends on undocumented source-code details.

## Details

### RECOMMENDED CANDIDATE 1 — Hierarchical Shrinkage (HS)

1. **Model name:** Hierarchical Shrinkage (HS).
2. **Full citation:** Abhineet Agarwal, Yan Shuo Tan, Omer Ronen, Chandan Singh, Bin Yu. "Hierarchical Shrinkage: Improving the accuracy and interpretability of tree-based models." Proceedings of the 39th International Conference on Machine Learning (ICML), 2022, PMLR 162:111–135 (session 17–23 July 2022).
3. **Paper link / PDF:** Publisher page https://proceedings.mlr.press/v162/agarwal22b.html ; PDF https://proceedings.mlr.press/v162/agarwal22b/agarwal22b.pdf
4. **Official GitHub:** https://github.com/csinva/imodels — the `imodels` package maintained by Chandan Singh (a paper co-author); Python; classes `HSTreeClassifier`, `HSTreeRegressor`, `HSTreeClassifierCV`, `HSTreeRegressorCV` in `imodels/tree/hierarchical_shrinkage.py`.
5. **Reproducibility route:** Combination — (a) official Python code in `imodels` authored by a co-author, PLUS (c) a fully closed-form central equation in the paper. Either route alone would suffice.
6. **Why easy to reproduce:** HS is a single closed-form post-hoc transformation applied after a standard CART tree or ensemble is trained: each leaf prediction is replaced by a telescoping weighted average of the mean responses along the root-to-leaf path, shrinking each node's prediction toward its ancestors' sample means using a single regularization parameter λ. There is no iterative optimizer, no hidden constant, and no unstated engineering trick — the whole method is a one-line recursion over ancestor nodes, and the `imodels` reference code confirms the exact node-based shrinkage scheme.
7. **Hard-filter confirmation:**
   - *Year:* ICML 2022 — satisfies 2022+ (confirmed on PMLR v162 record).
   - *Venue:* ICML is CORE A* (verified in the ICORE/CORE conference export at portal.core.edu.au, where ICML is listed A*).
   - *Not deep learning:* it regularizes CART decision trees / random forests — no neural components.
   - *Classification AND regression:* both supported and demonstrated in the paper (evaluated across classification and regression benchmark datasets); `imodels` ships both a classifier and a regressor, so a single method serves Tasks C, R and L.
   - *Core equation for Appendix A:* the hierarchical-shrinkage leaf-prediction formula (the ancestor-weighted telescoping average controlled by λ), which is trivial to derive by hand.
8. **Reproducibility confidence: 9.5 / 10** — a single closed-form update rule with an official Python reference implementation authored by the paper's own group; essentially no reverse-engineering risk.

### RECOMMENDED CANDIDATE 2 — Fast Interpretable Greedy-Tree Sums (FIGS)

1. **Model name:** Fast Interpretable Greedy-Tree Sums (FIGS).
2. **Full citation:** Yan Shuo Tan, Chandan Singh, Keyan Nasseri, Abhineet Agarwal, James Duncan, Omer Ronen, Matthew Epland, Aaron Kornblith, Bin Yu. "Fast Interpretable Greedy-Tree Sums." Proceedings of the National Academy of Sciences (PNAS), 2025, 122(7):e2310151122. DOI 10.1073/pnas.2310151122 (published online February 14, 2025; PubMed PMID 39951504).
3. **Paper link / PDF:** Publisher page https://www.pnas.org/doi/10.1073/pnas.2310151122 ; preprint PDF (arXiv 2201.11931, first posted January 2022) https://arxiv.org/abs/2201.11931
4. **Official GitHub:** https://github.com/csinva/imodels — `FIGSClassifier` and `FIGSRegressor`; Python.
5. **Reproducibility route:** Combination — (b) explicit numbered pseudocode ("Algorithm 1, FIGS fitting algorithm") PLUS (a) official Python code in `imodels`.
6. **Why easy to reproduce:** Algorithm 1 gives the full greedy training loop. FIGS generalizes CART by growing a flexible number of trees in a summation; at each step it considers the single split (across all current trees, plus starting a potential new tree) that most reduces residual impurity, where each tree is fit to the residual of the sum of all other trees. The split criterion is standard CART impurity reduction applied to residuals, and the total number of splits is capped by one threshold — so the entire loop is reconstructable from the pseudocode and prose, and cross-checkable against the `imodels` implementation.
7. **Hard-filter confirmation:**
   - *Year:* PNAS, published online February 14, 2025 — satisfies 2022+. This is exactly the "preprinted 2022 but formally published later" case the brief says is acceptable, and it is reported at its 2025 publication date (not the 2022 arXiv date).
   - *Venue:* PNAS proper (not PNAS Nexus) is Q1 in Multidisciplinary, verified on Scimago (SCImago category "Multidisciplinary (Q1)").
   - *Not deep learning:* sum-of-CART-trees, no neural components.
   - *Classification AND regression:* the paper covers both (a CART generalization evaluated on classification and regression datasets); `imodels` ships `FIGSClassifier` and `FIGSRegressor`.
   - *Core equation for Appendix A:* the greedy tree-sum split-selection / residual-update rule from Algorithm 1.
8. **Reproducibility confidence: 9.0 / 10** — numbered pseudocode plus official Python reference; the only mild care needed is matching the stopping rule (max total splits) and the residual bookkeeping across the multiple trees.

### BORDERLINE (BELOW THRESHOLD) — Frequency-Adjusted Borders Ordinal Forest (fabOF)

1. **Model name:** Frequency-Adjusted Borders Ordinal Forest (fabOF).
2. **Full citation:** Philip Buczak (sole author, TU Dortmund University). "Frequency-adjusted borders ordinal forest: A novel tree ensemble method for ordinal prediction." British Journal of Mathematical and Statistical Psychology, 2025, 78(2):594–616. DOI 10.1111/bmsp.12375 (accepted 2024-11-09; Epub 2024-12-08; issue date 2025 May; PubMed PMID 39648591).
3. **Paper link / PDF:** Publisher https://bpspsychub.onlinelibrary.wiley.com/doi/10.1111/bmsp.12375 ; open-access full text (PMC) https://pmc.ncbi.nlm.nih.gov/articles/PMC11971599/
4. **Official GitHub / code:** https://github.com/phibuc/fabOF — authored by Philip Buczak (the paper's sole author); **R** package; `Imports: psych, ranger` (relies on the `ranger` random-forest package for the underlying forest). Prototype code and data also archived at OSF: https://osf.io/fn8bg/ .
5. **Reproducibility route:** All three, partially — (a) official code exists but in R; (b) Algorithm 3 pseudocode exists (though embedded as an image in the open-access HTML); (c) the prose in Section 3.2 fully narrates the border-computation heuristic.
6. **Why (mostly) reproducible, and why it falls short of 8.5:** The method is conceptually simple. Assign fixed integer scores 1…k to the ordinal categories; fit a regression random forest on those scores; obtain out-of-bag (OOB) numeric predictions; then set the inner category borders b₂…b_k to the empirical quantiles of the OOB predictions at the cumulative relative frequencies π₁…π_{k−1} of the training categories, with outer borders fixed at b₁=1 and b_{k+1}=k; classify a new observation by the interval its averaged forest prediction falls into (AFTA aggregation ŷ_i^num = (1/B)Σⱼ ŷ_ij^num, then interval lookup). The central equation — frequency-adjusted borders as quantiles of OOB predictions at cumulative class frequencies — is clearly identifiable. Three honest concerns keep it below 8.5: (i) the exact quantile definition/interpolation type and interval-endpoint/tie conventions are not stated in the paper text and live only in the R source, so bit-for-bit reproduction needs the code; (ii) the base learner is a library random forest (`ranger`), so the "novel" reimplemented part is a thin post-processing layer rather than a full learner, sitting awkwardly with the "reimplement from scratch, not a library call" filter; (iii) the "core equation" is a frequency-matching heuristic rather than an update rule/objective, a weaker fit for the Appendix A hand-derivation requirement.
7. **Hard-filter confirmation:** Year 2025 (Epub Dec 2024) — OK. Venue is Q1 in Statistics & Probability (verified on Scimago; also Q1 in Mathematics-interdisciplinary and Psychology-mathematical) — OK. Not deep learning (regression random forest via `ranger`; neural nets appear only as others' unrelated prior work) — OK. Task coverage: it is an ordinal method that elegantly bridges the Home/Draw/Away classification and the goal-margin regression, but it demonstrates only the ordinal task, not classification and regression separately — weaker on filter 4.
8. **Reproducibility confidence: 8.0 / 10 (below the 8.5 threshold)** — the logic is fully clear and an experienced developer could reproduce the *method*, but exact numerical reproduction requires reading the R code for the quantile type and endpoint conventions, so I cannot honestly certify ≥8.5.

### Re-scored already-known candidates
- **Hierarchical Shrinkage** — RECOMMENDED (see above). Verified: ICML 2022, CORE A*, non-DL, both tasks, closed-form equation, official `imodels` Python code. 9.5.
- **FIGS** — RECOMMENDED (see above). Verified: PNAS 2025, Q1, non-DL, both tasks, Algorithm 1, official `imodels` Python code. 9.0.
- **Distributional Random Forests (DRF)** — Ćevid, Michel, Näf, Bühlmann, Meinshausen, JMLR 23(333):1–79, 2022. Year and venue (JMLR, which is Q1 — SJR ~2.0, best quartile Q1 on Scimago) are fine and it is non-deep. **Rejected on reproducibility:** its core is a novel MMD-based (Maximum Mean Discrepancy) split criterion, and the official repo https://github.com/lorismichel/drf is a large C++/Rcpp codebase forked from grf/ranger. A faithful from-scratch Python reimplementation of the fast MMD splitting is not realistic from the paper alone. Reproducibility confidence ~5–6, below threshold.
- **Gaussian Process Boosting (GPBoost) / Latent Gaussian Model Boosting (LaGaBoost)** — Sigrist, JMLR 23(232):1–46, 2022, and IEEE TPAMI 2023 (DOI 10.1109/TPAMI.2022.3168152). Year/venue/non-DL fine. **Rejected on reproducibility:** the method combines tree-boosting with Gaussian-process/mixed-effects models plus Vecchia and Laplace approximations; the official repo https://github.com/fabsig/GPBoost is predominantly C++ forked from LightGBM. From-scratch Python reproduction from the paper is unrealistic. Reproducibility confidence ~4–5, below threshold.

## Recommendations
- **Stage 1 — commit to Hierarchical Shrinkage as the primary P2 method now.** It is the safest choice: one closed-form equation ideal for the Appendix A hand derivation, an official Python reference in `imodels` for cross-checking, identical applicability to Tasks C, R and L (classifier, regressor, and snapshot retraining), and shrunk leaf means that play nicely with your downstream Platt/isotonic calibration, ECE/reliability diagrams, and SHAP. Benchmark against the de-vigged bookmaker odds.
- **Stage 2 — use FIGS as the second method / robustness comparison.** Also in `imodels`, also serves all three tasks, and Algorithm 1 gives a second clean hand-derivation option (the greedy tree-sum split criterion). If a single P2 method is required, pick HS for maximal derivation simplicity; pick FIGS if you want a more expressive additive tree-sum model that captures additive structure in the pre-match features.
- **Stage 3 — only pursue fabOF if you explicitly want the ordinal angle and accept the caveats.** It is the most conceptually elegant match for the ordinal Home/Draw/Away target (and would bridge Tasks C and R via one latent score), but before claiming faithful reproduction you must (a) confirm the quantile type and endpoint conventions by reading the R source, and (b) reimplement the random forest yourself rather than calling one, to satisfy filter 6. Resolve both and its score rises toward 8.5; until then treat it as a secondary experiment, not the core P2 model.
- **Optional lead to investigate:** Tutz (2022), "Ordinal Trees and Random Forests: Score-Free Recursive Partitioning and Improved Ensembles," Journal of Classification 39(2):241–263 — 2022, Q1 (Journal of Classification is Q1 on Scimago), non-deep, ordinal, with a clear score-free split principle. I found no confirmed official code repository, so its reproducibility rests on the paper text alone and I could not verify it to ≥8.5. Worth a closer read if an ordinal method is desired.
- **Thresholds that would change these calls:** if the TA relaxes filter 6 to allow a library base learner for the "wrapper" part of a method, fabOF moves into the recommended set. If the TA accepts JMLR-published C++ methods as reproducible via their official repos (contradicting the from-scratch requirement), DRF and GPBoost re-enter. If the date rule is relaxed to include 2020–2021, NGBoost and PGBM become strong probabilistic-calibration candidates.

## Caveats
- I verified ICML's A* rank against the CORE/ICORE portal and PNAS's, BJMSP's, Journal of Classification's, and JMLR's Q1 status against Scimago. JMLR's Q1 is confirmed (SJR ≈ 2.0, best quartile Q1, 2024) but is moot because DRF and GPBoost are rejected on reproducibility, not venue.
- None of HS, FIGS, or fabOF produce calibrated probabilities intrinsically; the project's planned Platt/isotonic post-hoc calibration is therefore necessary and appropriate for all three, and is a natural place to address the ~26.5% draw minority class.
- fabOF's Algorithm 3 is embedded as an image in the open-access HTML, so I relied on the paper's prose (which fully narrates the steps) rather than a machine-readable pseudocode listing.
- Zero-fabrication compliance: every URL, DOI, volume/page, and author list above was retrieved during research. Where I could not fully verify a detail (e.g., the exact quantile type inside fabOF's R code), I have said so explicitly rather than guessing.

## Considered and Rejected (with specific reasons)
- **NGBoost** (Duan, Anand, Ding, Thai, Basu, Ng, Schuler), ICML 2020, PMLR 119:2690–2700 (presented 13–18 July 2020) — REJECTED: actual publication date is 2020, before the 2022 cutoff. Otherwise attractive (probabilistic, A*, non-DL, official Python at github.com/stanfordmlgroup/ngboost).
- **Probabilistic Gradient Boosting Machines (PGBM)** (Sprangers, Schelter, de Rijke), KDD 2021 — REJECTED: published 2021, before cutoff (official Python at github.com/elephaint/pgbm).
- **Gradient and Newton Boosting for Classification and Regression** (Sigrist), Expert Systems with Applications 167:114080, 2021 — REJECTED: published 2021, before cutoff.
- **Condensed Gradient Boosting**, International Journal of Machine Learning and Cybernetics — REJECTED: journal is Q2/Q3 (already rejected by the TA; listed here to prevent re-discovery).
- **OGBoost** (arXiv 2502.13456, 2025) and **OrdinalGBT** (software package) — REJECTED: not published in a Q1 journal or an A*/A conference (arXiv preprint / software only).
- **Ordinal Forest** (Hornung), Journal of Classification 37:4–17 — REJECTED: published 2018/2020, before cutoff (though Journal of Classification is Q1).
- **Meta Ordinal Regression Forest (MORF)**, IEEE/CAA Journal of Automatica Sinica 9(7):1233–1247, 2022 — REJECTED: it is a deep ordinal regression forest (CNN + differentiable forest) and image-domain — fails both the non-deep-learning and the domain-agnostic filters.
- **XBART / Stochastic Tree Ensembles** (He, Yalov, Hahn; He & Hahn) — REJECTED (with uncertainty): the AISTATS conference version is 2019 and the JASA journal version's year is ambiguous across sources; the official implementation (github.com/JingyuHe/XBART) is a C++ codebase. Not confidently ≥2022 and not confidently reproducible from scratch.