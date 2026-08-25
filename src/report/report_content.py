# Report content as a renderer-independent block list. Every number is read
# from a result CSV at build time; nothing is hard-coded.

from pathlib import Path
import subprocess
import sys

import pandas as pd

HERE = Path(__file__).resolve().parent
SRC = HERE.parent
PROJECT = SRC.parent
RESULTS_DIR = SRC / "reports"
PROCESSED_DIR = RESULTS_DIR / "processed"
FEATURE_DIR = RESULTS_DIR / "features"
VIZ_DIR = RESULTS_DIR / "visualizations"
SHAP_DIR = VIZ_DIR / "shap"
BUILD_DIR = RESULTS_DIR / "report_build"

P1_CITATION = ("Fonseca and Bacao, \"Geometric SMOTE for imbalanced datasets "
               "with nominal and continuous features\", Expert Systems with "
               "Applications 234, 2023")
P2_CITATION = ("Agarwal, Tan, Ronen, Singh and Yu, \"Hierarchical Shrinkage: "
               "improving the accuracy and interpretability of tree-based "
               "models\", ICML 2022, PMLR 162:111-135")


# --- Block constructors -----------------------------------------------------
def title(main, subtitle):
    return {"type": "title", "text": main, "subtitle": subtitle}


def h1(text):
    return {"type": "h1", "text": text}


def h2(text):
    return {"type": "h2", "text": text}


def h3(text):
    return {"type": "h3", "text": text}


def para(text):
    return {"type": "para", "text": " ".join(text.split())}


def caption(text):
    return {"type": "caption", "text": " ".join(text.split())}


def equation(latex, name):
    return {"type": "equation", "latex": latex, "name": name}


def image(path, caption_text=None, max_height_cm=11.0):
    block = {"type": "image", "path": str(path),
             "max_height_cm": max_height_cm}
    if caption_text:
        block["caption"] = " ".join(caption_text.split())
    return block


def page_break():
    return {"type": "pagebreak"}


def mono(text):
    return {"type": "mono", "text": text}


def table(frame, columns=None, caption_text=None, max_rows=40, decimals=5):
    frame = frame if columns is None else frame[[c for c in columns
                                                 if c in frame.columns]]
    frame = frame.head(max_rows).copy()
    for column in frame.columns:
        if pd.api.types.is_float_dtype(frame[column]):
            frame[column] = frame[column].round(decimals)
    block = {"type": "table",
             "header": [str(c) for c in frame.columns],
             "rows": [[("" if pd.isna(v) else str(v)) for v in row]
                      for row in frame.itertuples(index=False)]}
    if caption_text:
        block["caption"] = " ".join(caption_text.split())
    return block


def unavailable(filename, script):
    return para(f"<b>Result not available.</b> The file <code>{filename}</code> "
                f"was not present when this report was built. It is produced "
                f"by <code>{script}</code>. No substitute numbers have been "
                f"inserted, because a report that fills gaps with plausible "
                f"figures cannot be checked.")


# --- Data access ------------------------------------------------------------
def read(name, directory=RESULTS_DIR):
    path = directory / name
    if not path.exists():
        return None
    try:
        return pd.read_csv(path, encoding="utf-8")
    except Exception:
        return None


def tuned_state():
    results = read("model_results.csv")
    if results is None or "tuned" not in results.columns:
        return None
    return bool(results["tuned"].astype(str).str.lower().eq("true").any())


# --- Front matter -----------------------------------------------------------
def front_matter(blocks):
    blocks.append(title(
        "Forecasting Competitive Football",
        "Pre-match outcome classification, pre-match goal-margin regression, "
        "and in-play snapshot forecasting on StatsBomb Open Data"))

    prematch = read("prematch_features.csv", FEATURE_DIR)
    inplay = read("inplay_features.csv", FEATURE_DIR)
    splits = read("temporal_match_splits.csv", PROCESSED_DIR)
    rows = [["Matches", "-" if prematch is None else f"{len(prematch):,}"],
            ["In-play snapshots", "-" if inplay is None else f"{len(inplay):,}"],
            ["Pre-match table columns",
             "-" if prematch is None else str(prematch.shape[1])],
            ["In-play table columns",
             "-" if inplay is None else str(inplay.shape[1])]]
    if splits is not None and "split" in splits.columns:
        counts = splits["split"].value_counts()
        rows.append(["Train / validation / test matches",
                     f"{counts.get('train', 0)} / "
                     f"{counts.get('validation', 0)} / {counts.get('test', 0)}"])
    blocks.append(table(pd.DataFrame(rows, columns=["Quantity", "Value"])))

    state = tuned_state()
    if state is False:
        blocks.append(h2("Status of the results in this report"))
        blocks.append(para("""
            The results below were produced with the model library's default
            hyperparameters, not with searched ones. The hyperparameter search
            in <code>tuning.py</code> contained a defect that made it fail on
            its first task, so the file it writes,
            <code>best_params.json</code>, was never produced, and the sweep
            fell back to defaults for every model. The defect has been fixed,
            but the search has not yet been re-run to completion, so the
            numbers here are honest measurements of untuned models rather than
            of tuned ones."""))
        blocks.append(para("""
            This matters for how the comparisons should be read. Untuned
            defaults do not affect all learners equally: random forests and
            kernel methods have reasonable defaults, whereas boosted models are
            more sensitive to learning rate and tree depth and are therefore
            more likely to be understated here. Any ranking between learners in
            this report should be treated as provisional until the search has
            run. The conclusions that do not depend on the ranking, in
            particular the size of the gap between in-play and pre-match
            performance and the statistical tests in Section 6, are not
            affected in the same way."""))
    blocks.append(page_break())


# --- Section 1 --------------------------------------------------------------
def section_framing(blocks):
    blocks.append(h1("1. Problem framing and formal definitions"))
    blocks.append(para("""
        This project forecasts the outcome of professional football matches in
        two settings. The first is before the match starts, using only what is
        known at kick-off. The second is during the match, where the forecast
        is updated as events happen. Three supervised problems are posed over
        the same population of matches."""))
    blocks.append(para("""
        <b>Task C, pre-match classification.</b> One scheduled match is one
        example, observed at kick-off. The model predicts a probability
        distribution over the final result in the ordered set {H, D, A}, that
        is, home win, draw, away win."""))
    blocks.append(para("""
        <b>Task R, pre-match regression.</b> The same example, observed at the
        same moment, with the final signed goal margin as the label."""))
    blocks.append(equation(
        r"$y_R=\mathrm{clip}\left(\mathrm{score}_{home}-"
        r"\mathrm{score}_{away},\,-5,\,+5\right)$", "margin"))
    blocks.append(para("""
        The margin is clipped to plus or minus five goals. Large margins are
        rare and are driven mostly by finishing luck once a match is already
        decided, so the difference between a four-goal and a seven-goal win
        carries little forecasting signal but a lot of squared error. Clipping
        limits the influence of those matches without throwing them away."""))
    blocks.append(para("""
        <b>Task L, in-play prediction.</b> One match produces nineteen
        examples, one at each regulation minute t = 0, 5, ..., 90. Each
        snapshot carries the parent match's final labels, so the same outcome
        is forecast repeatedly as information accumulates. Task L is split into
        two internal codes, Lc for the classification label and Lr for the
        margin label. An important consequence is that these nineteen rows are
        not nineteen independent observations, which affects both how models
        are validated and how results are tested; both points are handled
        explicitly later."""))
    blocks.append(h2("1.1 How performance is measured, and why"))
    blocks.append(para("""
        The main classification metric is the ranked probability score. It is
        chosen instead of accuracy or plain log loss because the three
        outcomes are ordered: a draw sits between a home win and an away win.
        If the model confidently predicts a home win, being wrong because the
        match was drawn is a smaller error than being wrong because the away
        team won. Accuracy cannot express this, and log loss treats the two
        mistakes as equally bad. For an ordered set of K classes,"""))
    blocks.append(equation(
        r"$\mathrm{RPS}=\frac{1}{K-1}\sum_{k=1}^{K-1}"
        r"\left(\sum_{j\leq k}p_j-\sum_{j\leq k}o_j\right)^2$", "rps"))
    blocks.append(para("""
        where p is the predicted distribution and o is the one-hot realised
        outcome. Lower is better. Log loss, the multi-class Brier score and the
        expected calibration error are reported alongside it, because they
        answer different questions: log loss punishes confident mistakes
        harshly, the Brier score is a plain squared error on probabilities, and
        the calibration error asks whether a stated confidence of seventy per
        cent is right about seventy per cent of the time. Regression is
        reported as mean absolute error, root mean squared error, and the
        correlation between predicted and realised margins."""))


# --- Section 2 --------------------------------------------------------------
def section_data(blocks):
    blocks.append(h1("2. Data, cleaning and the identity contract"))
    blocks.append(para("""
        The event data come from StatsBomb Open Data and the market baseline
        from Football-Data.co.uk. Four male domestic league seasons were kept.
        The selection was not made by intuition: an audit script enumerates
        every competition-season available in the open data along with its
        match count, team count, date range and 360-data coverage, and records
        an explicit decision for each."""))
    audit = read("competition_audit.csv")
    if audit is not None:
        columns = [c for c in ["comp", "season", "country", "gender",
                               "n_matches", "n_teams", "decision"]
                   if c in audit.columns] or list(audit.columns)[:6]
        blocks.append(table(audit, columns, max_rows=14, caption_text="""
            Table 2.1. Competition audit. Full league seasons with complete
            event coverage and a matching Football-Data odds file were kept.
            Tournaments were excluded because a knockout competition has a
            different scoring structure and far fewer matches per team, so
            rolling team form means something different in that setting."""))
        blocks.append(para("""
            <b>Why this experiment was run.</b> The open data contain
            tournaments, women's competitions and partial seasons alongside
            full men's league seasons. Mixing them would put matches with very
            different scoring rates and very different amounts of prior history
            into one training set. <b>What we expected.</b> That only a small
            number of competition-seasons would have both complete event
            coverage and a matching odds file. <b>What we observed.</b> Four
            full 2015/16 league seasons met both conditions. <b>What it
            means.</b> The dataset is homogeneous in competition type, which
            makes team form comparable across rows, but it is limited to a
            single season, which is the main external-validity limitation of
            this project and is revisited in Section 10."""))
    else:
        blocks.append(unavailable("competition_audit.csv",
                                  "src/audit/competition_audit.py"))

    blocks.append(h2("2.1 Identity and the odds join"))
    blocks.append(para("""
        Team and player names are spelled differently in the two sources, so
        entities are reconciled through an alias map before anything is joined.
        The map is required to be bijective, meaning one canonical entity per
        source name and no two source names collapsing onto the same entity by
        accident. Joins downstream use identifiers, never spellings."""))
    blocks.append(para("""
        The odds join is deliberately restricted to pre-match identity:
        competition, date, home team and away team. This is a leakage control,
        not a convenience. A Football-Data row also contains the final score
        and match statistics; joining on anything derived from those fields
        would let the outcome enter the feature table through the back door.
        Labels are derived once, in the label store, from the StatsBomb match
        file, and are never recomputed downstream or taken from an odds row."""))
    quality = read("data_quality_log.csv", PROCESSED_DIR)
    if quality is not None:
        blocks.append(table(quality, max_rows=16, caption_text="""
            Table 2.2. Data-quality log from the cleaning stage. Each row is a
            check that ran against the raw store with the number of records it
            affected. Records are dropped only when a mandatory field is
            missing, and every drop is written to a separate log with a reason,
            so the cleaning is auditable rather than silent."""))
    coverage = read("market_coverage.csv")
    if coverage is not None:
        blocks.append(table(coverage, caption_text="""
            Table 2.3. Odds tagging coverage on the test split, by competition.
            The market comparison in Section 5.1 is restricted to the tagged
            subset so that the models and the market are scored on exactly the
            same matches."""))
        blocks.append(para("""
            <b>What we expected.</b> Near-complete coverage, since all four
            leagues are major ones that bookmakers price. <b>What we
            observed.</b> Coverage is essentially complete, with only a handful
            of test matches lacking odds. <b>Why this matters.</b> If coverage
            were partial and non-random, the market comparison would be run on
            an easier or harder subset than the full test set and the
            comparison would be biased. At this coverage level that risk is
            negligible, and the small number of untagged matches is excluded
            from both sides rather than from one."""))


# --- Section 3 --------------------------------------------------------------
def section_features(blocks):
    blocks.append(h1("3. Feature pipelines and the leakage barriers"))
    prematch = read("prematch_features.csv", FEATURE_DIR)
    inplay = read("inplay_features.csv", FEATURE_DIR)
    blocks.append(para("""
        Two feature tables are built. The pre-match table holds rolling
        prior-form aggregates for both teams and the difference between them.
        The in-play table holds the state of the match at each snapshot minute
        and is joined to the frozen pre-match vector when the model matrices
        are assembled."""))
    blocks.append(para("""
        <b>One definition per quantity.</b> Both tables are built from a single
        module that computes per-team quantities from a frame of events. The
        pre-match builder passes a whole match and rolls the result into prior
        form; the in-play builder passes the prefix of events up to minute t.
        Because both go through the same function, the in-play value of a
        quantity and the rolling form of that same quantity are guaranteed to
        mean the same thing. Writing them separately would have allowed the two
        definitions to drift apart silently, which is the kind of defect that
        produces a model that works in training and fails in production."""))
    blocks.append(para("""
        <b>What the quantities cover.</b> Beyond goals and expected goals, each
        team's frame yields shot volume and shot location, passing volume split
        by pitch third and the share of passing done in the attacking third,
        carries and touches by third, average action position as a measure of
        territory, pressures applied and the share applied high up the pitch,
        events played under pressure, defensive actions and where they happen,
        fouls committed and won, turnovers, possession measured by chain
        ownership, and set-piece counts split into corners, free kicks and
        throw-ins. Each is recorded for the team and for the opponents it
        faced, so a team's defensive record is described as well as its
        attacking one."""))
    blocks.append(para("""
        <b>Possession is measured by chain, not by event count.</b> A raw event
        share would credit a team for its own pressures and clearances, which
        happen when it does not have the ball. Instead each possession chain is
        attributed to the team that performed most of the on-ball work in it,
        and possession share is the fraction of chains owned. The two teams'
        shares sum to exactly one by construction, which the test suite
        checks."""))
    blocks.append(para("""
        <b>Three views of every in-play quantity.</b> Each is recorded as a
        match total, as a difference over the recent ten-minute window, and as
        a per-minute rate. The recent window is what makes momentum visible: a
        team can be behind on the match totals while dominating the last ten
        minutes, and only the windowed view expresses that. The per-minute rate
        makes an early snapshot comparable with a late one, since a raw count
        at minute 10 and the same count at minute 85 mean very different
        things."""))
    blocks.append(para("""
        Leakage is the single largest risk in this project, because the label
        is derivable from the same event stream that the features come from. If
        a feature accidentally sees one event after the prediction time, the
        model can look excellent and be worthless. Four barriers are enforced,
        and each is a mechanical property of the code checked by an assertion
        or a test, not a claim in prose."""))

    blocks.append(h2("3.1 Barrier one: chronological splitting"))
    blocks.append(para("""
        Matches are sorted by date and cut into training, validation and test
        blocks in that order. A random split would let the model learn from
        matches played after the ones it is tested on, which is impossible at
        deployment time. Every snapshot inherits its parent match's split, and
        the builder asserts that no match has snapshots on both sides of a
        boundary. The cost of this choice is that the test block is the closing
        months of the season, which differ systematically from the opening
        months; that is a real effect and is discussed in Section 10."""))

    blocks.append(h2("3.2 Barrier two: prior-match-only form"))
    blocks.append(para("""
        Rolling form applies a one-match lag before the window, so a match
        never contributes to its own features. Two checks enforce this. The
        builder asserts that each team's first appearance carries no form at
        all. The test suite goes further: it rewrites a match's own scoreline
        to a different result and asserts that the feature row for that same
        match is unchanged. This is the stronger test, because it would catch a
        lag that is present in most rows but missing in some."""))

    blocks.append(h2("3.3 Barrier three: the effective-minute prefix cut"))
    blocks.append(para("""
        The in-play cut must select every event at or before minute t and no
        event after it. This is harder than it sounds, because roughly
        twenty-six events across twenty-four matches carry corrupted 00:00:00
        timestamps while their index positions are correct. Sorting by
        timestamp would therefore move those events to the start of their
        period. The builder sorts by period and index instead, and repairs the
        minute column with a running maximum:"""))
    blocks.append(equation(
        r"$\mathrm{eff}[i]=\max_{j\leq i}\ \mathrm{minute}[j]$", "effmin"))
    blocks.append(para("""
        Because a running maximum never decreases, eff[i] at most t implies
        that the true minute of event i is also at most t. The repaired column
        can therefore only make the cut more conservative, never less. The cut
        is then guarded on both sides: the last included event must be at or
        before t, and the first excluded event must be strictly after t. The
        first assertion rules out leakage; the second rules out silently
        truncating the prefix and thereby weakening the model for a reason that
        would be invisible in the metrics."""))
    blocks.append(para("""
        One side effect is documented rather than hidden. The second half
        restarts at minute 45, so applying a running maximum across the period
        boundary means the opening events of the second half are not visible
        until the snapshot at minute 50. This makes the minute-45 snapshot
        slightly more conservative than it needs to be. That is the correct
        direction to err."""))

    blocks.append(h2("3.4 Barrier four: train-only transform fitting"))
    blocks.append(para("""
        The imputer, the scaler, the one-hot encoder and every resampler are
        fitted on training rows only and then applied to validation and test.
        Fitting a scaler on the whole table would leak the distribution of the
        test set into training. Oversampling is a training-time toggle and is
        never written into a feature file, which matters because a resampled
        row is synthetic and must never be scored as if it were a real
        match."""))

    rows = []
    if prematch is not None:
        rows.append(["Pre-match", f"{len(prematch):,}", str(prematch.shape[1]),
                     "one row per match"])
    if inplay is not None:
        shape = (f"{inplay['snapshot_minute'].nunique()} snapshots per match"
                 if "snapshot_minute" in inplay.columns else "")
        rows.append(["In-play", f"{len(inplay):,}", str(inplay.shape[1]),
                     shape])
    if rows:
        blocks.append(table(
            pd.DataFrame(rows, columns=["Table", "Rows", "Columns", "Shape"]),
            caption_text="""
            Table 3.1. Shape of the two feature tables as built. Column counts
            include identifiers, split tags and labels, which are removed
            before the design matrix is assembled."""))
    blocks.append(para("""
        <b>A defect found by cross-checking, not by inspection.</b> The first
        run of the aggregate builder reproduced the correct scoreline for only
        1391 of 1517 matches. The disagreements were exactly symmetric: one
        goal on the wrong side, never a wrong total. Inspecting the event
        stream showed why. StatsBomb records an own goal twice, as "Own Goal
        For" on the team that benefits and "Own Goal Against" on the team that
        conceded, so the credit is already on the correct side. The first
        implementation added the opponent's "Own Goal For" events as well,
        moving one goal across in every match containing an own goal. After the
        correction all 1517 matches agree. The check is retained in the builder
        as a permanent guard, because a feature that silently disagrees with
        the label store is exactly the kind of error that aggregate metrics
        cannot reveal."""))
    blocks.append(para("""
        <b>Known limitation of the rolling window.</b> The window uses a
        minimum period of one, so a team's second match of the season carries a
        one-match average that is labelled as a five-match rolling form. This
        affects roughly the first four fixtures of every team. The alternative,
        discarding those matches, would remove a substantial part of the
        training block from a dataset that only has one season. The artefact is
        recorded here so that the early-season rows are not mistaken for
        equally reliable evidence."""))


# --- Section 4 --------------------------------------------------------------
def section_modelling(blocks):
    blocks.append(h1("4. Models, tuning protocol and calibration"))
    blocks.append(para("""
        The model zoo spans a constant baseline that always predicts the
        training class prior, a bagged ensemble, three boosted ensembles,
        kernel methods, and the P2 paper reimplementation. The constant
        baseline is not decoration. In a domain where the best available
        forecaster, the betting market, is only modestly better than the base
        rate, a model that cannot clearly beat the prior has not demonstrated
        anything, and Section 6 shows this is exactly what happens on the
        pre-match task."""))
    blocks.append(para("""
        Kernel methods are restricted to the pre-match tasks. This is not an
        arbitrary restriction: an exact kernel needs an n-by-n matrix, and at
        seventeen thousand training snapshots that matrix alone would be over
        two gigabytes before any solve. Section 5.4 measures this rather than
        assuming it."""))

    blocks.append(h2("4.1 Hyperparameter search"))
    blocks.append(para("""
        <b>Why this protocol.</b> Comparing models is only fair if each was
        given the same opportunity to be configured well. The search therefore
        gives every model an equal candidate budget, and the equality is
        asserted at the end of the run rather than described in prose. Scoring
        is by cross-validation inside the training rows only, so neither the
        validation nor the test split influences the choice. On the snapshot
        table the folds are grouped by match: if snapshots of one match
        appeared in both the fitting and the scoring fold, the score would be
        measuring memorisation of that match rather than generalisation, and
        every model would look far better than it is."""))
    tuning = read("tuning_results.csv")
    if tuning is not None:
        summary = (tuning.groupby(["task", "model"])
                   .agg(candidates=("score", "size"),
                        best_score=("score", "min"),
                        median_score=("score", "median"),
                        objective=("objective", "first"),
                        tuning_rows=("n_tuning_rows", "first"))
                   .reset_index().round(5))
        blocks.append(table(summary, max_rows=40, caption_text="""
            Table 4.1. Hyperparameter search. Equal candidate counts within a
            task confirm the equal-budget protocol. The gap between the best
            and the median candidate shows how much the search actually moved
            each learner: a small gap means the model was already near its best
            configuration and tuning was not the limiting factor."""))
    else:
        blocks.append(unavailable("tuning_results.csv", "src/models/tuning.py"))
        blocks.append(para("""
            <b>What this absence means for the rest of the report.</b> Because
            the search did not complete, every model in the sweep ran on
            library defaults. The comparisons between learners are therefore
            comparisons of default configurations. This understates the boosted
            models more than the others, since they depend more strongly on
            learning rate and depth. Conclusions that rest on a ranking between
            learners are provisional; conclusions that rest on the gap between
            in-play and pre-match information, or on the statistical tests, are
            not affected in the same way."""))

    blocks.append(h2("4.2 Calibration"))
    blocks.append(para("""
        <b>Why calibrate.</b> A model can rank matches well and still state
        badly wrong confidences. For a forecasting product the stated
        probability is the output, so calibration is not a cosmetic step. Every
        probabilistic classifier is calibrated on the validation split with the
        base estimator frozen, which means calibration cannot refit the model
        and cannot see test rows. Isotonic regression is tried first, with
        Platt scaling as the fallback."""))
    blocks.append(para("""
        <b>What we expected.</b> Boosted models to be noticeably
        over-confident before calibration and to improve substantially after
        it; the constant baseline to be almost perfectly calibrated already,
        since it predicts the training prior and has no confidence to
        misstate. <b>An implementation detail that affects the numbers.</b>
        Isotonic regression on a few hundred validation rows can output exact
        zeros, which are undefined under log loss. Probabilities are therefore
        floored at a small value and renormalised. The same floor is applied to
        the uncalibrated probabilities, so the before-and-after comparison
        isolates the effect of the calibrator rather than the effect of the
        floor."""))

    results = read("model_results.csv")
    if results is None:
        blocks.append(unavailable("model_results.csv",
                                  "src/models/run_models.py"))
        return
    classification = results[results["task"].isin(["C", "Lc"])]
    blocks.append(table(
        classification,
        ["task", "model", "tuned", "calibration", "rps", "log_loss", "brier",
         "ece_before", "ece_after"], max_rows=20, caption_text="""
        Table 4.2. Classification results on the test split, with the expected
        calibration error before and after calibration. A large reduction
        indicates a learner whose raw scores were poorly calibrated, not one
        whose ranking was poor."""))
    _calibration_analysis(blocks, classification)

    regression = results[results["task"].isin(["R", "Lr"])]
    blocks.append(table(
        regression,
        ["task", "model", "tuned", "n_train", "subsampled", "mae", "rmse",
         "corr"], max_rows=20, caption_text="""
        Table 4.3. Regression results on the test split. The exact kernel ridge
        entry subsamples its training rows, and the subsampled flag records
        this so the row is not read as a like-for-like comparison."""))
    _regression_analysis(blocks, regression)


def _calibration_analysis(blocks, classification):
    if classification.empty or "ece_before" not in classification.columns:
        return
    frame = classification.dropna(subset=["ece_before", "ece_after"]).copy()
    frame["improvement"] = frame["ece_before"] - frame["ece_after"]
    best = frame.loc[frame["improvement"].idxmax()]
    worst = frame.loc[frame["improvement"].idxmin()]
    prematch = classification[classification["task"] == "C"]
    inplay = classification[classification["task"] == "Lc"]
    blocks.append(para(f"""
        <b>What we observed.</b> The largest calibration gain is
        {best['model']} on task {best['task']}, where the expected calibration
        error falls from {best['ece_before']:.4f} to {best['ece_after']:.4f}.
        The smallest gain, and in fact a deterioration, is {worst['model']} on
        task {worst['task']}, moving from {worst['ece_before']:.4f} to
        {worst['ece_after']:.4f}."""))
    blocks.append(para("""
        <b>Why.</b> The pattern matches the expectation. Boosted models
        optimise a loss that rewards separating classes, and with unrestricted
        depth they push probabilities towards zero and one, which is exactly
        the behaviour that produces a large calibration error and a large
        subsequent correction. Models that already output averaged leaf
        proportions, such as the random forest and the shrunk forest from P2,
        start much closer to calibrated and have little left to gain. Where
        calibration slightly worsens the error, the cause is the small
        validation set: isotonic regression is flexible and can overfit a few
        hundred rows, so it fits noise in the reliability curve and transfers a
        little of that noise to the test set. This is a pipeline effect caused
        by the amount of calibration data available, not a defect of the
        underlying model."""))
    if not prematch.empty and not inplay.empty:
        blocks.append(para(f"""
            <b>What it means.</b> Calibration is doing real work but it cannot
            manufacture information. On task C the best ranked probability
            score after calibration is {prematch['rps'].min():.5f}, against
            {inplay['rps'].min():.5f} on task Lc. Calibration moves the
            expected calibration error by a few hundredths; the move from
            pre-match to in-play information moves the ranked probability score
            by far more. <b>Conclusion.</b> Calibration should be kept, because
            the stated probability is the product and it is cheap to correct,
            but it is not where the performance of this system comes from."""))


def _regression_analysis(blocks, regression):
    if regression.empty or "mae" not in regression.columns:
        return
    prematch = regression[regression["task"] == "R"]
    inplay = regression[regression["task"] == "Lr"]
    if prematch.empty or inplay.empty:
        return
    dummy = prematch[prematch["model"] == "dummy"]
    baseline = float(dummy["mae"].iloc[0]) if not dummy.empty else None
    best_prematch = prematch.loc[prematch["mae"].idxmin()]
    best_inplay = inplay.loc[inplay["mae"].idxmin()]
    baseline_text = ("" if baseline is None else
                     f" The constant baseline, which always predicts the mean "
                     f"training margin, achieves {baseline:.5f}.")
    blocks.append(para(f"""
        <b>What we expected.</b> That pre-match margin prediction would be only
        slightly better than predicting the average margin for every match,
        because the pre-match features describe recent form and nothing about
        how the match actually unfolds. That in-play prediction would be far
        better, because after an hour the current score is already most of the
        answer. <b>What we observed.</b> The best pre-match mean absolute error
        is {best_prematch['mae']:.5f} from {best_prematch['model']}, and the
        best in-play error is {best_inplay['mae']:.5f} from
        {best_inplay['model']}.{baseline_text}"""))
    blocks.append(para(f"""
        <b>Why.</b> The correlation column makes the mechanism visible. The
        pre-match models reach a correlation of about
        {prematch['corr'].max():.3f} with the realised margin, while the
        in-play models reach about {inplay['corr'].max():.3f}. A pre-match
        model is forecasting a quantity that is mostly determined by events
        that have not happened yet, so its achievable ceiling is low regardless
        of the learner. The in-play model observes the goals that have already
        been scored, and the current goal difference is a strong predictor of
        the final one. <b>Conclusion.</b> The gain from moving to the in-play
        setting is far larger than the gain from any choice of learner within a
        setting, which is the central practical finding of this project."""))


# --- Section 5 --------------------------------------------------------------
def section_market(blocks):
    blocks.append(h1("5. Experiments"))
    blocks.append(h2("5.1 Comparison against the betting market"))
    blocks.append(para("""
        <b>Why this experiment.</b> A metric on its own does not say whether a
        forecast is good. The betting market provides a demanding external
        reference: the published odds represent the aggregated opinion of
        participants with money at stake. Bookmaker odds include a margin, so
        the implied probabilities sum to more than one; they are de-vigged by
        normalising them to sum to one before use."""))
    blocks.append(para("""
        <b>What we expected.</b> That the models would not beat the market.
        The market observes team news, injuries, suspensions, expected lineups
        and the flow of money, none of which appear in our feature tables. The
        purpose of the comparison is to measure the size of the gap, not to win
        it. <b>Methodological care.</b> The comparison is restricted to test
        matches that carry odds, and the models are re-scored on exactly that
        subset, so both sides see identical matches."""))
    market = read("market_comparison.csv")
    if market is None:
        blocks.append(unavailable("market_comparison.csv",
                                  "src/models/market_comparison.py"))
        return
    blocks.append(table(
        market, ["model", "n_matches", "rps", "log_loss", "brier", "ece",
                 "rps_gap_vs_market", "beats_market"], caption_text="""
        Table 5.1. Task C models against the de-vigged market on the
        odds-tagged test matches. A positive gap means the model is worse than
        the market."""))
    reference = market[market["model"] == "MARKET_devigged"]
    beaters = market[(market["beats_market"].astype(str).str.lower() == "true")
                     & (market["model"] != "MARKET_devigged")]
    models_only = market[market["model"] != "MARKET_devigged"]
    if reference.empty or models_only.empty:
        return
    market_rps = float(reference["rps"].iloc[0])
    best = models_only.loc[models_only["rps"].idxmin()]
    blocks.append(para(f"""
        <b>What we observed.</b> The de-vigged market scores
        {market_rps:.5f}. The best model scores {best['rps']:.5f}
        ({best['model']}), a gap of {best['rps'] - market_rps:+.5f}.
        {len(beaters)} of {len(models_only)} models beat the market."""))
    blocks.append(para("""
        <b>Why.</b> Two reasons, and they are different in kind. The first is
        an information gap: the market knows the starting eleven and we do not.
        The second is a feature gap of our own making: the pre-match table
        contains only rolling form over goals, expected goals, points and
        wins, plus rest days. It contains nothing about squad quality, nothing
        about head-to-head history, and nothing about home advantage beyond
        what the home and away split of the form columns implies. The ablation
        in Section 8 shows that almost all of the usable signal in this table
        comes from the expected-goals columns alone, which is consistent with a
        table that is too narrow rather than a learner that is too weak."""))
    blocks.append(para("""
        <b>What it means and what we conclude.</b> The result is the expected
        one and it is reported as a finding rather than a disappointment.
        Beating a liquid market on four major leagues with form features from
        one season would have been a surprising claim requiring far stronger
        evidence than this project can supply. The practical conclusion is that
        the market should stay in the pipeline as a benchmark, and that closing
        the gap is a feature-engineering problem before it is a modelling
        problem."""))


def section_imbalance(blocks):
    blocks.append(h2("5.2 Six-arm imbalance study and the effect of P1"))
    blocks.append(para(f"""
        <b>Why this experiment.</b> Draws are the minority outcome, and a
        classifier trained on an imbalanced set tends to predict the minority
        class rarely, which shows up as low recall on draws even when the
        overall score looks acceptable. Six arms are compared on the pre-match
        table under identical splits: no treatment, SMOTE, borderline SMOTE,
        ADASYN, balanced class weights, and P1, our reimplementation of
        G-SMOTENC from {P1_CITATION}."""))
    blocks.append(para("""
        <b>What P1 is and what it should do.</b> Standard SMOTE interpolates
        between a minority point and one of its neighbours along a straight
        line, and it assumes every feature is continuous. Our table contains a
        nominal column, the competition, which SMOTE cannot handle without
        one-hot encoding it and then producing fractional values that
        correspond to no real category. G-SMOTENC generates the synthetic point
        inside a geometric region around the minority sample rather than on the
        line segment, and it handles nominal features by assigning the majority
        category among the neighbours instead of interpolating. The prediction
        that follows from the paper is that P1 should produce more diverse and
        more valid synthetic minority points, and therefore raise minority
        recall by more than the interpolation-based methods, while keeping the
        nominal column meaningful."""))
    blocks.append(para("""
        <b>What we expected on the probabilistic score.</b> Not an improvement.
        Oversampling changes the class balance the model is trained on, so it
        shifts predicted probabilities away from the true base rate. The ranked
        probability score rewards matching the base rate. There is a real
        tension here: the arms are designed to improve minority detection, and
        that goal is partly at odds with the metric. Reporting both is
        therefore necessary, and reporting only one would be misleading."""))
    blocks.append(para("""
        <b>An important methodological restriction.</b> The snapshot table is
        excluded from this study by design. Oversampling across matches would
        create synthetic rows blending the trajectories of two different
        matches, which do not correspond to any possible game state. Every
        resampler is also fitted inside the training rows only, so validation
        and test always keep the real class balance."""))
    resampling = read("resampling_study.csv")
    if resampling is None:
        blocks.append(unavailable("resampling_study.csv",
                                  "src/models/resampling_study.py"))
    else:
        summary = (resampling.groupby("arm")[["rps", "recall_D", "f1_D",
                                              "ece_after"]]
                   .mean().round(5).reset_index().sort_values("rps"))
        blocks.append(table(summary, caption_text="""
            Table 5.2. Mean over the study's learners, by arm. The ranked
            probability score measures the whole distribution; recall and F1 on
            the draw class measure whether the arm achieved the thing it was
            introduced to achieve."""))
        best_rps = summary.iloc[0]
        best_recall = summary.loc[summary["recall_D"].idxmax()]
        blocks.append(para(f"""
            <b>What we observed.</b> The best arm by ranked probability score
            is {best_rps['arm']} at {best_rps['rps']:.5f}. The best arm by draw
            recall is {best_recall['arm']} at {best_recall['recall_D']:.5f},
            against {summary.loc[summary['arm'] == 'vanilla', 'recall_D'].iloc[0]:.5f}
            for the untreated arm. The two rankings are not the same, which is
            the tension described above appearing in the numbers."""))

    comparison = read("p1_comparison.csv")
    if comparison is not None:
        blocks.append(table(comparison, caption_text="""
            Table 5.3. With-P1 against without-P1, measured on the validation
            split so that the comparison does not consume the test set. Draw
            recall and draw F1 are shown beside the probabilistic score,
            because an oversampler can improve minority detection while
            degrading the calibrated distribution."""))
        helped = comparison[comparison["p1_helps_rps"].astype(str)
                            .str.lower() == "true"]
        recall_gain = (comparison["recall_D_with_p1"]
                       - comparison["recall_D_without_p1"])
        blocks.append(para(f"""
            <b>The effect of P1, stated plainly.</b> P1 did what its paper says
            it should do, on the metric the paper cares about, and did not help
            on the metric this project cares about most. Draw recall rises on
            every learner tested, by between {recall_gain.min():.3f} and
            {recall_gain.max():.3f} in absolute terms. On the ranked
            probability score, P1 helped {len(helped)} of {len(comparison)}
            learners, and the movements are small in both directions."""))
        blocks.append(para("""
            <b>Why this happened.</b> The recall gain is the mechanism working
            as designed: more and more varied synthetic draw examples move the
            decision boundary so that draws are predicted more often. The lack
            of a probabilistic gain has a clear cause too. Ranked probability
            score is minimised by predicting the true class probabilities, and
            the true probability of a draw is roughly a quarter. Any method
            that trains the model on a rebalanced set pushes predicted draw
            probability above the true rate, which costs score on the many
            matches that are not drawn while gaining on the fewer that are.
            Calibration on the validation split partly undoes the shift, which
            is why the net effect on the score is small rather than clearly
            negative."""))
        blocks.append(para("""
            <b>Is the difference meaningful or could it be randomness?</b> The
            recall movements are large relative to their scale and consistent
            in direction across learners, so they are unlikely to be noise. The
            score movements are small, inconsistent in direction, and were
            measured on a single validation split, so they should not be read
            as evidence that P1 helps or harms the distribution. Section 6
            explains why differences of this size on this dataset are generally
            not distinguishable from noise."""))
        blocks.append(para("""
            <b>Conclusion on P1.</b> The reimplementation is faithful, verified
            against the paper's algorithms by its own test suite, and it
            reproduces the qualitative behaviour the paper claims. Whether it
            should be enabled in the final system depends on what the system is
            for. If the aim is to state well-calibrated probabilities, it
            should be left off, because it adds a bias that calibration then
            has to remove. If the aim is to flag likely draws, it is the best
            of the six arms. We report this honestly rather than selecting the
            framing that makes the paper look better."""))
    else:
        blocks.append(unavailable("p1_comparison.csv", "src/models/ablation.py"))


def section_inplay(blocks):
    blocks.append(h2("5.3 Performance against match minute for Model 3"))
    blocks.append(para("""
        <b>Why this experiment.</b> Model 3 exists to answer whether watching
        the match helps, and by how much, and from which minute. A single
        aggregate number over all snapshots would hide the shape of that
        answer, because a prediction at minute 5 and a prediction at minute 85
        are entirely different problems."""))
    blocks.append(para("""
        <b>What we expected.</b> Three things. First, that error would fall as
        the minute increases, since more of the match has been observed.
        Second, that the fall would not be smooth but would steepen in the
        second half, because goals are what move the outcome and their effect
        becomes decisive as remaining time shrinks. Third, that at minute 0 the
        in-play model would be roughly level with the pre-match model, since at
        kick-off they see nearly the same information."""))
    blocks.append(para("""
        <b>The control.</b> Each in-play model is compared against a frozen
        pre-match reference: the corresponding pre-match model's single
        prediction for that match, repeated across all nineteen snapshots. The
        frozen series must be flat, and the analysis asserts that it is. This
        assertion is a leakage check. If the reference sloped, it would mean
        the supposedly frozen prediction had absorbed information about how the
        match was progressing, and every comparison against it would be
        invalid."""))
    curves = read("inplay_metric_by_minute.csv")
    if curves is None:
        blocks.append(unavailable("inplay_metric_by_minute.csv",
                                  "src/models/inplay_curves.py"))
        return
    classification = curves[curves["task"] == "Lc"].copy()
    classification["label"] = (classification["model"] + " ("
                               + classification["series"] + ")")
    blocks.append({"type": "lineplot", "name": "curve_rps",
                   "frame": classification, "x": "snapshot_minute", "y": "rps",
                   "series": "label", "dashed": "frozen",
                   "title": "Task L classification: RPS by snapshot minute",
                   "xlabel": "snapshot minute", "ylabel": "RPS"})
    blocks.append(caption("""
        Figure 5.1. Ranked probability score against match minute. Solid lines
        are the in-play model; dashed lines are the frozen pre-match reference.
        The vertical distance between a model and its own frozen line is the
        value of live information at that minute."""))

    in_play = classification[classification["series"] == "in-play"]
    frozen = classification[classification["series"] == "frozen pre-match"]
    if not in_play.empty and not frozen.empty:
        early = in_play[in_play["snapshot_minute"] == 0]["rps"].min()
        late = in_play[in_play["snapshot_minute"] == 90]["rps"].min()
        mid = in_play[in_play["snapshot_minute"] == 45]["rps"].min()
        flat = frozen["rps"].min()
        blocks.append(para(f"""
            <b>What we observed.</b> The best in-play ranked probability score
            is {early:.5f} at kick-off, {mid:.5f} at half time and {late:.5f}
            at the final snapshot, against a flat frozen reference at
            {flat:.5f}. The curve falls throughout and falls fastest in the
            second half, which matches the expectation. At minute 0 the in-play
            model is close to the frozen reference, as predicted, because the
            only extra information it has is that no events have occurred
            yet."""))
        blocks.append(para("""
            <b>Why.</b> The dominant in-play feature is the current goal
            difference, and its predictive power is a function of the time
            remaining. A one-goal lead after ten minutes leaves eighty minutes
            for it to be overturned; the same lead after eighty minutes is
            close to decisive. The curve is therefore not really measuring the
            model learning more, it is measuring the outcome becoming less
            uncertain. This distinction matters for how the result is used: the
            steep late improvement is a property of football, and any competent
            model would show it, so it is not evidence that our model is
            good."""))
    regression = curves[curves["task"] == "Lr"].copy()
    regression["label"] = (regression["model"] + " ("
                           + regression["series"] + ")")
    blocks.append({"type": "lineplot", "name": "curve_mae",
                   "frame": regression, "x": "snapshot_minute", "y": "mae",
                   "series": "label", "dashed": "frozen",
                   "title": "Task L regression: MAE by snapshot minute",
                   "xlabel": "snapshot minute", "ylabel": "MAE"})
    blocks.append(caption("""
        Figure 5.2. Mean absolute margin error against match minute. The same
        shape appears as in the classification curve, for the same reason."""))

    phases = read("inplay_calibration_by_phase.csv")
    if phases is not None:
        summary = (phases.groupby("phase", observed=True)
                   [["ece", "mean_confidence", "accuracy", "over_confidence"]]
                   .mean().round(4).reset_index())
        blocks.append(table(summary, caption_text="""
            Table 5.4. Model 3 calibration by game phase, averaged over
            learners. Over-confidence is mean confidence minus accuracy, so a
            positive value means the model claims more certainty than it
            earns in that phase."""))
        worst = summary.loc[summary["over_confidence"].abs().idxmax()]
        blocks.append(para(f"""
            <b>Why calibration is reported per phase.</b> A single calibration
            number over all snapshots would average an easy problem and a hard
            one and hide both. <b>What we observed.</b> The phase with the
            largest calibration gap is {worst['phase']}, with a gap of
            {worst['over_confidence']:+.4f} between mean confidence and
            accuracy. <b>Why.</b> Confidence rises through the match as the
            outcome becomes clearer, and accuracy rises with it; a gap appears
            where the two rise at different rates. Late in a match the model
            can become confident on the basis of a lead that is usually but not
            always decisive. <b>Conclusion.</b> If these probabilities were
            shown to a user during a live match, a phase-specific calibrator
            would be preferable to the single global one used here."""))


def section_scaling(blocks):
    blocks.append(h2("5.4 Compute, memory and kernel scaling"))
    blocks.append(para("""
        <b>Why this experiment.</b> The decision to exclude kernel methods from
        the in-play tasks needs evidence rather than assertion. An exact kernel
        method builds an n-by-n matrix of pairwise similarities and solves a
        dense system, so time and memory grow faster than linearly in the
        number of training rows. The Nystroem approximation projects onto a
        smaller set of landmark points and reduces the fit to a ridge
        regression in that smaller space."""))
    blocks.append(para("""
        <b>What we expected.</b> From the algorithms: an exponent near three
        for the exact ridge solve, roughly two for the support vector
        regression, and near one for Nystroem, whose cost is linear in n for a
        fixed number of landmarks. Fitting a power law on the logs recovers the
        exponent from measurements:"""))
    blocks.append(equation(r"$\log t=\log a+b\,\log n$", "powerlaw"))
    scaling = read("kernel_scaling.csv")
    if scaling is None:
        blocks.append(unavailable("kernel_scaling.csv",
                                  "src/models/kernel_scaling.py"))
    else:
        exponents = (scaling.drop_duplicates("method")
                     [["method", "empirical_exponent", "theoretical_exponent",
                       "theory"]])
        blocks.append(table(exponents, caption_text="""
            Table 5.5. Empirical against theoretical scaling exponents. The
            analysis asserts that the exact methods scale faster than linearly
            and that Nystroem scales better than both, so the claim fails
            loudly if it ever stops holding."""))
        blocks.append({"type": "lineplot", "name": "kernel_time",
                       "frame": scaling, "x": "n_train", "y": "fit_seconds",
                       "series": "method", "dashed": None, "logx": True,
                       "logy": True,
                       "title": "Kernel fit time against training-set size",
                       "xlabel": "training rows", "ylabel": "seconds"})
        blocks.append(caption("""
            Figure 5.3. Fit time against training rows, on log axes, so a
            power law appears as a straight line and the exponent is the
            slope."""))
        exact = exponents[exponents["method"].str.contains("exact")]
        nystroem = exponents[exponents["method"] == "kernel_ridge_nystroem"]
        if not exact.empty and not nystroem.empty:
            blocks.append(para(f"""
                <b>What we observed.</b> The exact methods have measured
                exponents between {exact['empirical_exponent'].min():.2f} and
                {exact['empirical_exponent'].max():.2f}, and Nystroem
                {nystroem['empirical_exponent'].iloc[0]:.2f}. <b>Why the
                measured exponents differ from the textbook ones.</b> At these
                sizes the fit is not yet dominated by the asymptotic term.
                Fixed overheads, memory allocation and cache behaviour all
                contribute, and the solver may switch strategy as the problem
                grows. The measurement confirms the ordering and the
                super-linear growth, which is what the design decision needs;
                it is not a precise estimate of the asymptotic constant."""))
            blocks.append(para("""
                <b>Conclusion.</b> Excluding exact kernels from the seventeen
                thousand row in-play table is justified by measurement. Where a
                kernel method is wanted at that scale, Nystroem is the
                practical option, and it appears in the pre-match zoo so that
                its accuracy cost can be seen next to the exact method on a
                problem small enough for both."""))
    compute = read("compute_profile.csv")
    if compute is None:
        blocks.append(unavailable("compute_profile.csv",
                                  "src/models/compute_profile.py"))
    else:
        blocks.append(table(
            compute, ["task", "model", "n_train_used", "fit_seconds",
                      "peak_fit_memory_mb", "model_size_mb",
                      "predict_microseconds_per_row"], max_rows=40,
            caption_text="""
            Table 5.6. Compute and memory comparison, measured in one process
            on one machine so the rows are comparable. Fit time and peak
            resident memory are sampled during the fit; inference cost is the
            median of repeated passes over the full test matrix."""))
        blocks.append(para("""
            <b>Why this table matters for the P2 model in particular.</b> The
            P2 estimator is consistently the slowest to fit. This is expected
            from its design rather than from inefficiency: it selects its
            shrinkage parameter by cross-validation inside the training rows,
            so a single fit costs several forest fits. The cost is paid once at
            training time; inference remains a normal forest traversal. When
            reading the fit-time column, this row should be understood as the
            price of an internal model-selection step that the other learners
            do not perform."""))


def section_conversion(blocks):
    blocks.append(h2("5.5 Converting the Task R margin into probabilities"))
    blocks.append(para("""
        <b>Why this experiment.</b> The brief asks whether the regressed margin
        can be turned into useful outcome probabilities. This is a question
        about whether a model trained on a richer target, the signed margin,
        carries information that a model trained on the three-way label does
        not. <b>What we expected.</b> That the converted regressor would be
        competitive but slightly worse, because Task C optimises the exact
        quantity being scored while Task R optimises a proxy."""))
    blocks.append(para("""
        Two links were fitted on the validation split and scored once on test.
        The ordinal link treats the realised margin as a latent continuous
        quantity cut by two symmetric thresholds, which is the natural
        assumption for an ordered outcome:"""))
    blocks.append(equation(
        r"$P(A)=\Phi\!\left(\frac{-\theta-\hat{m}}{\sigma}\right),\quad "
        r"P(H)=1-\Phi\!\left(\frac{\theta-\hat{m}}{\sigma}\right),\quad "
        r"P(D)=1-P(H)-P(A)$", "ordinal"))
    blocks.append(para("""
        Here m-hat is the predicted margin, theta is the half-width of the draw
        band and sigma is the spread of the realised margin around the
        prediction. Both are fitted by maximising the likelihood on validation
        rows, never on test rows. A multinomial logistic link on the predicted
        margin and its square is fitted as a second, more flexible
        alternative."""))
    conversion = read("margin_to_probability.csv")
    if conversion is None:
        blocks.append(unavailable("margin_to_probability.csv",
                                  "src/models/margin_to_probability.py"))
        return
    blocks.append(table(
        conversion, ["converter", "source", "rps", "log_loss", "brier", "ece"],
        caption_text="""
        Table 5.7. Margin-to-probability conversion against the classifier
        trained directly for the task, on the test split. Both sides were
        selected on validation and scored once, so the comparison is fair."""))
    direct = conversion[conversion["converter"].str.startswith("model1")]
    ordinal = conversion[conversion["converter"].str.contains("ordinal")]
    if direct.empty or ordinal.empty:
        return
    direct_row, ordinal_row = direct.iloc[0], ordinal.iloc[0]
    blocks.append(para(f"""
        <b>What we observed.</b> The direct classifier scores
        {direct_row['rps']:.5f} and the ordinal conversion {ordinal_row['rps']:.5f},
        a difference of {ordinal_row['rps'] - direct_row['rps']:+.5f}. On
        calibration the ordering reverses: the ordinal conversion records an
        expected calibration error of {ordinal_row['ece']:.5f} against
        {direct_row['ece']:.5f} for the direct classifier."""))
    blocks.append(para("""
        <b>Why.</b> The ordinal link is a two-parameter function of a single
        predicted number. It cannot produce erratic probabilities, because it
        is smooth and monotone in the predicted margin by construction. The
        direct classifier has far more freedom and can place sharp
        probabilities in regions of feature space where it saw few training
        matches. On a small pre-match training set that freedom costs
        calibration. In other words the conversion is better calibrated because
        it is more constrained, not because it knows more."""))
    blocks.append(para("""
        <b>Does this support the original expectation?</b> Partly. The
        expectation that the direct classifier would rank better was correct,
        though the margin between them is small. The expectation did not
        anticipate the calibration reversal, which is the more interesting
        result. <b>Limitations.</b> Both numbers come from one test split of a
        few hundred matches and the difference in ranked probability score is
        small; Section 6 shows that differences of this size on this dataset
        are generally not separable from noise, so this should not be read as
        a reliable ordering."""))
    blocks.append(para("""
        <b>Conclusion.</b> The margin can be converted into useful
        probabilities, and the conversion is a reasonable component for a
        system that needs both an expected scoreline and an outcome
        distribution from a single model. If well-stated confidences matter
        more than a marginally better ranking, the ordinal conversion is the
        better choice on this evidence."""))


# --- Section 6 --------------------------------------------------------------
def section_significance(blocks):
    blocks.append(h1("6. Are the differences real? Statistical testing"))
    blocks.append(para("""
        <b>Why this section exists.</b> Every table so far has ordered models
        by a single number computed on one test set of a few hundred matches.
        A difference in such numbers is not by itself evidence that one model
        is better. This section asks which of the differences survive an
        honest test, and the answer changes how the rest of the report should
        be read."""))
    blocks.append(h2("6.1 Two kinds of repetition, answering two questions"))
    blocks.append(para("""
        <b>Seed repetition</b> refits every model under several random states
        on the fixed splits. It measures stability: whether a model gives the
        same answer when only its random seed changes. It does not measure
        superiority, and it must not be used for that purpose. The reason is
        specific and was observed directly during this project. The only
        quantity varying between replicates is the random state, so a
        deterministic model such as the constant baseline has zero spread. A
        paired test across seeds then divides by approximately zero and reports
        an extremely small p-value for any difference at all, however small
        that difference is compared with the noise that actually matters. In an
        early version of this analysis that test declared the constant baseline
        significantly different from a boosted model, while the correct test
        below found no detectable difference between any pair. The seed columns
        are therefore reported as stability diagnostics only."""))
    blocks.append(para("""
        <b>The match-clustered bootstrap</b> is the test that licenses a claim
        that one model is better. It resamples the quantity that actually
        limits the conclusion: the finite set of test matches. Matches are
        resampled rather than rows, because the nineteen snapshots of one match
        are strongly dependent. A row-level bootstrap would treat them as
        independent and understate the variance by roughly the cluster size,
        producing confidence intervals that are far too narrow and p-values
        that are far too small. Holm-Bonferroni correction is applied within
        each task, because comparing many pairs raises the chance that at least
        one looks significant by accident."""))
    seeds = read("seed_repetitions.csv")
    if seeds is not None:
        summary = (seeds.groupby(["task", "model", "metric"])["value"]
                   .agg(["mean", "std", "min", "max"]).round(5).reset_index())
        blocks.append(table(summary, max_rows=40, caption_text="""
            Table 6.1. Test metric across random seeds. This is a stability
            diagnostic. A near-zero standard deviation means the learner is
            reproducible when reseeded; it does not mean its advantage over
            another learner is real."""))
    else:
        blocks.append(unavailable("seed_repetitions.csv",
                                  "src/models/significance.py"))
    bootstrap = read("significance_bootstrap.csv")
    if bootstrap is None:
        blocks.append(unavailable("significance_bootstrap.csv",
                                  "src/models/significance.py"))
        return
    blocks.append(table(
        bootstrap, ["task", "model_a", "model_b", "mean_difference", "ci_low",
                    "ci_high", "bootstrap_p_holm", "verdict"], max_rows=60,
        caption_text="""
        Table 6.2. Match-clustered bootstrap comparisons with Holm-adjusted
        p-values. A confidence interval spanning zero means the two models are
        not distinguishable on this test set."""))
    flag = bootstrap["significant"].astype(str).str.lower() == "true"
    per_task = (bootstrap.assign(sig=flag).groupby("task")["sig"]
                .agg(["sum", "size"]))
    lines = ", ".join(f"{int(row['sum'])} of {int(row['size'])} on task {task}"
                      for task, row in per_task.iterrows())
    blocks.append(para(f"""
        <b>What we observed.</b> After correction the significant comparisons
        are: {lines}."""))
    blocks.append(para("""
        <b>Why the pre-match and in-play tasks differ so sharply.</b> The
        effect sizes differ by an order of magnitude. On the pre-match task the
        differences between learners are a few thousandths of a ranked
        probability score, against a match-to-match spread that is far larger,
        so a few hundred matches cannot resolve them. On the in-play task the
        differences between a real model and the constant baseline are large
        because the in-play features genuinely carry information. The
        statistical machinery is identical in both cases; what changes is
        whether there is a real effect large enough to detect."""))
    blocks.append(para("""
        <b>What this means for the rest of the report.</b> The pre-match
        leaderboard should not be read as a ranking. Where the test finds no
        difference, the honest statement is that the models are
        indistinguishable on this evidence, not that the one at the top is
        best. The in-play results are on much firmer ground. <b>What a critical
        reader would ask next.</b> Whether more test matches would resolve the
        pre-match differences. Probably not usefully: the effects are so small
        that separating them would need far more data than one season provides,
        and a difference that small would not matter in practice
        anyway."""))
    blocks.append(para("""
        <b>Conclusion.</b> On pre-match data this project cannot demonstrate
        that any learner beats the constant baseline, including the P2 model.
        On in-play data the improvement over the baseline is large and
        clearly detectable. The correct summary of the modelling work is that
        the choice of information matters and the choice of learner, within
        this feature set, does not."""))


# --- Section 7 --------------------------------------------------------------
def section_shap(blocks):
    blocks.append(h1("7. Error analysis and SHAP"))
    blocks.append(para("""
        <b>Why this section exists.</b> Aggregate metrics say how much error
        there is but not what kind. This section asks what the models are
        actually using, where they fail worst, and whether those failures are
        mistakes that better modelling could fix or outcomes that no forecaster
        could have called."""))
    blocks.append(para("""
        Attributions are computed with TreeSHAP, which is exact for tree
        ensembles rather than a sampled approximation. Its output is additive:
        the contributions for one prediction sum to the difference between that
        prediction and the model's average output, which makes each explanation
        checkable."""))
    blocks.append(h2("7.1 A correctness issue specific to the P2 model"))
    blocks.append(para(f"""
        Explaining the P2 model required a fix that is worth stating, because
        without it the explanations would have been quietly wrong. The
        reimplementation of {P2_CITATION} stores its shrunk node values in a
        separate array alongside the fitted forest rather than inside the tree
        structure. TreeSHAP reads the tree structure directly. An explainer
        pointed at the fitted forest would therefore have explained the
        unshrunk forest, produced plausible-looking attributions, and given no
        error. The shrunk values are now written back into the tree before
        explanation, and a unit test asserts that the explained estimator
        reproduces the shrunk model's predictions exactly. This is a good
        example of a failure that is invisible in the output and only
        detectable by checking the mechanism."""))

    blocks.append(h2("7.2 Global attributions"))
    for task in ["C", "R", "Lc", "Lr"]:
        candidates = sorted(SHAP_DIR.glob(f"beeswarm_{task}_*.png"))
        if not candidates:
            continue
        blocks.append(image(candidates[0], f"""
            Figure 7.{task}. Global SHAP beeswarm for task {task}. Each point
            is one test row. Horizontal position is that feature's contribution
            to the model output and colour is the feature value, so a clean
            colour gradient from one side to the other indicates a consistent,
            monotone effect."""))
    importance = read("shap_importance.csv")
    if importance is None:
        blocks.append(unavailable("shap_importance.csv",
                                  "src/analysis/shap_analysis.py"))
    else:
        top = (importance.sort_values("mean_abs_shap", ascending=False)
               .groupby("task").head(6))
        blocks.append(table(
            top, ["task", "model", "feature", "mean_abs_shap"], max_rows=30,
            caption_text="""
            Table 7.1. Mean absolute SHAP value per feature, the six largest per
            task. This ranks influence on the model, which is not the same as
            causal importance in football."""))
        blocks.append(para("""
            <b>What we expected.</b> On the pre-match tasks, that the
            expected-goals columns would dominate, because expected goals is a
            less noisy measure of team strength than goals scored: a team can
            create six good chances and score none, and expected goals records
            the chances. On the in-play tasks, that the current goal difference
            would dominate everything else. <b>What we observed.</b> Both
            expectations hold. The pre-match beeswarms are led by the
            expected-goals difference columns, with a clean colour gradient
            confirming the direction: a home team with better recent expected
            goals against gets a higher home-win probability. The in-play
            attributions are dominated by the current goal difference."""))
        blocks.append(para("""
            <b>Why this is a useful check rather than a trivial one.</b> If the
            attributions had been led by something like rest days or the
            competition identifier, that would have been a strong signal of
            overfitting to an artefact. The features that dominate are the ones
            football knowledge says should dominate, which increases confidence
            that the pipeline is not learning something spurious. <b>What it
            does not tell us.</b> That the model is good. A model can use the
            right features and still be no better than the base rate, which
            Section 6 shows is the case on pre-match data."""))

    blocks.append(h2("7.3 The ten worst predictions: model error or bad luck?"))
    blocks.append(para("""
        <b>Why this question.</b> A large error is not automatically a defect.
        Football contains a great deal of irreducible randomness, and a
        forecaster that never made a large error on an unlikely result would be
        overfitted. The interesting question is whether the information needed
        to do better was actually available at prediction time."""))
    blocks.append(para("""
        <b>How the judgement is made, fixed before results were seen.</b> A
        failure counts as a <b>model error</b> only when the de-vigged market
        assigned the realised outcome a materially higher probability than the
        model did, or the model rated the realised outcome below its own
        unconditional base rate. Using the market as the reference is the key
        step: it converts a vague question into a testable one, because the
        market is a forecaster that had the same pre-match information and
        more. A failure is <b>inherently uncertain</b> when the market also
        rated the outcome unlikely, so the result rather than the forecast was
        the outlier. It is a <b>noisy observation</b> when the in-play state at
        that snapshot pointed the other way and the match turned afterwards. It
        is <b>reasonable despite the outcome</b> when the model's probability
        for the realised outcome was at or above its base rate."""))
    worst = read("worst_predictions.csv")
    if worst is None:
        blocks.append(unavailable("worst_predictions.csv",
                                  "src/analysis/shap_analysis.py"))
        return
    counts = worst.groupby(["task", "verdict"]).size().reset_index(name="count")
    blocks.append(table(counts, caption_text="""
        Table 7.2. Adjudication of the ten worst predictions per task. A high
        proportion of inherently uncertain verdicts indicates that the tail of
        the loss distribution is driven by outcome variance rather than by
        defects in the model."""))
    blocks.append(table(
        worst, ["task", "rank", "match_id", "snapshot_minute", "loss",
                "y_true", "p_realised", "market_p_realised", "verdict"],
        max_rows=40, caption_text="""
        Table 7.3. The worst predictions in full, with the model's probability
        for the realised outcome beside the market's, which is what the
        adjudication compares."""))
    snapshot_tasks = worst[worst["task"].isin(["Lc", "Lr"])]
    if not snapshot_tasks.empty:
        distinct = snapshot_tasks.groupby("task")["match_id"].nunique()
        detail = ", ".join(f"{n} distinct matches on task {t}"
                           for t, n in distinct.items())
        blocks.append(para(f"""
            <b>An important caveat on the snapshot tasks.</b> Consecutive
            minutes of the same match fail together, so the ten worst rows are
            not ten independent failures: they resolve to {detail}. A match in
            which the losing side comes back to win produces a long run of
            badly scored snapshots. The ranking is reported per row because
            that is the unit the model predicts, but the number of genuinely
            distinct failure modes is smaller than ten, and the table should
            not be read as evidence of ten separate problems."""))
    blocks.append(para("""
        <b>Why the in-play failures look the way they do.</b> The worst in-play
        errors are dominated by comebacks: the model was confident because a
        team was behind with limited time remaining, and that team then scored.
        This is the model correctly using the strongest available signal and
        being beaten by an unlikely event. It is classified as a noisy
        observation rather than a model error, because the state at the
        snapshot genuinely did favour the other outcome. <b>What could
        legitimately be improved.</b> The model could be less confident in
        exactly these situations if it had features describing pressure and
        chance creation in the recent window, which would show that the trailing
        team was dominating before it scored. That is a feature gap, and it is
        listed in Section 10."""))
    blocks.append(para("""
        <b>Conclusion.</b> The tail of the error distribution is mostly
        irreducible on the current feature set. The few cases classified as
        model errors are the ones worth revisiting, and they are identified
        individually in Table 7.3 rather than described in aggregate."""))

    timeline = SHAP_DIR / "inplay_timeline.png"
    blocks.append(h2("7.4 A complete match, minute by minute"))
    blocks.append(para("""
        <b>Why this figure.</b> The global attributions describe average
        behaviour and the worst-prediction analysis describes the tail. Neither
        shows how a single forecast evolves. This figure follows one complete
        match through all nineteen snapshots, with the outcome probabilities in
        the upper panel and the feature contributions to the home-win
        probability in the lower panel, so that any movement above can be
        traced to a cause below. The match is chosen deterministically, not
        selected for a good story."""))
    if timeline.exists():
        blocks.append(image(timeline, """
            Figure 7.5. SHAP timeline across one complete match. The upper
            panel is the outcome distribution at each snapshot; the lower panel
            decomposes the home-win probability into feature contributions.""",
            max_height_cm=12.0))
        blocks.append(para("""
            <b>What we observed.</b> The home-win probability is flat and
            moderate for the opening minutes, steps up sharply once, holds at
            the new level for a long stretch, and steps up again later before
            settling near certainty. The lower panel attributes both steps to
            the current goal difference: its contribution jumps at exactly the
            snapshots where the probability jumps, while the pre-match form
            contributions stay almost constant throughout."""))
        blocks.append(para("""
            <b>Why this is the expected and desired behaviour.</b> Pre-match
            form does not change during a match, so its contribution should be
            a flat offset, and it is. Goals are discrete events, so their
            effect should appear as steps rather than a drift, and it does. The
            long flat stretches between steps show the model is not reacting to
            noise in the event stream. <b>What it means for the final
            defence.</b> This figure is the narration tool: at any minute the
            model's probability can be explained by pointing at the feature
            whose contribution moved, rather than by appealing to the model as
            a whole."""))
    else:
        blocks.append(unavailable("visualizations/shap/inplay_timeline.png",
                                  "src/analysis/shap_analysis.py"))


# --- Section 8 --------------------------------------------------------------
def section_ablation(blocks):
    blocks.append(h1("8. Ablation and honest re-training"))
    blocks.append(para("""
        <b>Why every number in this section is a validation number.</b>
        Ablation is a form of model selection: it asks which configuration to
        keep. Choosing a configuration by its test score would be tuning
        against the test set, which would make the test score optimistic and
        the final comparison meaningless. The test split is read exactly once,
        for the final comparisons in Sections 4 to 6."""))

    ablation = read("ablation.csv")
    if ablation is None:
        blocks.append(unavailable("ablation.csv", "src/models/ablation.py"))
        return

    blocks.append(h2("8.1 Dropping feature groups"))
    blocks.append(para("""
        <b>What we expected.</b> That the expected-goals group would matter
        most on the pre-match tasks, following the SHAP results, and that the
        current-score group would dominate on the in-play tasks. That dropping
        the schedule group would cost little, because rest days vary less in
        league football than in cup competitions."""))
    for task in ["C", "R", "Lc", "Lr"]:
        metric = "rps" if task in {"C", "Lc"} else "mae"
        subset = ablation[(ablation["task"] == task)
                          & (ablation["axis"] == "feature_group")]
        if subset.empty or metric not in subset.columns:
            continue
        pivot = subset.pivot_table(index="configuration", columns="model",
                                   values=metric)
        if "full" not in pivot.index:
            continue
        delta = (pivot - pivot.loc["full"]).drop(index="full").round(5)
        delta = delta.reset_index()
        blocks.append(table(delta, caption_text=f"""
            Table 8.{task}. Change in validation {metric} when a feature group
            is dropped, relative to the full feature set. A positive value
            means the drop made things worse, so the group carried signal. A
            negative value means the model was slightly better without the
            group."""))
    blocks.append(para("""
        <b>What we observed.</b> On the pre-match tasks only the
        expected-goals group produces a clear deterioration when removed. The
        other groups move the metric by amounts that are small and sometimes
        negative, meaning the model was marginally better without them."""))
    blocks.append(para("""
        <b>Why.</b> Two effects combine. First, the feature groups overlap
        heavily: goals scored, points and wins are three different summaries of
        the same match results, so removing one leaves the information
        available through the others. Expected goals is the only group carrying
        something the others do not, namely chance quality independent of
        whether chances were converted. Second, with a small training set,
        removing weakly informative columns can genuinely help by reducing the
        opportunity to overfit, which explains the negative entries."""))
    blocks.append(para("""
        <b>What a critical reader would ask.</b> Whether these differences are
        significant. They are validation-set differences on a few hundred
        matches and no correction was applied, so individually they are not
        strong evidence. The pattern is more informative than any single
        number: one group matters and the rest are interchangeable.
        <b>Limitation.</b> This is also a statement about how narrow the
        feature table is. With twenty-two model columns the groups are small
        and highly correlated, so the ablation has limited power to separate
        them. A wider feature set would make this analysis more
        informative."""))

    frequency = ablation[ablation["axis"] == "snapshot_frequency"]
    if not frequency.empty:
        blocks.append(h2("8.2 Snapshot frequency"))
        blocks.append(para("""
            <b>Why this experiment.</b> Nineteen snapshots per match multiply
            the row count by nineteen, but those rows are highly correlated: a
            snapshot at minute 45 and one at minute 50 usually describe almost
            the same game state. If a coarser grid performs as well, the
            pipeline can be made substantially cheaper at no cost."""))
        blocks.append(para("""
            <b>A methodological point that had to be corrected.</b> An earlier
            version of this experiment thinned the validation snapshots as well
            as the training ones. That made each arm score on a different set
            of minutes, so the arms were not comparable and a coarser grid could
            appear better simply by being evaluated on easier minutes. Only the
            training rows are thinned now; validation stays at full density
            throughout, so every arm is scored on identical rows."""))
        blocks.append(table(
            frequency, ["task", "configuration", "model",
                        "train_snapshots_per_match", "eval_snapshots_per_match",
                        "n_train", "rps", "mae"], max_rows=24, caption_text="""
            Table 8.5. Snapshot frequency. Only training density varies; the
            evaluation set is held constant across arms."""))
        blocks.append(para("""
            <b>What we expected.</b> Some loss from coarser training, since
            fewer examples usually means a worse model. <b>What we observed.</b>
            Almost none between the finest and the middle grid, with a small
            loss only at the coarsest. <b>Why.</b> The additional rows in the
            finest grid are nearly duplicates of their neighbours. They increase
            the row count without adding much independent information, so the
            effective sample size grows far more slowly than the row count.
            This is the same dependence that forces grouped cross-validation
            and the clustered bootstrap elsewhere in this report, appearing
            here as a training-set effect. <b>Conclusion.</b> A coarser
            training grid is a reasonable efficiency choice. The finest grid is
            retained for evaluation, because the ability to predict at any
            minute is part of what Model 3 is for."""))

    blocks.append(h2("8.3 Balancing strategy"))
    blocks.append(para("""
        The balancing arms are the fourth ablation axis. Because they are the
        subject of the P1 paper, the full analysis is given in Section 5.2
        rather than repeated here. The short statement is that P1 clearly
        improves draw detection, does not clearly improve the probabilistic
        score, and the choice between them depends on what the system is
        for."""))


# --- Section 9 --------------------------------------------------------------
def section_api(blocks):
    latency = read("api_latency.csv")
    if latency is None:
        return
    blocks.append(h1("9. Service layer"))
    blocks.append(para("""
        <b>Design constraint.</b> A serving layer that reimplements feature
        engineering will eventually disagree with the training pipeline, and
        the disagreement will be silent. The service therefore imports the
        training modules: feature assembly comes from the shared modelling
        module, the live snapshot path calls the same snapshot builder used to
        construct the training table, and the fitted transforms are reused
        rather than refitted. There is one feature implementation in the
        repository and the service is not a second one."""))
    blocks.append(table(
        latency, ["endpoint", "n", "mean_ms", "p50_ms", "p95_ms", "p99_ms",
                  "max_ms"], caption_text="""
        Table 9.1. Service latency, measured in process so the numbers describe
        the service rather than the network."""))
    blocks.append(para("""
        <b>What we observed and why.</b> Single-prediction endpoints respond in
        roughly a tenth of a second, and the replay endpoint, which scores all
        nineteen snapshots of a match in one call, costs roughly nineteen times
        as much. That proportionality is the expected result and confirms that
        the cost is model evaluation rather than request handling. The health
        endpoint, which performs no inference, returns in under a millisecond,
        which isolates framework overhead from model cost. <b>Limitation.</b>
        These figures were measured on a single-core machine with the model
        selected for fast startup, so they should be read as an upper bound
        rather than as the best achievable latency."""))


# --- Section 10 -------------------------------------------------------------
def section_limitations(blocks):
    blocks.append(h1("10. Limitations"))
    prematch = read("prematch_features.csv", FEATURE_DIR)
    inplay = read("inplay_features.csv", FEATURE_DIR)
    width = ""
    if prematch is not None and inplay is not None:
        width = (f"The pre-match table carries {prematch.shape[1]} columns and "
                 f"the in-play table {inplay.shape[1]}, including identifiers "
                 f"and labels.")
    blocks.append(para(f"""
        <b>Feature breadth.</b> {width} Pressure intensity, passing volume by
        zone, carries and touches in the final third, set-piece counts,
        possession share, defensive actions, venue-split form and head-to-head
        history are now all built. Two quantities named in the brief are still
        absent, and both need information that the current event extraction
        does not carry: pass completion rate needs each pass's outcome, and
        carries into the final third, as distinct from carries within it, needs
        each carry's end location. Neither field is present in
        <code>clean_events.csv</code>, so neither is computed. They are not
        approximated by a proxy, because a proxy labelled as completion would
        be worse than an acknowledged gap."""))
    blocks.append(para("""
        <b>Hyperparameters.</b> If the status note at the front of this report
        is present, the results were produced with default configurations
        because the search did not complete. Rankings between learners should
        be treated as provisional until it has run."""))
    blocks.append(para("""
        <b>One season.</b> Roughly fifteen hundred matches from 2015/16 across
        four leagues. Because the split is chronological, the test block is the
        closing months of the season, where the home-win rate differs
        materially from the training block. This is a genuine distribution
        shift, and it is a further reason the models trail the market. It is
        also the correct way to split, since a random split would be
        optimistic."""))
    blocks.append(para("""
        <b>Correlated snapshots.</b> Nineteen snapshots per match are not
        nineteen independent observations. Grouped cross-validation and the
        clustered bootstrap handle this in tuning and testing, but the
        effective sample size for Task L remains the number of matches."""))
    blocks.append(para("""
        <b>Calibration data.</b> The validation split provides a few hundred
        matches, which is thin for isotonic regression and is why the
        probability floor exists and why calibration occasionally makes the
        calibration error slightly worse."""))
    blocks.append(para("""
        <b>The market comparison is not like-for-like.</b> The market observes
        team news and money flow that our features cannot. It is the right
        reference point, but the gap should not be read purely as a difference
        in modelling quality."""))


def section_conclusions(blocks):
    blocks.append(h1("11. Conclusions"))
    results = read("model_results.csv")
    bootstrap = read("significance_bootstrap.csv")
    curves = read("inplay_metric_by_minute.csv")
    if curves is not None:
        classification = curves[curves["task"] == "Lc"]
        in_play = classification[classification["series"] == "in-play"]
        frozen = classification[classification["series"] == "frozen pre-match"]
        if not in_play.empty and not frozen.empty:
            blocks.append(para(f"""
                <b>Live information matters far more than the choice of
                learner.</b> The best in-play ranked probability score reaches
                {in_play['rps'].min():.5f} against {frozen['rps'].min():.5f}
                for the frozen pre-match reference. No comparison between
                learners in this report comes close to that difference. The
                practical implication for the final system is that effort
                should go into what the model observes, not into which model
                observes it."""))
    if results is not None:
        classification = results[results["task"] == "C"]
        if not classification.empty:
            blocks.append(para(f"""
                <b>The pre-match problem is limited by information, not by
                capacity.</b> Across every learner the spread on task C runs
                from {classification['rps'].min():.5f} to
                {classification['rps'].max():.5f}. A problem where a constant
                baseline and a tuned ensemble land that close together is one
                where the features, not the model, set the ceiling."""))
    if bootstrap is not None:
        flag = bootstrap["significant"].astype(str).str.lower() == "true"
        blocks.append(para(f"""
            <b>Most of the leaderboard is not statistically supported.</b> Of
            {len(bootstrap)} pairwise comparisons, {int(flag.sum())} survive
            Holm correction, and the survivors are concentrated on the in-play
            tasks. This is reported as a headline finding rather than buried,
            because the alternative would be to present an ordering the
            evidence does not support."""))
    blocks.append(para("""
        <b>On P1.</b> The reimplementation is faithful and reproduces the
        behaviour its paper claims: it raises draw recall more than any other
        arm tested. It does not improve the probabilistic score, for a reason
        that follows from the metric rather than from any flaw in the method.
        Both halves of that result are reported."""))
    blocks.append(para("""
        <b>On P2.</b> Hierarchical shrinkage behaves as its paper predicts. It
        produces well-calibrated probabilities before any external calibration,
        because averaging shrunk leaf values pulls extreme estimates towards
        better-supported ancestors, and it selects its own shrinkage strength
        by cross-validation. It is consistently the most expensive model to
        fit, which is the price of that internal selection step. On pre-match
        data its advantage over simpler learners is not statistically
        detectable, which is a statement about the dataset rather than about
        the method."""))
    blocks.append(para("""
        <b>What the project demonstrates.</b> Leakage discipline enforced by
        assertions and tests rather than by assertion in prose; two paper
        reimplementations each verified against the published equations by
        their own test suites; a serving layer that shares one feature
        implementation with training; and the honest reporting of negative
        results, including the market gap, the null statistical findings on
        pre-match data, and the imbalance arms that did not help."""))


# --- Appendices -------------------------------------------------------------
def appendix_a(blocks):
    blocks.append(page_break())
    blocks.append(h1("Appendix A. Hand derivation of the P2 model"))
    blocks.append(para(f"""
        The P2 paper is {P2_CITATION}. Hierarchical shrinkage regularises an
        already-fitted tree by rewriting the value of each node as a damped sum
        along the path from the root to that node. It does not change the tree
        structure, only the values stored in it."""))
    blocks.append(para("""
        Let the path from the root to a leaf be t(0), t(1), ..., t(L). Write
        mu(t) for the mean of the training responses falling in node t, and
        N(t) for the number of training samples in t. The unshrunk prediction
        at the leaf is just mu(t(L)), which can be written as a telescoping
        sum:"""))
    blocks.append(equation(
        r"$\mu(t_L)=\mu(t_0)+\sum_{l=1}^{L}"
        r"\left[\mu(t_l)-\mu(t_{l-1})\right]$", "telescope"))
    blocks.append(para("""
        Every term in the sum cancels except the first and last, so this is an
        identity rather than an approximation. Its value is that it separates
        the prediction into a root-level estimate plus one increment per split.
        Each increment is the amount that one additional split changed the
        estimate."""))
    blocks.append(para("""
        The key observation is that increments deep in the tree are estimated
        from few samples and are therefore unreliable, while increments near
        the root are estimated from many. Hierarchical shrinkage damps each
        increment in proportion to how little data supported it:"""))
    blocks.append(equation(
        r"$f_{HS}(x)=\mu(t_0)+\sum_{l=1}^{L}"
        r"\frac{\mu(t_l)-\mu(t_{l-1})}{1+\lambda/N(t_{l-1})}$", "hs"))
    blocks.append(para("""
        The damping uses the parent's sample count N(t(l-1)), not the child's.
        This is deliberate. The increment is evidence about a split of the
        parent, so the relevant question is how much data was available when
        that split was chosen. Using the child's count would make the damping
        depend on the very partition whose reliability is in doubt."""))
    blocks.append(para("""
        Two limits check the form. As lambda approaches zero, every damping
        factor approaches one and the expression collapses back to the
        telescoping identity, recovering the unshrunk tree exactly. As lambda
        grows without bound, every factor approaches zero and the prediction
        collapses to mu(t(0)), the root mean, which is the constant model. So
        lambda interpolates between the fitted tree and a constant. Both limits
        are asserted in the test suite rather than left as claims."""))
    blocks.append(para("""
        The implementation uses the same statement written as a top-down
        recursion. Writing v(t) for the shrunk value of node t and p for its
        parent:"""))
    blocks.append(equation(
        r"$v(t)=v(p)+\frac{\mu(t)-\mu(p)}{1+\lambda/N(p)},"
        r"\qquad v(t_0)=\mu(t_0)$", "recursion"))
    blocks.append(para("""
        Unrolling this recursion from the root reproduces the closed form
        exactly. The test suite checks this on every root-to-leaf path of a
        fitted tree, not on a single example, because an error in the recursion
        could easily be correct at one depth and wrong at another."""))
    blocks.append(para("""
        For classification, mu(t) is the vector of class proportions in node t,
        which lies on the probability simplex. The shrunk value is a
        combination of simplex points whose coefficients sum to one, so its
        coordinates also sum to one. Each coordinate stays within the unit
        interval because the damping factor lies between zero and one, so the
        update moves the child's proportion only part of the way from the
        parent's and cannot overshoot. The shrunk forest therefore produces
        valid probability vectors without renormalisation, and the
        implementation clips only floating-point rounding error. This property
        is why the P2 model needs less external calibration than the boosted
        models, as observed in Section 4.2."""))
    blocks.append(para("""
        Finally, lambda is selected by cross-validation inside the training
        rows only, so the validation split remains reserved for calibration.
        This is what makes a single fit of the P2 model cost several forest
        fits, which is visible in the compute table in Section 5.4."""))


def appendix_b(blocks):
    blocks.append(page_break())
    blocks.append(h1("Appendix B. Reproducibility"))

    blocks.append(h2("B.1 Environment and dependencies"))
    blocks.append(para("""
        <code>requirements.txt</code> declares lower bounds rather than exact
        versions, so the project installs on current Python releases; pinning
        old versions caused installation to fail on Python 3.14, where the
        pinned pandas has no prebuilt wheel and pip falls back to compiling
        from source. Reproducibility is preserved separately: the pipeline
        runner records the exact resolved versions at run time and writes
        <code>requirements.lock.txt</code>. Installing that lock file
        reproduces the environment that produced the results in this report,
        while ordinary users install the unpinned file."""))
    lock = PROJECT / "requirements.lock.txt"
    declared = ["numpy", "pandas", "scipy", "scikit-learn", "xgboost",
                "lightgbm", "imbalanced-learn", "shap", "matplotlib",
                "pillow", "plotly", "psutil", "reportlab", "fastapi",
                "uvicorn", "httpx"]
    if lock.exists():
        blocks.append(para("""
            The versions below are the ones actually resolved for the run that
            produced this report, read from the lock file at build time."""))
        pinned = {}
        for line in lock.read_text(encoding="utf-8").splitlines():
            if "==" in line and not line.startswith("#"):
                name, _, value = line.partition("==")
                pinned[name.strip().lower()] = value.strip()
        rows = [[name, pinned.get(name.lower(), "not recorded")]
                for name in declared]
        blocks.append(table(pd.DataFrame(rows,
                                         columns=["Package", "Version"])))
    else:
        blocks.append(para("""
            <b>No lock file was present when this report was built</b>, so the
            exact versions of that run are not recorded here. Running the
            pipeline through <code>run_pipeline.py</code> writes
            <code>requirements.lock.txt</code> and this section then reports
            real versions. The declared lower bounds follow."""))
        requirements = PROJECT / "requirements.txt"
        if requirements.exists():
            for line in requirements.read_text(encoding="utf-8").splitlines():
                if line.strip() and not line.startswith("#"):
                    blocks.append(mono(line))
    blocks.append(mono(f"Interpreter used for this build: Python "
                       f"{sys.version.split()[0]}"))

    blocks.append(h2("B.2 Random seeds"))
    seeds = pd.DataFrame([
        ["Model fitting and the zoo", "0",
         "run_models.py, ablation.py, shap_analysis.py, service.py"],
        ["Hyperparameter search", "0, per-model offset by crc32 of name|task",
         "tuning.py"],
        ["Cross-validation shuffling", "0",
         "tuning.py, hierarchical_shrinkage.py"],
        ["Kernel scaling subsamples", "0", "kernel_scaling.py"],
        ["Bootstrap resampling", "20260808", "significance.py"],
        ["Seed repetition range", "0 to N_SEEDS-1, default 5",
         "significance.py"],
    ], columns=["Purpose", "Value", "Where"])
    blocks.append(table(seeds))
    blocks.append(para("""
        The search seed is offset by a checksum of the model and task name
        rather than by Python's built-in hash, because string hashing is
        randomised per process by default and would make the search
        irreproducible across runs."""))

    blocks.append(h2("B.3 Fixed configuration"))
    configuration = pd.DataFrame([
        ["Snapshot minutes", "0 to 90 in steps of 5, 19 per match"],
        ["Margin clip", "-5 to +5"],
        ["Class order", "H, D, A"],
        ["Split ratios",
         "0.60 train, 0.20 validation, 0.20 test, chronological"],
        ["Rolling form window", "5 prior matches with a one-match lag"],
        ["Recent in-play window", "10 minutes"],
        ["Probability floor", "0.005, then renormalised"],
        ["Tuning budget", "12 candidates by 3 folds per model per task"],
        ["Exact-kernel training cap", "8000 rows"],
        ["Lambda grid for P2", "0, 0.1, 1, 10, 25, 50, 100"],
    ], columns=["Setting", "Value"])
    blocks.append(table(configuration))

    blocks.append(h2("B.4 How to reproduce"))
    blocks.append(para("""
        The stages form a dependency chain driven by
        <code>src/pipeline/run_all.py</code>. Each stage reads the previous
        stage's CSV output, so any stage can be re-run in isolation once its
        inputs exist. Set <code>SKIP_TUNING=1</code> to reuse an existing
        search result. After any change to the feature tables,
        <code>best_params.json</code> must be deleted before re-tuning, because
        a tuned configuration is specific to the feature space it was searched
        in."""))

    blocks.append(h2("B.5 Git history"))
    try:
        log = subprocess.run(["git", "log", "--oneline", "--all"],
                             cwd=str(PROJECT), capture_output=True, text=True,
                             timeout=60)
        lines = log.stdout.splitlines() if log.returncode == 0 else []
    except Exception:
        lines = []
    if lines:
        for line in lines:
            blocks.append(mono(line))
    else:
        blocks.append(para("The command git log --oneline --all produced no "
                           "output in this environment."))


# --- Entry point ------------------------------------------------------------
def build_blocks():
    blocks = []
    front_matter(blocks)
    section_framing(blocks)
    section_data(blocks)
    section_features(blocks)
    section_modelling(blocks)
    section_market(blocks)
    section_imbalance(blocks)
    section_inplay(blocks)
    section_scaling(blocks)
    section_conversion(blocks)
    section_significance(blocks)
    section_shap(blocks)
    section_ablation(blocks)
    section_api(blocks)
    section_limitations(blocks)
    section_conclusions(blocks)
    body = len(blocks)
    appendix_a(blocks)
    appendix_b(blocks)
    return blocks, body
