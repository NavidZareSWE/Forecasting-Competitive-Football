from pathlib import Path

import pandas as pd


HERE = Path(__file__).resolve().parent
PROJECT = HERE.parent
PROCESSED_DIR = PROJECT / "reports" / "processed"
FEATURE_DIR = PROJECT / "reports" / "features"

ROLLING_WINDOW = 5
FORM_COLUMNS = ["gf", "ga", "xgf", "xga", "points", "win"]

# Per-team per-match aggregates computed from clean_events.csv (brief 2.2 /
# flowchart 5A). Each becomes a leakage-safe rolling form column exactly like
# FORM_COLUMNS: shifted one match, then averaged over the window.
EVENT_FORM_COLUMNS = [
    "shots", "shots_on_target", "pressures", "possession_share",
    "passes", "pass_completion", "passes_def", "passes_mid", "passes_fin",
    "carries_final_third", "corners", "free_kicks", "throw_ins",
    "defensive_actions",
]

ON_TARGET_OUTCOMES = {"Goal", "Saved", "Saved To Post"}
DEFENSIVE_TYPES = {"Clearance", "Block", "Interception", "Ball Recovery",
                   "Duel", "Foul Committed"}
FINAL_THIRD_X = 80.0  # pitch is 120 long; the final third starts at x = 80
MIDDLE_THIRD_X = 40.0


def _present_form_columns(frame):
    return [c for c in FORM_COLUMNS + EVENT_FORM_COLUMNS if c in frame.columns]


# ---- Build Team-Match Long Format ----
def build_team_match_long_table(matches, events_index):
    merged = matches.merge(
        events_index[["match_id", "home_xg", "away_xg"]], on="match_id", how="left")

    home = pd.DataFrame({
        "match_id": merged["match_id"], "match_date": merged["match_date"],
        "team_id": merged["home_team_id"], "opponent_id": merged["away_team_id"],
        "venue": "home",
        "gf": merged["home_score"], "ga": merged["away_score"],
        "xgf": merged["home_xg"], "xga": merged["away_xg"],
    })
    away = pd.DataFrame({
        "match_id": merged["match_id"], "match_date": merged["match_date"],
        "team_id": merged["away_team_id"], "opponent_id": merged["home_team_id"],
        "venue": "away",
        "gf": merged["away_score"], "ga": merged["home_score"],
        "xgf": merged["away_xg"], "xga": merged["home_xg"],
    })
    team_match_df = pd.concat([home, away], ignore_index=True)
    team_match_df["points"] = 1
    team_match_df["points"] = team_match_df["points"].mask(
        team_match_df["gf"] > team_match_df["ga"], 3)
    team_match_df["points"] = team_match_df["points"].mask(
        team_match_df["gf"] < team_match_df["ga"], 0)
    team_match_df["win"] = (team_match_df["gf"] >
                            team_match_df["ga"]).astype(int)
    return team_match_df


# ---- Per-Match Event Aggregates (brief 2.2 / flowchart 5A) ----
def build_event_aggregates(events):
    """One row per (match_id, team_id) of within-match event totals.

    These describe the match itself; they only become pre-match features after
    the shift-then-roll in add_rolling_form_features, like every FORM column.
    """
    events = events.copy()
    is_pass = events["type"] == "Pass"
    is_complete = is_pass & events["pass_outcome"].isna()

    events["shots"] = (events["type"] == "Shot").astype(int)
    events["shots_on_target"] = (
        events["shot_outcome"].isin(ON_TARGET_OUTCOMES)).astype(int)
    events["pressures"] = (events["type"] == "Pressure").astype(int)
    events["passes"] = is_pass.astype(int)
    events["passes_complete"] = is_complete.astype(int)
    events["passes_def"] = (is_pass & (events["x"] < MIDDLE_THIRD_X)).astype(int)
    events["passes_mid"] = (is_pass & (events["x"] >= MIDDLE_THIRD_X)
                            & (events["x"] < FINAL_THIRD_X)).astype(int)
    events["passes_fin"] = (is_pass & (events["x"] >= FINAL_THIRD_X)).astype(int)
    events["carries_final_third"] = ((events["type"] == "Carry")
                                     & (events["end_x"] >= FINAL_THIRD_X)).astype(int)
    events["corners"] = (events["pass_type"] == "Corner").astype(int)
    events["free_kicks"] = (events["pass_type"] == "Free Kick").astype(int)
    events["throw_ins"] = (events["pass_type"] == "Throw-in").astype(int)
    events["defensive_actions"] = events["type"].isin(
        DEFENSIVE_TYPES).astype(int)

    count_columns = ["shots", "shots_on_target", "pressures", "passes",
                     "passes_complete", "passes_def", "passes_mid",
                     "passes_fin", "carries_final_third", "corners",
                     "free_kicks", "throw_ins", "defensive_actions"]
    per_team = (events.dropna(subset=["team_id"])
                .groupby(["match_id", "team_id"])[count_columns]
                .sum().reset_index())
    per_team["team_id"] = per_team["team_id"].astype(int)

    # Rates and shares derived from the counts.
    per_team["pass_completion"] = (
        per_team["passes_complete"]
        / per_team["passes"].mask(per_team["passes"] == 0))
    match_passes = per_team.groupby("match_id")["passes"].transform("sum")
    per_team["possession_share"] = (
        per_team["passes"] / match_passes.mask(match_passes == 0))
    return per_team.drop(columns="passes_complete")


def add_event_aggregates(team_match_df, event_aggregates):
    merged = team_match_df.merge(
        event_aggregates, on=["match_id", "team_id"], how="left")
    assert len(merged) == len(team_match_df), \
        "event aggregates must not duplicate team-match rows"
    return merged


# ---- Compute Rolling Form Features ----
def _prior_window_mean(team_series):
    return team_series.shift(1).rolling(ROLLING_WINDOW, min_periods=1).mean()


def add_rolling_form_features(team_match_df):
    team_match_df = team_match_df.sort_values(
        ["team_id", "match_date", "match_id"]).reset_index(drop=True)
    grouped = team_match_df.groupby("team_id", group_keys=False)
    for column in _present_form_columns(team_match_df):
        team_match_df[f"form_{column}"] = grouped[column].apply(
            _prior_window_mean)
    team_match_df["rest_days"] = grouped["match_date"].apply(
        lambda dates: dates.diff().dt.days)
    team_match_df["played_prior"] = grouped.cumcount()
    return team_match_df


# ---- Head-to-Head Form ----
def add_head_to_head(team_match_df):
    """Expanding mean of prior meetings against this specific opponent.

    Shifted one meeting, so the current match never contributes to its own
    row; a pair's first meeting carries NaN like every other form column.
    """
    team_match_df = team_match_df.sort_values(
        ["team_id", "opponent_id", "match_date", "match_id"]).reset_index(drop=True)
    pair_keys = [team_match_df["team_id"], team_match_df["opponent_id"]]

    def prior_expanding_mean(series):
        return series.groupby(pair_keys, group_keys=False).apply(
            lambda s: s.shift(1).expanding().mean())

    team_match_df["h2h_margin"] = prior_expanding_mean(
        team_match_df["gf"] - team_match_df["ga"])
    team_match_df["h2h_win"] = prior_expanding_mean(
        team_match_df["win"].astype(float))
    return team_match_df.sort_values(
        ["team_id", "match_date", "match_id"]).reset_index(drop=True)


# ---- Pivot to Pre-Match Features ----
def build_prematch_features(team_match_df):
    rolled = [f"form_{c}" for c in _present_form_columns(team_match_df)
              if f"form_{c}" in team_match_df.columns]
    h2h_cols = [c for c in ["h2h_margin", "h2h_win"]
                if c in team_match_df.columns]
    form_cols = rolled + h2h_cols + ["rest_days", "played_prior"]
    home = team_match_df[team_match_df["venue"] == "home"].set_index("match_id")[
        form_cols]
    away = team_match_df[team_match_df["venue"] == "away"].set_index("match_id")[
        form_cols]
    home = home.add_prefix("home_")
    away = away.add_prefix("away_")
    features = home.join(away, how="inner")
    for column in rolled + h2h_cols + ["rest_days"]:
        features[f"diff_{column}"] = features[f"home_{column}"] - \
            features[f"away_{column}"]
    return features.reset_index()


def main():
    # ---- Load Data ----
    match_path = PROCESSED_DIR / "match_store.csv"
    events_path = PROCESSED_DIR / "events_index.csv"
    clean_events_path = PROCESSED_DIR / "clean_events.csv"
    splits_path = PROCESSED_DIR / "temporal_match_splits.csv"
    for path in [match_path, events_path, clean_events_path, splits_path]:
        if not path.exists():
            raise FileNotFoundError(
                f"Missing {path}. Run the Phase 1-4 pipeline before Phase 5A.")

    matches = pd.read_csv(match_path, parse_dates=[
                          "match_date"], encoding="utf-8")
    events_index = pd.read_csv(events_path, encoding="utf-8")
    clean_events = pd.read_csv(
        clean_events_path, encoding="utf-8",
        usecols=["match_id", "team_id", "type", "x", "end_x", "shot_outcome",
                 "pass_outcome", "pass_type"])
    splits = pd.read_csv(splits_path, parse_dates=[
                         "match_date"], encoding="utf-8")

    # ---- Build Features ----
    team_match_df = build_team_match_long_table(matches, events_index)
    team_match_df = add_event_aggregates(
        team_match_df, build_event_aggregates(clean_events))
    team_match_df = add_rolling_form_features(team_match_df)
    team_match_df = add_head_to_head(team_match_df)

    # ---- Leakage Self-Check ----
    rolled_columns = [
        f"form_{c}" for c in _present_form_columns(team_match_df)
        if f"form_{c}" in team_match_df.columns]
    first_rows = team_match_df.sort_values(["team_id", "match_date", "match_id"]) \
        .groupby("team_id").head(1)
    assert first_rows[rolled_columns].isna().all().all(), \
        "prior-only violation: first match carries rolling form"
    first_meetings = team_match_df.sort_values(
        ["team_id", "opponent_id", "match_date", "match_id"]) \
        .groupby(["team_id", "opponent_id"]).head(1)
    assert first_meetings[["h2h_margin", "h2h_win"]].isna().all().all(), \
        "prior-only violation: a pair's first meeting carries head-to-head form"

    features = build_prematch_features(team_match_df)
    meta = splits[["match_id", "competition_name", "match_date", "split",
                   "label_result", "label_margin"]]
    prematch = meta.merge(features, on="match_id", how="inner",
                          validate="one_to_one")
    prematch = prematch.sort_values(
        ["match_date", "match_id"]).reset_index(drop=True)

    assert prematch["match_id"].is_unique, "one pre-match row per match expected"

    # ---- Save Output ----
    FEATURE_DIR.mkdir(parents=True, exist_ok=True)
    output_path = FEATURE_DIR / "prematch_features.csv"
    prematch.to_csv(output_path, index=False, encoding="utf-8")
    print(f"Pre-match feature rows: {len(prematch)} matches")
    print(prematch.groupby("split").size().reindex(
        ["train", "validation", "test"]).to_string())
    print(f"Wrote {output_path}")


if __name__ == "__main__":
    main()
