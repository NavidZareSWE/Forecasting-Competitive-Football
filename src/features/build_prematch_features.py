from pathlib import Path
import sys

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

from event_aggregates import QUANTITIES

PROJECT = HERE.parent
PROCESSED_DIR = PROJECT / "reports" / "processed"
FEATURE_DIR = PROJECT / "reports" / "features"

ROLLING_WINDOW = 5
HEAD_TO_HEAD_WINDOW = 3
BASE_FORM_COLUMNS = ["gf", "ga", "xgf", "xga", "points", "win"]

AGGREGATE_COLUMNS = ([f"agg_{q}" for q in QUANTITIES]
                     + [f"agg_opp_{q}" for q in QUANTITIES] + ["red_cards"])
FORM_COLUMNS = BASE_FORM_COLUMNS + AGGREGATE_COLUMNS


# ---- Build Team-Match Long Format ----
def build_team_match_long_table(matches, events_index, aggregates=None):
    merged = matches.merge(
        events_index[["match_id", "home_xg", "away_xg"]], on="match_id", how="left")

    home = pd.DataFrame({
        "match_id": merged["match_id"], "match_date": merged["match_date"],
        "team_id": merged["home_team_id"], "venue": "home",
        "gf": merged["home_score"], "ga": merged["away_score"],
        "xgf": merged["home_xg"], "xga": merged["away_xg"],
    })
    away = pd.DataFrame({
        "match_id": merged["match_id"], "match_date": merged["match_date"],
        "team_id": merged["away_team_id"], "venue": "away",
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
    team_match_df["opponent_id"] = team_match_df["match_id"].map(
        matches.set_index("match_id")["home_team_id"])
    away_rows = team_match_df["venue"] == "away"
    team_match_df.loc[~away_rows, "opponent_id"] = team_match_df.loc[
        ~away_rows, "match_id"].map(
            matches.set_index("match_id")["away_team_id"])

    if aggregates is not None:
        team_match_df = team_match_df.merge(
            aggregates, on=["match_id", "team_id", "venue"], how="left",
            validate="one_to_one")
    return team_match_df


# ---- Compute Rolling Form Features ----
def _prior_window_mean(team_series):
    return team_series.shift(1).rolling(ROLLING_WINDOW, min_periods=1).mean()


def add_rolling_form_features(team_match_df):
    team_match_df = team_match_df.sort_values(
        ["team_id", "match_date", "match_id"]).reset_index(drop=True)
    grouped = team_match_df.groupby("team_id", group_keys=False)
    present = [c for c in FORM_COLUMNS if c in team_match_df.columns]
    rolled = {f"form_{column}": grouped[column].apply(_prior_window_mean)
              for column in present}
    rolled["rest_days"] = grouped["match_date"].apply(
        lambda dates: dates.diff().dt.days)
    rolled["played_prior"] = grouped.cumcount()

    by_venue = team_match_df.groupby(["team_id", "venue"], group_keys=False)
    for column in BASE_FORM_COLUMNS:
        rolled[f"venue_form_{column}"] = by_venue[column].apply(
            _prior_window_mean)
    rolled["venue_played_prior"] = by_venue.cumcount()

    team_match_df = pd.concat(
        [team_match_df, pd.DataFrame(rolled, index=team_match_df.index)],
        axis=1)
    return team_match_df


def add_head_to_head_features(team_match_df):
    frame = team_match_df.sort_values(
        ["team_id", "opponent_id", "match_date", "match_id"]).copy()
    pair = frame.groupby(["team_id", "opponent_id"], group_keys=False)
    for column in ["gf", "ga", "points"]:
        frame[f"h2h_{column}"] = pair[column].apply(
            lambda s: s.shift(1).rolling(HEAD_TO_HEAD_WINDOW,
                                         min_periods=1).mean())
    frame["h2h_played_prior"] = pair.cumcount()
    return frame


# ---- Pivot to Pre-Match Features ----
def build_prematch_features(team_match_df):
    rolled = [f"form_{c}" for c in FORM_COLUMNS
              if f"form_{c}" in team_match_df.columns]
    venue_cols = [f"venue_form_{c}" for c in BASE_FORM_COLUMNS
                  if f"venue_form_{c}" in team_match_df.columns]
    venue_cols += [c for c in ["venue_played_prior"]
                   if c in team_match_df.columns]
    h2h_cols = [c for c in ["h2h_gf", "h2h_ga", "h2h_points",
                            "h2h_played_prior"]
                if c in team_match_df.columns]
    form_cols = rolled + venue_cols + h2h_cols + ["rest_days", "played_prior"]
    home = team_match_df[team_match_df["venue"] == "home"].set_index("match_id")[
        form_cols]
    away = team_match_df[team_match_df["venue"] == "away"].set_index("match_id")[
        form_cols]
    home = home.add_prefix("home_")
    away = away.add_prefix("away_")
    features = home.join(away, how="inner")
    for column in rolled + venue_cols + h2h_cols + ["rest_days"]:
        if column.endswith("played_prior"):
            continue
        features[f"diff_{column}"] = features[f"home_{column}"] - \
            features[f"away_{column}"]
    return features.reset_index()


def main():
    # ---- Load Data ----
    match_path = PROCESSED_DIR / "match_store.csv"
    events_path = PROCESSED_DIR / "events_index.csv"
    splits_path = PROCESSED_DIR / "temporal_match_splits.csv"
    for path in [match_path, events_path, splits_path]:
        if not path.exists():
            raise FileNotFoundError(
                f"Missing {path}. Run the Phase 1-4 pipeline before Phase 5A.")

    matches = pd.read_csv(match_path, parse_dates=[
                          "match_date"], encoding="utf-8")
    events_index = pd.read_csv(events_path, encoding="utf-8")
    splits = pd.read_csv(splits_path, parse_dates=[
                         "match_date"], encoding="utf-8")

    # ---- Build Features ----
    aggregates_path = PROCESSED_DIR / "team_match_aggregates.csv"
    aggregates = None
    if aggregates_path.exists():
        aggregates = pd.read_csv(aggregates_path, encoding="utf-8")
        print(f"Event aggregates: {aggregates.shape[1] - 3} per-team "
              f"quantities over {aggregates['match_id'].nunique()} matches")
    else:
        print("team_match_aggregates.csv not found; building form from the "
              "basic columns only. Run build_team_match_aggregates.py first "
              "for the full feature set.")

    team_match_df = build_team_match_long_table(matches, events_index,
                                                aggregates)
    team_match_df = add_rolling_form_features(team_match_df)
    team_match_df = add_head_to_head_features(team_match_df)

    # ---- Leakage Self-Check ----
    first_rows = team_match_df.sort_values(["team_id", "match_date", "match_id"]) \
        .groupby("team_id").head(1)
    rolled = [f"form_{c}" for c in FORM_COLUMNS
              if f"form_{c}" in team_match_df.columns]
    assert first_rows[rolled].isna().all().all(), \
        "prior-only violation: first match carries rolling form"
    assert (first_rows["played_prior"] == 0).all(), \
        "prior-only violation: first match reports prior appearances"

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
