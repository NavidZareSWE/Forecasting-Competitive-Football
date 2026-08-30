"""Run from the repository root with:

    python src/features/build_extended_prematch_features.py
"""
from pathlib import Path

import pandas as pd

HERE = Path(__file__).resolve().parent
PROJECT = HERE.parent
PROCESSED_DIR = PROJECT / "reports" / "processed"
FEATURE_DIR = PROJECT / "reports" / "features"

ROLLING_WINDOW = 5
HEAD_TO_HEAD_WINDOW = 3
BASE_FORM_COLUMNS = ["gf", "ga", "points", "win"]


def build_team_match_long_table(matches):
    home = pd.DataFrame({
        "match_id": matches["match_id"], "match_date": matches["match_date"],
        "team_id": matches["home_team_id"], "venue": "home",
        "opponent_id": matches["away_team_id"],
        "gf": matches["home_score"], "ga": matches["away_score"]})
    away = pd.DataFrame({
        "match_id": matches["match_id"], "match_date": matches["match_date"],
        "team_id": matches["away_team_id"], "venue": "away",
        "opponent_id": matches["home_team_id"],
        "gf": matches["away_score"], "ga": matches["home_score"]})
    long = pd.concat([home, away], ignore_index=True)
    long["points"] = 1
    long["points"] = long["points"].mask(long["gf"] > long["ga"], 3)
    long["points"] = long["points"].mask(long["gf"] < long["ga"], 0)
    long["win"] = (long["gf"] > long["ga"]).astype(int)
    return long


def _prior_window_mean(series):
    return series.shift(1).rolling(ROLLING_WINDOW, min_periods=1).mean()


def add_rolling_form_features(long):
    long = long.sort_values(["team_id", "match_date", "match_id"]).reset_index(
        drop=True)
    grouped = long.groupby("team_id", group_keys=False)
    rolled = {f"form_{c}": grouped[c].apply(_prior_window_mean)
              for c in BASE_FORM_COLUMNS}
    rolled["rest_days"] = grouped["match_date"].apply(
        lambda dates: dates.diff().dt.days)
    rolled["played_prior"] = grouped.cumcount()
    by_venue = long.groupby(["team_id", "venue"], group_keys=False)
    for column in BASE_FORM_COLUMNS:
        rolled[f"venue_form_{column}"] = by_venue[column].apply(
            _prior_window_mean)
    rolled["venue_played_prior"] = by_venue.cumcount()
    return pd.concat([long, pd.DataFrame(rolled, index=long.index)], axis=1)


def add_head_to_head_features(long):
    frame = long.sort_values(
        ["team_id", "opponent_id", "match_date", "match_id"]).copy()
    pair = frame.groupby(["team_id", "opponent_id"], group_keys=False)
    for column in ["gf", "ga", "points"]:
        frame[f"h2h_{column}"] = pair[column].apply(
            lambda s: s.shift(1).rolling(HEAD_TO_HEAD_WINDOW,
                                         min_periods=1).mean())
    frame["h2h_played_prior"] = pair.cumcount()
    return frame


def pivot_prematch(long):
    rolled = [f"form_{c}" for c in BASE_FORM_COLUMNS]
    venue_cols = [f"venue_form_{c}" for c in BASE_FORM_COLUMNS] \
        + ["venue_played_prior"]
    h2h_cols = ["h2h_gf", "h2h_ga", "h2h_points", "h2h_played_prior"]
    form_cols = rolled + venue_cols + h2h_cols + ["rest_days", "played_prior"]
    home = long[long["venue"] == "home"].set_index("match_id")[form_cols]
    away = long[long["venue"] == "away"].set_index("match_id")[form_cols]
    features = home.add_prefix("home_").join(away.add_prefix("away_"),
                                             how="inner")
    for column in rolled + venue_cols + h2h_cols + ["rest_days"]:
        if column.endswith("played_prior"):
            continue
        features[f"diff_{column}"] = (features[f"home_{column}"]
                                      - features[f"away_{column}"])
    return features.reset_index()


def main():
    splits_path = PROCESSED_DIR / "temporal_match_splits_extended.csv"
    for path in [splits_path, PROCESSED_DIR / "team_ratings.csv",
                 PROCESSED_DIR / "rating_features.csv",
                 PROCESSED_DIR / "stat_form_features.csv"]:
        if not path.exists():
            raise FileNotFoundError(f"Missing {path}. Run earlier stages first.")

    splits = pd.read_csv(splits_path, encoding="utf-8",
                         parse_dates=["match_date"])
    ratings = pd.read_csv(PROCESSED_DIR / "team_ratings.csv",
                          encoding="utf-8")
    rating_features = pd.read_csv(PROCESSED_DIR / "rating_features.csv",
                                  encoding="utf-8")

    long = build_team_match_long_table(splits)
    long = add_rolling_form_features(long)
    long = add_head_to_head_features(long)

    first_rows = long.sort_values(["team_id", "match_date", "match_id"]) \
        .groupby("team_id").head(1)
    rolled = [f"form_{c}" for c in BASE_FORM_COLUMNS]
    assert first_rows[rolled].isna().all().all(), \
        "prior-only violation: first match carries rolling form"
    assert (first_rows["played_prior"] == 0).all()

    features = pivot_prematch(long)
    meta = splits[["match_id", "league", "era", "season", "match_date",
                   "split", "label_result", "label_margin"]].rename(
        columns={"league": "competition_name"})
    prematch = meta.merge(features, on="match_id", how="inner",
                          validate="one_to_one")
    prematch = prematch.merge(
        ratings[["match_id", "elo_home_pre", "elo_away_pre", "elo_diff",
                 "elo_expected_home", "pi_home_home_pre", "pi_home_away_pre",
                 "pi_away_home_pre", "pi_away_away_pre", "pi_expected_gd"]],
        on="match_id", how="left", validate="one_to_one")
    prematch = prematch.merge(rating_features, on="match_id", how="left",
                              validate="one_to_one")
    stat_form = pd.read_csv(PROCESSED_DIR / "stat_form_features.csv",
                            encoding="utf-8")
    prematch = prematch.merge(stat_form, on="match_id", how="left",
                              validate="one_to_one")

    assert prematch["match_id"].is_unique
    assert len(prematch) == len(splits)
    assert prematch["elo_diff"].notna().all(), "match missing team ratings"

    prematch = prematch.sort_values(["match_date", "match_id"]).reset_index(
        drop=True)
    FEATURE_DIR.mkdir(parents=True, exist_ok=True)
    output = FEATURE_DIR / "prematch_features_extended.csv"
    prematch.to_csv(output, index=False, encoding="utf-8")
    print(f"Extended pre-match rows: {len(prematch)} matches, "
          f"{prematch.shape[1]} columns")
    print(prematch.groupby("split").size().reindex(
        ["train", "validation", "test", "excluded"]).to_string())
    print(f"Wrote {output}")


if __name__ == "__main__":
    main()
