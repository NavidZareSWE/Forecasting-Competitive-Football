from pathlib import Path

import pandas as pd


HERE = Path(__file__).resolve().parent
PROJECT = HERE.parent
PROCESSED_DIR = PROJECT / "reports" / "processed"
FEATURE_DIR = PROJECT / "reports" / "features"

ROLLING_WINDOW = 5
FORM_COLUMNS = ["gf", "ga", "xgf", "xga", "points", "win"]


# ---- Build Team-Match Long Format ----
def build_team_match_long_table(matches, events_index):
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
    return team_match_df


# ---- Compute Rolling Form Features ----
def _prior_window_mean(team_series):
    return team_series.shift(1).rolling(ROLLING_WINDOW, min_periods=1).mean()


def add_rolling_form_features(team_match_df):
    team_match_df = team_match_df.sort_values(
        ["team_id", "match_date", "match_id"]).reset_index(drop=True)
    grouped = team_match_df.groupby("team_id", group_keys=False)
    for column in FORM_COLUMNS:
        team_match_df[f"form_{column}"] = grouped[column].apply(
            _prior_window_mean)
    team_match_df["rest_days"] = grouped["match_date"].apply(
        lambda dates: dates.diff().dt.days)
    team_match_df["played_prior"] = grouped.cumcount()
    return team_match_df


# ---- Pivot to Pre-Match Features ----
def build_prematch_features(team_match_df):
    form_cols = [f"form_{c}" for c in FORM_COLUMNS] + \
        ["rest_days", "played_prior"]
    home = team_match_df[team_match_df["venue"] == "home"].set_index("match_id")[
        form_cols]
    away = team_match_df[team_match_df["venue"] == "away"].set_index("match_id")[
        form_cols]
    home = home.add_prefix("home_")
    away = away.add_prefix("away_")
    features = home.join(away, how="inner")
    for column in [f"form_{c}" for c in FORM_COLUMNS] + ["rest_days"]:
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
    team_match_df = build_team_match_long_table(matches, events_index)
    team_match_df = add_rolling_form_features(team_match_df)

    # ---- Leakage Self-Check ----
    first_rows = team_match_df.sort_values(["team_id", "match_date", "match_id"]) \
        .groupby("team_id").head(1)
    assert first_rows[[f"form_{c}" for c in FORM_COLUMNS]].isna().all().all(), \
        "prior-only violation: first match carries rolling form"

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
