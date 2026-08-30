"""Run from the repository root with:

    python src/pipeline/build_extended_market_baseline.py
"""
from pathlib import Path
import sqlite3
import sys

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

from download_extended_data import read_football_data_csv  # noqa: E402
from build_extended_match_store import DIV_TO_LEAGUE  # noqa: E402

PROJECT = HERE.parent
PROCESSED_DIR = PROJECT / "reports" / "processed"
ESD_PATH = PROJECT / "data" / "european_soccer_db" / "database.sqlite"
FOOTBALL_DATA_DIR = PROJECT / "data" / "Football_Data"

ESD_BOOKIES = ["B365", "BW", "IW", "LB", "PS", "WH", "SJ", "VC", "GB", "BS"]
TEST_COVERAGE_FLOOR = 0.98


def de_vig(frame):
    implied = pd.DataFrame({
        "p_home": 1.0 / frame["B365H"],
        "p_draw": 1.0 / frame["B365D"],
        "p_away": 1.0 / frame["B365A"]})
    overround = implied.sum(axis=1)
    for column in implied:
        implied[column] = implied[column] / overround
    implied["overround"] = overround
    return pd.concat([frame.reset_index(drop=True),
                      implied.reset_index(drop=True)], axis=1)


def statsbomb_rows(store):
    baseline = pd.read_csv(PROCESSED_DIR / "market_baseline.csv",
                           encoding="utf-8")
    keep = baseline[["match_id", "B365H", "B365D", "B365A",
                     "p_home", "p_draw", "p_away", "overround"]].copy()
    keep["odds_source"] = "b365"
    merged = keep.merge(store[store["era"] == "statsbomb"]
                        [["match_id", "league", "season", "match_date",
                          "home_team", "away_team"]],
                        on="match_id", how="inner")
    assert len(merged) == len(keep), \
        "market_baseline.csv rows missing from the extended store"
    return merged


def esd_rows(store):
    columns = ", ".join(f"{b}{o}" for b in ESD_BOOKIES for o in "HDA")
    with sqlite3.connect(ESD_PATH) as connection:
        odds = pd.read_sql(f"SELECT match_api_id, {columns} FROM Match",
                           connection)
    esd_store = store[store["era"] == "esd"]
    merged = esd_store.merge(odds, left_on="source_match_id",
                             right_on="match_api_id", how="left")

    triplets = np.stack([
        merged[[f"{b}H", f"{b}D", f"{b}A"]].to_numpy(dtype=float)
        for b in ESD_BOOKIES], axis=1)
    complete = ~np.isnan(triplets).any(axis=2)
    triplets[~complete] = np.nan
    median = np.nanmedian(triplets, axis=1)

    has_b365 = complete[:, ESD_BOOKIES.index("B365")]
    has_any = complete.any(axis=1)
    frame = merged[["match_id", "league", "season", "match_date",
                    "home_team", "away_team"]].copy()
    b365 = merged[["B365H", "B365D", "B365A"]].to_numpy(dtype=float)
    chosen = np.where(has_b365[:, None], b365, median)
    frame[["B365H", "B365D", "B365A"]] = chosen
    frame["odds_source"] = np.where(has_b365, "b365",
                                    np.where(has_any, "bookie_median", "none"))
    kept = frame[frame["odds_source"] != "none"].copy()
    dropped = frame[frame["odds_source"] == "none"]
    return de_vig(kept), dropped


def football_data_rows(store):
    fd_store = store[store["era"] == "football_data"]
    alias = pd.read_csv(PROCESSED_DIR / "alias_map_extended.csv",
                        encoding="utf-8")
    alias = alias[alias["source"] == "football_data"].set_index(
        ["league", "season", "source_name"])["canonical_team_id"]
    rows = []
    for season_dir in sorted(FOOTBALL_DATA_DIR.iterdir()):
        if not season_dir.is_dir():
            continue
        season = f"20{season_dir.name[:2]}/20{season_dir.name[2:]}"
        for path in sorted(season_dir.glob("*.csv")):
            league = DIV_TO_LEAGUE.get(path.stem)
            if league is None:
                continue
            frame = read_football_data_csv(path).dropna(
                subset=["HomeTeam", "AwayTeam"])
            frame["match_date"] = pd.to_datetime(
                frame["Date"], dayfirst=True, format="mixed").dt.normalize()
            for r in frame.itertuples():
                rows.append({
                    "league": league, "season": season,
                    "match_date": r.match_date,
                    "home_team_id": int(alias[(league, season, r.HomeTeam)]),
                    "away_team_id": int(alias[(league, season, r.AwayTeam)]),
                    "B365H": getattr(r, "B365H", np.nan),
                    "B365D": getattr(r, "B365D", np.nan),
                    "B365A": getattr(r, "B365A", np.nan)})
    odds = pd.DataFrame(rows)
    merged = fd_store[["match_id", "league", "season", "match_date",
                       "home_team", "away_team", "home_team_id",
                       "away_team_id"]].merge(
        odds, on=["league", "season", "match_date",
                  "home_team_id", "away_team_id"], how="left",
        validate="one_to_one")
    merged = merged.drop(columns=["home_team_id", "away_team_id"])
    kept = merged[merged[["B365H", "B365D", "B365A"]].notna().all(axis=1)].copy()
    dropped = merged[merged[["B365H", "B365D", "B365A"]].isna().any(axis=1)]
    kept["odds_source"] = "b365"
    return de_vig(kept), dropped


COLUMNS = ["match_id", "league", "season", "match_date", "home_team",
           "away_team", "B365H", "B365D", "B365A", "odds_source",
           "p_home", "p_draw", "p_away", "overround"]


def main():
    store = pd.read_csv(PROCESSED_DIR / "extended_match_store.csv",
                        encoding="utf-8", parse_dates=["match_date"])
    splits = pd.read_csv(PROCESSED_DIR / "temporal_match_splits_extended.csv",
                         encoding="utf-8", usecols=["match_id", "split"])

    sb = statsbomb_rows(store)
    esd, esd_dropped = esd_rows(store)
    fd, fd_dropped = football_data_rows(store)
    baseline = pd.concat([sb[COLUMNS], esd[COLUMNS], fd[COLUMNS]],
                         ignore_index=True)

    assert baseline["match_id"].is_unique
    sums = baseline[["p_home", "p_draw", "p_away"]].sum(axis=1)
    assert (sums.sub(1.0).abs() < 1e-9).all(), "probabilities do not sum to 1"
    assert (baseline["overround"] > 1.0).all(), "overround must exceed 1"

    coverage = (baseline.groupby(["league", "season"]).size()
                .rename("with_odds").reset_index())
    totals = (store.groupby(["league", "season"]).size()
              .rename("matches").reset_index())
    coverage = totals.merge(coverage, on=["league", "season"], how="left")
    coverage["with_odds"] = coverage["with_odds"].fillna(0).astype(int)
    coverage["coverage"] = (coverage["with_odds"] / coverage["matches"]).round(4)

    scope = (store[["match_id", "season"]]
             .merge(splits, on="match_id", how="left")
             .groupby("season")["split"]
             .agg(lambda s: "out_of_scope"
                  if (s == "out_of_scope").all() else "modelled"))
    coverage["scope"] = coverage["season"].map(scope).fillna("modelled")

    failures = pd.concat(
        [esd_dropped.assign(reason="no complete bookmaker triplet in ESD")
         [["match_id", "league", "season", "match_date", "reason"]],
         fd_dropped.assign(reason="missing B365 odds in season CSV")
         [["match_id", "league", "season", "match_date", "reason"]]],
        ignore_index=True)

    test_ids = set(splits.loc[splits["split"] == "test", "match_id"])
    test_cov = baseline["match_id"].isin(test_ids).sum() / len(test_ids)
    assert test_cov >= TEST_COVERAGE_FLOOR, \
        f"test-era odds coverage {test_cov:.4f} below {TEST_COVERAGE_FLOOR}"

    baseline.to_csv(PROCESSED_DIR / "market_baseline_extended.csv",
                    index=False, encoding="utf-8")
    coverage.to_csv(PROCESSED_DIR / "odds_coverage_extended.csv",
                    index=False, encoding="utf-8")
    failures.to_csv(PROCESSED_DIR / "odds_failures_extended.csv",
                    index=False, encoding="utf-8")
    in_scope = coverage[coverage["scope"] == "modelled"]
    print(f"Baseline rows: {len(baseline)} / {len(store)} matches "
          f"({len(baseline) / len(store):.1%}); test-era coverage {test_cov:.4f}")
    print(f"Seasons inside the pinned split window: "
          f"{in_scope['matches'].sum()} matches, "
          f"{in_scope['with_odds'].sum()} tagged "
          f"({in_scope['with_odds'].sum() / in_scope['matches'].sum():.4f}). "
          f"Report these, not the all-seasons totals.")
    print(baseline.groupby("odds_source").size().to_string())
    print("Wrote market_baseline_extended.csv, odds_coverage_extended.csv, "
          "odds_failures_extended.csv")


if __name__ == "__main__":
    main()
