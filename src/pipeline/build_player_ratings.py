"""Run from the repository root with:

    python src/pipeline/build_player_ratings.py
"""
from pathlib import Path
import sqlite3
import sys

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

from download_extended_data import FIFA_FILTERED_PATH, FIFA_LEAGUE_IDS  # noqa: E402

PROJECT = HERE.parent
PROCESSED_DIR = PROJECT / "reports" / "processed"
ESD_PATH = PROJECT / "data" / "european_soccer_db" / "database.sqlite"

POSITION_BUCKETS = {
    "GK": "GK",
    "CB": "DEF", "LB": "DEF", "RB": "DEF", "LWB": "DEF", "RWB": "DEF",
    "CDM": "MID", "CM": "MID", "CAM": "MID", "LM": "MID", "RM": "MID",
    "ST": "ATT", "CF": "ATT", "LW": "ATT", "RW": "ATT",
}


def bucket_positions(positions):
    first = str(positions).split(",")[0].strip()
    return POSITION_BUCKETS.get(first, "MID")


def esd_rows():
    with sqlite3.connect(ESD_PATH) as connection:
        attributes = pd.read_sql(
            """SELECT player_api_id, date, overall_rating, potential
               FROM Player_Attributes
               WHERE overall_rating IS NOT NULL""", connection)
        players = pd.read_sql(
            "SELECT player_api_id, player_name, birthday FROM Player",
            connection)
    frame = attributes.merge(players, on="player_api_id", how="left")
    frame["effective_date"] = pd.to_datetime(frame["date"]).dt.normalize()
    birthday = pd.to_datetime(frame["birthday"])
    frame["age"] = ((frame["effective_date"] - birthday).dt.days
                    / 365.25).round(1)
    out = pd.DataFrame({
        "player_key": "esd:" + frame["player_api_id"].astype(str),
        "source": "esd",
        "effective_date": frame["effective_date"],
        "player_name": frame["player_name"],
        "overall": frame["overall_rating"],
        "potential": frame["potential"],
        "age": frame["age"],
        "position_bucket": np.nan,
        "league": np.nan,
        "club_team_id": np.nan,
    })
    return out.sort_values(["player_key", "effective_date"]).drop_duplicates(
        ["player_key", "effective_date"], keep="last")


def sofifa_rows():
    if not FIFA_FILTERED_PATH.exists():
        print("Filtered FIFA CSV absent; writing ESD ratings only. "
              "Rerun after download_extended_data.py completes.")
        return None
    frame = pd.read_csv(FIFA_FILTERED_PATH, low_memory=False,
                        encoding="utf-8")
    alias = pd.read_csv(PROCESSED_DIR / "alias_map_extended.csv",
                        encoding="utf-8")
    alias = alias[alias["source"] == "fifa"].dropna(
        subset=["canonical_team_id"])
    club_map = alias.drop_duplicates(["league", "source_name"]).set_index(
        ["league", "source_name"])["canonical_team_id"]

    frame["league"] = frame["league_id"].map(FIFA_LEAGUE_IDS)
    frame = frame.dropna(subset=["league", "overall", "fifa_update_date"])
    keys = pd.MultiIndex.from_frame(frame[["league", "club_name"]])
    frame["canonical_club"] = club_map.reindex(keys).to_numpy()
    unmatched = frame["canonical_club"].isna().mean()
    print(f"sofifa rows without a canonical club: {unmatched:.2%}")

    out = pd.DataFrame({
        "player_key": "sofifa:" + frame["player_id"].astype(int).astype(str),
        "source": "sofifa",
        "effective_date": pd.to_datetime(
            frame["fifa_update_date"]).dt.normalize(),
        "player_name": frame["short_name"],
        "overall": frame["overall"],
        "potential": frame["potential"],
        "age": frame["age"],
        "position_bucket": frame["player_positions"].map(bucket_positions),
        "league": frame["league"],
        "club_team_id": frame["canonical_club"],
    })
    return out.sort_values(["player_key", "effective_date"]).drop_duplicates(
        ["player_key", "effective_date"], keep="last")


def main():
    assert ESD_PATH.exists(), f"Missing {ESD_PATH}"
    frames = [esd_rows()]
    sofifa = sofifa_rows()
    if sofifa is not None:
        frames.append(sofifa)
    ratings = pd.concat(frames, ignore_index=True)

    assert (ratings["overall"].between(20, 99)).all(), \
        "overall rating outside a plausible range"
    assert not ratings.duplicated(["player_key", "effective_date"]).any()

    output = PROCESSED_DIR / "player_ratings.csv"
    ratings.to_csv(output, index=False, encoding="utf-8")
    span = (ratings["effective_date"].min().date(),
            ratings["effective_date"].max().date())
    print(f"Player-rating snapshots: {len(ratings)} rows, "
          f"{ratings['player_key'].nunique()} players, {span[0]} -> {span[1]}")
    print(ratings.groupby("source").size().to_string())
    print(f"Wrote {output}")


if __name__ == "__main__":
    main()
