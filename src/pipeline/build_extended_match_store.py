"""Run from the repository root with:

    python src/pipeline/build_extended_match_store.py
"""
from pathlib import Path
import sqlite3
import sys

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

from download_extended_data import read_football_data_csv  # noqa: E402

PROJECT = HERE.parent
PROCESSED_DIR = PROJECT / "reports" / "processed"
ESD_PATH = PROJECT / "data" / "european_soccer_db" / "database.sqlite"
FOOTBALL_DATA_DIR = PROJECT / "data" / "Football_Data"

ESD_ID_BASE = 1_000_000_000
FD_ID_BASE = 2_000_000_000
MARGIN_CLIP = 5

STATSBOMB_LEAGUES = {"Premier League", "La Liga", "Serie A", "Ligue 1"}
STATSBOMB_SEASON = "2015/2016"

ESD_LEAGUE_TO_CANONICAL = {
    "England Premier League": "Premier League",
    "Spain LIGA BBVA": "La Liga",
    "Italy Serie A": "Serie A",
    "France Ligue 1": "Ligue 1",
    "Germany 1. Bundesliga": "Bundesliga",
    "Netherlands Eredivisie": "Eredivisie",
    "Portugal Liga ZON Sagres": "Primeira Liga",
    "Belgium Jupiler League": "Pro League",
    "Scotland Premier League": "Scottish Premiership",
    "Poland Ekstraklasa": "Ekstraklasa",
    "Switzerland Super League": "Swiss Super League",
}
DIV_TO_LEAGUE = {"E0": "Premier League", "SP1": "La Liga", "I1": "Serie A",
                 "F1": "Ligue 1", "D1": "Bundesliga", "N1": "Eredivisie",
                 "P1": "Primeira Liga", "B1": "Pro League",
                 "SC0": "Scottish Premiership"}


def derive_labels(frame):
    frame = frame.copy()
    frame["result"] = np.select(
        [frame["home_score"] > frame["away_score"],
         frame["home_score"] < frame["away_score"]], ["H", "A"], default="D")
    frame["margin_raw"] = frame["home_score"] - frame["away_score"]
    frame["margin"] = frame["margin_raw"].clip(-MARGIN_CLIP, MARGIN_CLIP)
    return frame


def load_alias(source):
    alias = pd.read_csv(PROCESSED_DIR / "alias_map_extended.csv",
                        encoding="utf-8")
    return alias[alias["source"] == source]


def load_esd_matches():
    with sqlite3.connect(ESD_PATH) as connection:
        matches = pd.read_sql(
            """SELECT m.match_api_id, l.name AS esd_league, m.season, m.date,
                      m.home_team_api_id, m.away_team_api_id,
                      m.home_team_goal AS home_score,
                      m.away_team_goal AS away_score,
                      m.home_player_1 IS NOT NULL
                      AND m.away_player_1 IS NOT NULL AS has_lineups
               FROM Match m JOIN League l ON l.id = m.league_id""", connection)
        teams = pd.read_sql(
            "SELECT team_api_id, team_long_name FROM Team", connection)
    matches["league"] = matches["esd_league"].map(ESD_LEAGUE_TO_CANONICAL)
    assert matches["league"].notna().all()
    matches["match_date"] = pd.to_datetime(matches["date"]).dt.normalize()
    names = teams.set_index("team_api_id")["team_long_name"]
    matches["home_team"] = matches["home_team_api_id"].map(names)
    matches["away_team"] = matches["away_team_api_id"].map(names)
    return matches


def build_statsbomb_rows(esd_matches):
    store = pd.read_csv(PROCESSED_DIR / "match_store.csv", encoding="utf-8",
                        parse_dates=["match_date"])
    alias = load_alias("statsbomb").set_index(["league", "source_name"])
    ids = alias["canonical_team_id"]

    frame = pd.DataFrame({
        "match_id": store["match_id"],
        "source": "statsbomb",
        "source_match_id": store["match_id"],
        "league": store["competition_name"],
        "season": STATSBOMB_SEASON,
        "match_date": store["match_date"].dt.normalize(),
        "home_team_id": [ids[(l, n)] for l, n in
                         zip(store["competition_name"], store["home_team"])],
        "away_team_id": [ids[(l, n)] for l, n in
                         zip(store["competition_name"], store["away_team"])],
        "home_team": store["home_team"],
        "away_team": store["away_team"],
        "home_score": store["home_score"],
        "away_score": store["away_score"],
        "era": "statsbomb",
        "has_events": True,
    })
    frame = derive_labels(frame)

    assert (frame["result"] == store["result"]).all(), \
        "extended-store labels disagree with match_store.csv"
    assert (frame["margin"] == store["margin"]).all(), \
        "extended-store margins disagree with match_store.csv"

    esd_cell = esd_matches[
        (esd_matches["season"] == STATSBOMB_SEASON)
        & esd_matches["league"].isin(STATSBOMB_LEAGUES)]
    esd_key = esd_cell.set_index(
        ["league", "home_team_api_id", "away_team_api_id"])
    counterpart, has_lineups = [], []
    for row in frame.itertuples():
        key = (row.league, row.home_team_id, row.away_team_id)
        if key in esd_key.index:
            candidate = esd_key.loc[[key]]
            gap = (candidate["match_date"] - row.match_date).abs().dt.days
            candidate = candidate[gap <= 1]
            if len(candidate):
                counterpart.append(int(candidate["match_api_id"].iloc[0]))
                has_lineups.append(bool(candidate["has_lineups"].iloc[0]))
                continue
        counterpart.append(np.nan)
        has_lineups.append(False)
    frame["esd_match_api_id"] = counterpart
    frame["has_lineups"] = has_lineups
    found = frame["esd_match_api_id"].notna().mean()
    print(f"StatsBomb rows with an ESD counterpart: {found:.1%}")
    assert found >= 0.95, "too many 2015/16 fixtures missing from ESD"
    return frame


def build_esd_rows(esd_matches):
    keep = ~(esd_matches["league"].isin(STATSBOMB_LEAGUES)
             & (esd_matches["season"] == STATSBOMB_SEASON))
    matches = esd_matches[keep].copy()
    frame = pd.DataFrame({
        "match_id": ESD_ID_BASE + matches["match_api_id"],
        "source": "esd",
        "source_match_id": matches["match_api_id"],
        "league": matches["league"],
        "season": matches["season"],
        "match_date": matches["match_date"],
        "home_team_id": matches["home_team_api_id"],
        "away_team_id": matches["away_team_api_id"],
        "home_team": matches["home_team"],
        "away_team": matches["away_team"],
        "home_score": matches["home_score"],
        "away_score": matches["away_score"],
        "era": "esd",
        "has_events": False,
        "esd_match_api_id": matches["match_api_id"],
        "has_lineups": matches["has_lineups"].astype(bool),
    })
    return derive_labels(frame)


def build_football_data_rows():
    alias = load_alias("football_data").set_index(
        ["league", "season", "source_name"])
    ids = alias["canonical_team_id"]
    display = alias["canonical_name"]
    rows = []
    for season_dir in sorted(FOOTBALL_DATA_DIR.iterdir()):
        if not season_dir.is_dir():
            continue
        season = f"20{season_dir.name[:2]}/20{season_dir.name[2:]}"
        for path in sorted(season_dir.glob("*.csv")):
            league = DIV_TO_LEAGUE.get(path.stem)
            if league is None:
                continue
            frame = read_football_data_csv(path)
            frame = frame.dropna(subset=["HomeTeam", "AwayTeam", "FTHG", "FTAG"])
            frame["match_date"] = pd.to_datetime(frame["Date"], dayfirst=True,
                                                 format="mixed").dt.normalize()
            for r in frame.itertuples():
                home_key = (league, season, r.HomeTeam)
                away_key = (league, season, r.AwayTeam)
                rows.append({
                    "league": league, "season": season,
                    "match_date": r.match_date,
                    "home_team_id": int(ids[home_key]),
                    "away_team_id": int(ids[away_key]),
                    "home_team": display[home_key],
                    "away_team": display[away_key],
                    "home_score": int(r.FTHG), "away_score": int(r.FTAG),
                })
    frame = pd.DataFrame(rows).sort_values(
        ["league", "season", "match_date", "home_team", "away_team"]
    ).reset_index(drop=True)
    frame["match_id"] = FD_ID_BASE + frame.index
    frame["source"] = "football_data"
    frame["source_match_id"] = frame.index
    frame["era"] = "football_data"
    frame["has_events"] = False
    frame["esd_match_api_id"] = np.nan
    frame["has_lineups"] = False
    return derive_labels(frame)


COLUMNS = ["match_id", "source", "source_match_id", "league", "season",
           "match_date", "home_team_id", "away_team_id", "home_team",
           "away_team", "home_score", "away_score", "result", "margin_raw",
           "margin", "era", "has_events", "has_lineups", "esd_match_api_id"]


def main():
    for path in [ESD_PATH, PROCESSED_DIR / "alias_map_extended.csv",
                 PROCESSED_DIR / "match_store.csv"]:
        assert Path(path).exists(), f"Missing {path}; run earlier stages first."

    esd_matches = load_esd_matches()
    statsbomb = build_statsbomb_rows(esd_matches)
    esd = build_esd_rows(esd_matches)
    football_data = build_football_data_rows()
    store = pd.concat([statsbomb[COLUMNS], esd[COLUMNS],
                       football_data[COLUMNS]], ignore_index=True)
    store = store.sort_values(["match_date", "match_id"]).reset_index(drop=True)

    assert store["match_id"].is_unique, "duplicate match_id across sources"
    cell_sources = store.groupby(["league", "season"])["source"].nunique()
    assert (cell_sources == 1).all(), \
        f"(league, season) cells with mixed sources:\n{cell_sources[cell_sources > 1]}"
    fixture = store.duplicated(
        ["league", "season", "match_date", "home_team_id", "away_team_id"])
    assert not fixture.any(), "duplicate fixture rows"
    assert store["result"].isin({"H", "D", "A"}).all()
    assert store["margin"].between(-MARGIN_CLIP, MARGIN_CLIP).all()
    sizes = store.groupby(["league", "season"]).size()
    undersized = sizes[sizes < 150]
    if len(undersized):
        print("Dropping incomplete (league, season) cells:")
        print(undersized.to_string())
        keep_index = pd.MultiIndex.from_frame(store[["league", "season"]])
        store = store[~keep_index.isin(undersized.index)].reset_index(drop=True)
    sizes = store.groupby(["league", "season"]).size()
    assert sizes.between(150, 420).all(), \
        f"implausible season sizes:\n{sizes[~sizes.between(150, 420)]}"

    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    output = PROCESSED_DIR / "extended_match_store.csv"
    store.to_csv(output, index=False, encoding="utf-8")
    print(f"Extended store: {len(store)} matches, "
          f"{store['league'].nunique()} leagues, "
          f"{store['season'].nunique()} seasons "
          f"({store['match_date'].min().date()} -> "
          f"{store['match_date'].max().date()})")
    print(store.groupby("era").size().to_string())
    print(f"Wrote {output}")


if __name__ == "__main__":
    main()
