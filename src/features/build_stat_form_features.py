"""Run from the repository root with:

    python src/features/build_stat_form_features.py
"""
from pathlib import Path
import sys

import pandas as pd

HERE = Path(__file__).resolve().parent
PROJECT = HERE.parent
sys.path.insert(0, str(PROJECT / "pipeline"))

from download_extended_data import read_football_data_csv  # noqa: E402

PROCESSED_DIR = PROJECT / "reports" / "processed"
FOOTBALL_DATA_DIR = PROJECT / "data" / "Football_Data"

ROLLING_WINDOW = 5
STAT_COLUMNS = {"HS": "shots", "AS": "shots", "HST": "sot", "AST": "sot",
                "HC": "corners", "AC": "corners", "HF": "fouls", "AF": "fouls",
                "HY": "yellows", "AY": "yellows", "HR": "reds", "AR": "reds"}
QUANTITIES = ["shots", "sot", "corners", "fouls", "yellows", "reds"]
DIV_TO_LEAGUE = {"E0": "Premier League", "SP1": "La Liga", "I1": "Serie A",
                 "F1": "Ligue 1", "D1": "Bundesliga", "N1": "Eredivisie",
                 "P1": "Primeira Liga", "B1": "Pro League",
                 "SC0": "Scottish Premiership"}


def load_alias_maps():
    alias = pd.read_csv(PROCESSED_DIR / "alias_map_extended.csv",
                        encoding="utf-8")
    fd = alias[alias["source"] == "football_data"].set_index(
        ["league", "season", "source_name"])["canonical_team_id"]
    sb = alias[alias["source"] == "statsbomb"].set_index(
        ["league", "source_name"])["canonical_team_id"]
    legacy = pd.read_csv(PROCESSED_DIR / "alias_map.csv", encoding="utf-8")
    sb_2016 = {}
    for row in legacy.itertuples():
        key = (row.league, row.sb_name)
        if key in sb.index:
            sb_2016[(row.league, row.fd_name)] = int(sb.loc[key])
    return fd, sb_2016


def stat_rows():
    fd_alias, sb_2016 = load_alias_maps()
    rows = []

    for path in sorted(FOOTBALL_DATA_DIR.glob("*.csv")):
        frame = read_football_data_csv(path).dropna(
            subset=["HomeTeam", "AwayTeam"])
        div = str(frame["Div"].iloc[0])
        league = DIV_TO_LEAGUE.get(div)
        if league is None:
            continue
        frame["match_date"] = pd.to_datetime(frame["Date"], dayfirst=True,
                                             format="mixed").dt.normalize()
        for r in frame.itertuples():
            home = sb_2016.get((league, r.HomeTeam))
            away = sb_2016.get((league, r.AwayTeam))
            if home is None or away is None:
                continue
            rows.append(_pair(r, league, r.match_date, home, away))

    for season_dir in sorted(d for d in FOOTBALL_DATA_DIR.iterdir()
                             if d.is_dir()):
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
                home = fd_alias.get((league, season, r.HomeTeam))
                away = fd_alias.get((league, season, r.AwayTeam))
                if home is None or away is None:
                    continue
                rows.append(_pair(r, league, r.match_date, int(home),
                                  int(away)))
    return [r for pair in rows for r in pair]


def _pair(r, league, date, home_id, away_id):
    def value(column):
        raw = getattr(r, column, None)
        try:
            return float(raw)
        except (TypeError, ValueError):
            return float("nan")

    home = {"league": league, "match_date": date, "team_id": home_id,
            "opponent_id": away_id, "venue": "home"}
    away = {"league": league, "match_date": date, "team_id": away_id,
            "opponent_id": home_id, "venue": "away"}
    for column, quantity in STAT_COLUMNS.items():
        target = home if column.startswith("H") else away
        other = away if column.startswith("H") else home
        target[f"{quantity}_f"] = value(column)
        other[f"{quantity}_a"] = value(column)
    return home, away


def main():
    long = pd.DataFrame(stat_rows())
    long = long.dropna(subset=[f"{q}_f" for q in QUANTITIES], how="all")
    long = long.sort_values(["team_id", "match_date"]).reset_index(drop=True)
    long = long.drop_duplicates(["team_id", "match_date"])

    grouped = long.groupby("team_id", group_keys=False)
    stat_columns = [f"{q}_{side}" for q in QUANTITIES for side in ["f", "a"]]
    for column in stat_columns:
        long[f"form_stat_{column}"] = grouped[column].apply(
            lambda s: s.shift(1).rolling(ROLLING_WINDOW, min_periods=1).mean())

    first_rows = long.groupby("team_id").head(1)
    rolled = [f"form_stat_{c}" for c in stat_columns]
    assert first_rows[rolled].isna().all().all(), \
        "prior-only violation: first match carries stat form"

    store = pd.read_csv(PROCESSED_DIR / "extended_match_store.csv",
                        encoding="utf-8", parse_dates=["match_date"])
    keyed = long.set_index(["team_id", "match_date"])[rolled]
    sides = []
    for side in ["home", "away"]:
        index = pd.MultiIndex.from_frame(
            store[[f"{side}_team_id", "match_date"]])
        block = keyed.reindex(index).reset_index(drop=True)
        block.index = store["match_id"]
        sides.append(block.add_prefix(f"{side}_"))
    features = sides[0].join(sides[1])
    for column in rolled:
        features[f"diff_{column}"] = (features[f"home_{column}"]
                                      - features[f"away_{column}"])
    features.insert(0, "stat_form_available",
                    features[f"home_{rolled[0]}"].notna().astype(int))
    features = features.reset_index()

    coverage = features.set_index("match_id")["stat_form_available"]
    by_era = store.set_index("match_id")["era"].to_frame().join(coverage)
    print(by_era.groupby("era")["stat_form_available"].mean().round(3)
          .to_string())
    fd_cov = by_era.loc[by_era["era"] == "football_data",
                        "stat_form_available"].mean()
    assert fd_cov >= 0.95, f"football-data stat coverage only {fd_cov:.1%}"

    output = PROCESSED_DIR / "stat_form_features.csv"
    features.to_csv(output, index=False, encoding="utf-8")
    print(f"Stat-form rows: {len(features)} matches, "
          f"{features.shape[1]} columns")
    print(f"Wrote {output}")


if __name__ == "__main__":
    main()
