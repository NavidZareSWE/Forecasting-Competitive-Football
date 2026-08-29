"""Run from the repository root with:

    python src/pipeline/build_team_registry.py
"""
from pathlib import Path
import sqlite3
import sys

import pandas as pd

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

from build_market_baseline import normalize, token_similarity, levenshtein  # noqa: E402
from download_extended_data import (FIFA_FILTERED_PATH, FIFA_LEAGUE_IDS,  # noqa: E402
                                    read_football_data_csv)

PROJECT = HERE.parent
PROCESSED_DIR = PROJECT / "reports" / "processed"
ESD_PATH = PROJECT / "data" / "european_soccer_db" / "database.sqlite"
FOOTBALL_DATA_DIR = PROJECT / "data" / "Football_Data"

SYNTHETIC_BASE = 900000
FUZZY_MIN_RATIO = 0.5

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

MANUAL_OVERRIDES = {
    ("La Liga", "Ath Madrid"): "atletico madrid",
    ("La Liga", "Ath Bilbao"): "athletic club de bilbao",
    ("Ligue 1", "Gazélec Ajaccio"): "gfc ajaccio",
    ("Ligue 1", "Gazelec"): "gfc ajaccio",
    ("Ligue 1", "Ajaccio"): "ac ajaccio",
    ("Ligue 1", "Ajaccio GFCO"): "gfc ajaccio",
    ("Ligue 1", "Rennes"): "stade rennais fc",
    ("Ligue 1", "Stade Malherbe Caen"): "sm caen",
    ("Serie A", "Inter Milan"): "inter",
    ("Serie A", "Inter"): "inter",
    ("Bundesliga", "Union Berlin"): None,
    ("Eredivisie", "FC Emmen"): None,
    ("La Liga", "Cadiz"): None,
    ("Primeira Liga", "Aves"): None,
    ("Primeira Liga", "AVS"): None,
    ("Pro League", "Beerschot VA"): None,
    ("La Liga", "Cádiz CF"): "cadiz",
    ("La Liga", "Levante Unión Deportiva"): "levante ud",
    ("Premier League", "Brighton & Hove Albion"): "brighton",
    ("Premier League", "Leeds United"): "leeds",
    ("Serie A", "U.S. Sassuolo Calcio"): "sassuolo",
    ("Serie A", "US Salernitana 1919"): "salernitana",
    ("Ligue 1", "Toulouse Football Club"): "toulouse fc",
    ("Ligue 1", "Clermont Foot 63"): "clermont",
    ("Ligue 1", "RC Strasbourg Alsace"): "strasbourg",
    ("Bundesliga", "Sport-Club Freiburg"): "sc freiburg",
    ("Primeira Liga", "Sporting Braga"): "sc braga",
    ("Primeira Liga", "Vitória FC"): "vitoria setubal",
    ("Primeira Liga", "Vitória SC"): "vitoria guimaraes",
    ("Primeira Liga", "Clube Sport Marítimo"): "cs maritimo",
    ("Primeira Liga", "Futebol Clube de Famalicão"): "famalicao",
    ("Primeira Liga", "Desportivo Aves"): "aves",
    ("Primeira Liga", "Desportivo das Aves"): "aves",
    ("Pro League", "Royal Antwerp FC"): "antwerp",
    ("Pro League", "Royal Charleroi S.C."): "sporting charleroi",
    ("Pro League", "Beerschot"): "beerschot va",
    ("Pro League", "Royale Union Saint-Gilloise"): "st. gilloise",
    ("Pro League", "Union Saint-Gilloise"): "st. gilloise",
    # Paris FC was promoted to Ligue 1 for 2025/26 and has no ESD entry, so it
    # needs a synthetic. It also sorts BEFORE "Paris SG", so without these two
    # lines it takes a synthetic id first and then wins the fuzzy match for
    # PSG - which would silently move Paris Saint-Germain off its real ESD id
    # 9847 and onto a brand-new club, taking 17 seasons of Elo with it. Same
    # failure class as Ath Madrid / Ath Bilbao above: no string metric
    # separates these two, so both sides are pinned by hand.
    ("Ligue 1", "Paris SG"): "paris saint germain",
    ("Ligue 1", "Paris FC"): None,
}


class LeaguePool:
    """Canonical teams of one league: ESD teams plus grown synthetics.

    Short names ("MAL", "VAL") take part in exact matching only - inside the
    fuzzy scorer a three-letter code ties unrelated clubs (Málaga/Mallorca
    both own "MAL"-ish prefixes).
    """

    def __init__(self, league):
        self.league = league
        self.names = {}
        self.fuzzy_names = {}
        self.display = {}
        self.id_source = {}

    def add(self, team_id, names, display, id_source, fuzzy_names=None):
        self.names[team_id] = [n for n in names if isinstance(n, str) and n]
        self.fuzzy_names[team_id] = ([n for n in fuzzy_names
                                      if isinstance(n, str) and n]
                                     if fuzzy_names is not None
                                     else self.names[team_id])
        self.display[team_id] = display
        self.id_source[team_id] = id_source

    def resolve(self, name):
        """-> (canonical_id, method, ratio) or (None, reason, ratio)."""
        forced_synthetic = ((self.league, name) in MANUAL_OVERRIDES
                            and MANUAL_OVERRIDES[(self.league, name)] is None)
        if not forced_synthetic:
            override = MANUAL_OVERRIDES.get((self.league, name))
            if override is not None:
                for team_id, names in self.names.items():
                    if override in {normalize(n) for n in names}:
                        return team_id, "override", 1.0
                raise AssertionError(
                    f"override {(self.league, name)} -> {override!r} matches "
                    "no canonical team")
        wanted = normalize(name)
        for team_id, names in self.names.items():
            if wanted in {normalize(n) for n in names}:
                return team_id, "exact", 1.0
        if forced_synthetic:
            return None, "forced_synthetic", 0.0
        scored = []
        for team_id, names in self.fuzzy_names.items():
            best = None
            for candidate in names:
                ratio, distance = token_similarity(name, candidate)
                edit = levenshtein(wanted, normalize(candidate))
                key = (ratio, -distance, -edit)
                if best is None or key > best:
                    best = key
            if best is not None:
                scored.append((best, team_id))
        if not scored:
            return None, "below_threshold", 0.0
        scored.sort(reverse=True)
        (ratio, neg_distance, neg_edit), team_id = scored[0]
        if ratio < FUZZY_MIN_RATIO:
            return None, "below_threshold", ratio
        if len(scored) > 1 and scored[1][0] == scored[0][0]:
            return None, "ambiguous", ratio
        return team_id, "fuzzy", ratio


def load_esd_pools():
    with sqlite3.connect(ESD_PATH) as connection:
        leagues = pd.read_sql("SELECT id, name FROM League", connection)
        teams = pd.read_sql(
            "SELECT team_api_id, team_long_name, team_short_name FROM Team",
            connection)
        appearances = pd.read_sql(
            """SELECT DISTINCT league_id, season, home_team_api_id AS team_api_id
               FROM Match
               UNION
               SELECT DISTINCT league_id, season, away_team_api_id FROM Match""",
            connection)
    leagues["league"] = leagues["name"].map(ESD_LEAGUE_TO_CANONICAL)
    assert leagues["league"].notna().all(), "unmapped ESD league name"
    appearances = appearances.merge(
        leagues[["id", "league"]], left_on="league_id", right_on="id")
    team_names = teams.set_index("team_api_id")

    pools = {league: LeaguePool(league) for league in leagues["league"]}
    for (league, team_id), _ in appearances.groupby(["league", "team_api_id"]):
        row = team_names.loc[team_id]
        pools[league].add(int(team_id),
                          [row["team_long_name"], row["team_short_name"]],
                          row["team_long_name"], "esd",
                          fuzzy_names=[row["team_long_name"]])
    season_teams = {
        (league, season): set(group["team_api_id"].astype(int))
        for (league, season), group in appearances.groupby(["league", "season"])}
    return pools, season_teams


def map_statsbomb(pools, season_teams, alias_rows):
    store = pd.read_csv(PROCESSED_DIR / "match_store.csv", encoding="utf-8")
    mapping = {}
    for league, group in store.groupby("competition_name"):
        names = sorted(set(group["home_team"]) | set(group["away_team"]))
        resolved = {}
        for name in names:
            team_id, method, ratio = pools[league].resolve(name)
            assert team_id is not None, \
                f"StatsBomb team {name!r} ({league}) unresolved: {method}"
            resolved[name] = team_id
            alias_rows.append({"source": "statsbomb", "league": league,
                               "season": "2015/2016", "source_name": name,
                               "canonical_team_id": team_id,
                               "canonical_name": pools[league].display[team_id],
                               "method": method, "ratio": round(ratio, 3)})
        expected = season_teams[(league, "2015/2016")]
        assert set(resolved.values()) == expected, (
            f"{league}: StatsBomb teams are not a bijection onto the ESD "
            f"2015/16 season ({sorted(set(resolved.values()) ^ expected)})")
        mapping.update({(league, k): v for k, v in resolved.items()})
    return mapping


def map_football_data(pools, alias_rows, synthetic_counter):
    for season_dir in sorted(FOOTBALL_DATA_DIR.iterdir()):
        if not season_dir.is_dir():
            continue
        season = f"20{season_dir.name[:2]}/20{season_dir.name[2:]}"
        for path in sorted(season_dir.glob("*.csv")):
            div = path.stem
            league = DIV_TO_LEAGUE.get(div)
            if league is None:
                continue
            frame = read_football_data_csv(path)
            frame = frame.dropna(subset=["HomeTeam", "AwayTeam"])
            names = sorted(set(frame["HomeTeam"]) | set(frame["AwayTeam"]))
            season_map = {}
            for name in names:
                team_id, method, ratio = pools[league].resolve(name)
                if team_id is None:
                    team_id = SYNTHETIC_BASE + synthetic_counter[0]
                    synthetic_counter[0] += 1
                    pools[league].add(team_id, [name], name, "synthetic")
                    method = f"synthetic({method})"
                season_map[name] = team_id
                alias_rows.append({"source": "football_data", "league": league,
                                   "season": season, "source_name": name,
                                   "canonical_team_id": team_id,
                                   "canonical_name": pools[league].display[team_id],
                                   "method": method, "ratio": round(ratio, 3)})
            assert len(set(season_map.values())) == len(season_map), (
                f"{league} {season}: two football-data names share one "
                f"canonical team: {season_map}")


def map_fifa_clubs(pools, alias_rows):
    if not FIFA_FILTERED_PATH.exists():
        print("Filtered FIFA CSV absent; skipping FIFA club aliases "
              "(rerun after download_extended_data.py completes)")
        return
    usecols = ["fifa_version", "club_name", "league_id"]
    frame = pd.read_csv(FIFA_FILTERED_PATH, usecols=usecols,
                        low_memory=False).dropna()
    frame = frame[frame["fifa_version"] >= 17]
    frame["league"] = frame["league_id"].map(FIFA_LEAGUE_IDS)
    frame = frame.dropna(subset=["league"])
    clubs = frame[["fifa_version", "league", "club_name"]].drop_duplicates()
    unmatched = 0
    for (version, league), group in clubs.groupby(["fifa_version", "league"]):
        season = f"20{int(version) - 1}/20{int(version)}"
        seen = {}
        for name in sorted(group["club_name"]):
            team_id, method, ratio = pools[league].resolve(name)
            if team_id is None:
                unmatched += 1
            else:
                seen[name] = team_id
            alias_rows.append({"source": "fifa", "league": league,
                               "season": season, "source_name": name,
                               "canonical_team_id": team_id,
                               "canonical_name": (pools[league].display[team_id]
                                                  if team_id else None),
                               "method": method, "ratio": round(ratio, 3)})
        del seen
    if unmatched:
        print(f"FIFA clubs left unmatched (logged, not fatal): {unmatched}")


def main():
    assert ESD_PATH.exists(), \
        f"Missing {ESD_PATH}. Run download_extended_data.py first."
    pools, season_teams = load_esd_pools()
    alias_rows = []

    map_statsbomb(pools, season_teams, alias_rows)
    synthetic_counter = [0]
    map_football_data(pools, alias_rows, synthetic_counter)
    map_fifa_clubs(pools, alias_rows)

    registry_rows = []
    for league, pool in pools.items():
        for team_id, display in pool.display.items():
            registry_rows.append({"canonical_team_id": team_id,
                                  "canonical_name": display,
                                  "league": league,
                                  "id_source": pool.id_source[team_id]})
    registry = pd.DataFrame(registry_rows).sort_values(
        ["league", "canonical_team_id"])
    assert registry["canonical_team_id"].is_unique, \
        "a canonical team id appears in two leagues"

    alias_map = pd.DataFrame(alias_rows)
    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    registry.to_csv(PROCESSED_DIR / "team_registry.csv", index=False,
                    encoding="utf-8")
    alias_map.to_csv(PROCESSED_DIR / "alias_map_extended.csv", index=False,
                     encoding="utf-8")

    fuzzy = alias_map[alias_map["method"] == "fuzzy"]
    synthetic = alias_map[alias_map["method"].str.startswith("synthetic")]
    print(f"Registry teams: {len(registry)} "
          f"({(registry['id_source'] == 'synthetic').sum()} synthetic)")
    print(f"Alias rows: {len(alias_map)} "
          f"(fuzzy {len(fuzzy)}, synthetic {synthetic['source_name'].nunique()} "
          "distinct names)")
    low = fuzzy[fuzzy["ratio"] < 0.75]
    if len(low):
        print("Low-confidence fuzzy matches to eyeball:")
        print(low[["source", "league", "season", "source_name",
                   "canonical_name", "ratio"]].drop_duplicates(
            ["source", "league", "source_name"]).to_string(index=False))
    print("Wrote team_registry.csv, alias_map_extended.csv")


if __name__ == "__main__":
    main()
