"""Run from the repository root with:

    python src/pipeline/build_team_ratings.py
"""
from pathlib import Path
import json
import math

import pandas as pd

HERE = Path(__file__).resolve().parent
PROJECT = HERE.parent
PROCESSED_DIR = PROJECT / "reports" / "processed"

ELO_INIT = 1500.0
ELO_K = 20.0
ELO_HOME_ADV = 70.0
ELO_SEASON_CARRY = 0.75
ELO_NEW_TEAM_PENALTY = 50.0

PI_C = 3.0
PI_LAMBDA = 0.035
PI_GAMMA = 0.7


def elo_expected(home_rating, away_rating):
    return 1.0 / (1.0 + 10.0 ** (-(home_rating + ELO_HOME_ADV - away_rating)
                                  / 400.0))


def elo_update(home_rating, away_rating, home_score, away_score):
    observed = 1.0 if home_score > away_score else \
        0.0 if home_score < away_score else 0.5
    delta = ELO_K * (observed - elo_expected(home_rating, away_rating))
    return home_rating + delta, away_rating - delta


def pi_expected_gd(rating):
    sign = 1.0 if rating >= 0 else -1.0
    return sign * (10.0 ** (abs(rating) / PI_C) - 1.0)


def pi_update(home, away, home_score, away_score):
    """home/away are dicts {"h": ..., "a": ...}; returns updated copies."""
    expected = pi_expected_gd(home["h"]) - pi_expected_gd(away["a"])
    error = (home_score - away_score) - expected
    psi = PI_C * math.log10(1.0 + abs(error))
    signed = psi if error >= 0 else -psi
    home_new = {"h": home["h"] + PI_LAMBDA * signed,
                "a": home["a"] + PI_GAMMA * PI_LAMBDA * signed}
    away_new = {"a": away["a"] - PI_LAMBDA * signed,
                "h": away["h"] - PI_GAMMA * PI_LAMBDA * signed}
    return home_new, away_new


def run_league(matches):
    elo, pi = {}, {}
    season = None
    first_season = matches["season"].iloc[0]
    rows = []

    def elo_get(team_id, current_season):
        if team_id not in elo:
            if current_season == first_season or not elo:
                elo[team_id] = ELO_INIT
            else:
                mean = sum(elo.values()) / len(elo)
                elo[team_id] = mean - ELO_NEW_TEAM_PENALTY
        return elo[team_id]

    for row in matches.itertuples():
        if season is not None and row.season != season and elo:
            mean = sum(elo.values()) / len(elo)
            for team_id in elo:
                elo[team_id] = (ELO_SEASON_CARRY * elo[team_id]
                                + (1.0 - ELO_SEASON_CARRY) * mean)
        season = row.season

        home_elo = elo_get(row.home_team_id, row.season)
        away_elo = elo_get(row.away_team_id, row.season)
        home_pi = pi.setdefault(row.home_team_id, {"h": 0.0, "a": 0.0})
        away_pi = pi.setdefault(row.away_team_id, {"h": 0.0, "a": 0.0})
        rows.append({
            "match_id": row.match_id,
            "elo_home_pre": round(home_elo, 2),
            "elo_away_pre": round(away_elo, 2),
            "elo_diff": round(home_elo - away_elo, 2),
            "elo_expected_home": round(elo_expected(home_elo, away_elo), 4),
            "pi_home_home_pre": round(home_pi["h"], 4),
            "pi_home_away_pre": round(home_pi["a"], 4),
            "pi_away_home_pre": round(away_pi["h"], 4),
            "pi_away_away_pre": round(away_pi["a"], 4),
            "pi_expected_gd": round(pi_expected_gd(home_pi["h"])
                                    - pi_expected_gd(away_pi["a"]), 4),
        })
        elo[row.home_team_id], elo[row.away_team_id] = elo_update(
            home_elo, away_elo, row.home_score, row.away_score)
        pi[row.home_team_id], pi[row.away_team_id] = pi_update(
            home_pi, away_pi, row.home_score, row.away_score)
    return rows


def load_tuned_params():
    global ELO_K, ELO_HOME_ADV, PI_LAMBDA
    path = PROCESSED_DIR / "rating_params.json"
    if not path.exists():
        return
    with open(path, encoding="utf-8") as source:
        params = json.load(source)
    ELO_K = float(params.get("ELO_K", ELO_K))
    ELO_HOME_ADV = float(params.get("ELO_HOME_ADV", ELO_HOME_ADV))
    PI_LAMBDA = float(params.get("PI_LAMBDA", PI_LAMBDA))
    print(f"Tuned rating parameters: K={ELO_K}, HA={ELO_HOME_ADV}, "
          f"lambda={PI_LAMBDA}")


def main():
    load_tuned_params()
    store = pd.read_csv(PROCESSED_DIR / "extended_match_store.csv",
                        encoding="utf-8", parse_dates=["match_date"])
    rows = []
    for _, league_matches in store.groupby("league"):
        ordered = league_matches.sort_values(["match_date", "match_id"])
        rows.extend(run_league(ordered))
    ratings = pd.DataFrame(rows)

    assert len(ratings) == len(store)
    assert ratings["match_id"].is_unique
    assert ratings["elo_expected_home"].between(0, 1).all()

    output = PROCESSED_DIR / "team_ratings.csv"
    ratings.to_csv(output, index=False, encoding="utf-8")
    merged = ratings.merge(store[["match_id", "result"]], on="match_id")
    home_wins = (merged["result"] == "H").mean()
    predicted = (merged["elo_expected_home"] > 0.5).mean()
    print(f"Ratings for {len(ratings)} matches; base home-win rate "
          f"{home_wins:.3f}, share with Elo favouring home {predicted:.3f}")
    print(f"Wrote {output}")


if __name__ == "__main__":
    main()
