"""Run from the repository root with:

    python src/pipeline/test_team_ratings.py
"""
from pathlib import Path
import math
import sys

import pandas as pd

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

from build_team_ratings import (ELO_HOME_ADV, ELO_K, PI_C, PI_GAMMA,  # noqa: E402
                                PI_LAMBDA, elo_expected, elo_update,
                                pi_expected_gd, pi_update, run_league)

PASSED = []


def check(name, condition):
    assert condition, f"FAIL {name}"
    PASSED.append(name)
    print(f"PASS {name}")


def close(a, b, tol=1e-9):
    return abs(a - b) < tol


expected = 1.0 / (1.0 + 10.0 ** (-(1500 + ELO_HOME_ADV - 1500) / 400.0))
check("elo_home_advantage_shifts_expectation",
      close(elo_expected(1500, 1500), expected) and expected > 0.5)

home_after, away_after = elo_update(1500, 1500, 2, 0)
check("elo_home_win_moves_k_times_surprise",
      close(home_after, 1500 + ELO_K * (1.0 - expected))
      and close(away_after, 1500 - ELO_K * (1.0 - expected)))

home_after, away_after = elo_update(1500, 1500, 1, 1)
check("elo_draw_punishes_the_favourite",
      home_after < 1500 < away_after
      and close(home_after - 1500, ELO_K * (0.5 - expected)))

check("elo_zero_sum",
      close(sum(elo_update(1622, 1498, 0, 3)), 1622 + 1498))

check("pi_zero_rating_expects_zero_goal_diff", close(pi_expected_gd(0.0), 0.0))
check("pi_expectation_is_odd", close(pi_expected_gd(-1.5),
                                     -pi_expected_gd(1.5)))

home = {"h": 0.0, "a": 0.0}
away = {"h": 0.0, "a": 0.0}
home_new, away_new = pi_update(home, away, 3, 1)
psi = PI_C * math.log10(1.0 + 2.0)
check("pi_home_win_raises_home_rating_by_lambda_psi",
      close(home_new["h"], PI_LAMBDA * psi)
      and close(home_new["a"], PI_GAMMA * PI_LAMBDA * psi))
check("pi_update_is_antisymmetric",
      close(away_new["a"], -PI_LAMBDA * psi)
      and close(away_new["h"], -PI_GAMMA * PI_LAMBDA * psi))

expected_gd = pi_expected_gd(home_new["h"]) - pi_expected_gd(away_new["a"])
error = (1 - 1) - expected_gd
psi2 = PI_C * math.log10(1.0 + abs(error))
home2, _ = pi_update(home_new, away_new, 1, 1)
check("pi_draw_pulls_leader_back",
      close(home2["h"], home_new["h"] - PI_LAMBDA * psi2))

matches = pd.DataFrame({
    "match_id": [1, 2, 3, 4],
    "season": ["2008/2009"] * 3 + ["2009/2010"],
    "home_team_id": [10, 20, 10, 10],
    "away_team_id": [20, 10, 20, 20],
    "home_score": [2, 0, 1, 0],
    "away_score": [0, 2, 1, 1],
})
rows = run_league(matches)
check("run_league_first_match_starts_flat",
      rows[0]["elo_home_pre"] == 1500.0 and rows[0]["pi_expected_gd"] == 0.0)
check("run_league_second_match_sees_first_result",
      rows[1]["elo_home_pre"] < 1500.0 < rows[1]["elo_away_pre"])
exp1 = elo_expected(1500, 1500)
h1 = 1500 + ELO_K * (1 - exp1)
check("run_league_hand_computed_second_row",
      close(rows[1]["elo_away_pre"], round(h1, 2), tol=0.005))
check("run_league_season_reversion_compresses_toward_mean",
      abs(rows[3]["elo_home_pre"] - rows[3]["elo_away_pre"])
      < abs(rows[2]["elo_home_pre"] - rows[2]["elo_away_pre"]))

print(f"\n{len(PASSED)}/{len(PASSED)} team-rating tests passed")
