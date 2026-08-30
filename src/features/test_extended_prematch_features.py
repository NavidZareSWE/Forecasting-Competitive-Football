"""Run from the repository root with:

    python src/features/test_extended_prematch_features.py
"""
from pathlib import Path
import sys

import pandas as pd

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

from build_extended_prematch_features import (add_head_to_head_features,  # noqa: E402
                                              add_rolling_form_features,
                                              build_team_match_long_table,
                                              pivot_prematch)

PASSED = []


def check(name, condition):
    assert condition, f"FAIL {name}"
    PASSED.append(name)
    print(f"PASS {name}")


matches = pd.DataFrame({
    "match_id": [1, 2, 3, 4, 5],
    "match_date": pd.to_datetime(["2020-01-04", "2020-01-11", "2020-01-18",
                                  "2020-01-25", "2020-02-01"]),
    "home_team_id": [10, 20, 10, 30, 10],
    "away_team_id": [20, 10, 30, 10, 20],
    "home_score": [2, 1, 0, 2, 3],
    "away_score": [0, 1, 0, 1, 1],
})

long = build_team_match_long_table(matches)
check("long_two_rows_per_match", len(long) == 2 * len(matches))
row = long[(long["match_id"] == 1) & (long["venue"] == "away")].iloc[0]
check("long_away_mirrors_score",
      row["gf"] == 0 and row["ga"] == 2 and row["points"] == 0
      and row["win"] == 0)

long = add_rolling_form_features(long)
team10 = long[long["team_id"] == 10].sort_values("match_date")
check("form_first_match_is_nan", pd.isna(team10["form_points"].iloc[0]))
check("form_prior_mean_hand_computed",
      abs(team10["form_points"].iloc[4] - (3 + 1 + 1 + 0) / 4) < 1e-9)
check("rest_days_calendar_gap", team10["rest_days"].iloc[1] == 7)
check("played_prior_counts", list(team10["played_prior"]) == [0, 1, 2, 3, 4])
home10 = team10[team10["venue"] == "home"]
check("venue_form_home_only",
      pd.isna(home10["venue_form_gf"].iloc[0])
      and abs(home10["venue_form_gf"].iloc[2] - (2 + 0) / 2) < 1e-9)

long = add_head_to_head_features(long)
pair = long[(long["team_id"] == 10) & (long["opponent_id"] == 20)] \
    .sort_values("match_date")
check("h2h_first_meeting_is_nan", pd.isna(pair["h2h_points"].iloc[0]))
check("h2h_prior_mean", abs(pair["h2h_points"].iloc[2] - (3 + 1) / 2) < 1e-9)

features = pivot_prematch(long)
check("pivot_one_row_per_match", features["match_id"].is_unique
      and len(features) == len(matches))
row5 = features[features["match_id"] == 5].iloc[0]
check("pivot_diff_is_home_minus_away",
      abs(row5["diff_form_points"]
          - (row5["home_form_points"] - row5["away_form_points"])) < 1e-9)

print(f"\n{len(PASSED)}/{len(PASSED)} extended pre-match tests passed")
