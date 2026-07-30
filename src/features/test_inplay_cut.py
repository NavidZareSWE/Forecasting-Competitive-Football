from pathlib import Path
import sys

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

from build_inplay_features import (
    effective_minute, prefix_length_at, assert_prefix,
    snapshot_state, build_match_snapshots, SNAPSHOT_MINUTES,
)


def _events(rows):
    columns = ["index", "minute", "type", "team_id", "is_goal",
               "shot_outcome", "shot_xg", "card"]
    return pd.DataFrame(rows, columns=columns)


def _match(home=1, away=2):
    # index-ordered events; note the corrupted minute=0 at index 5 (a real
    # StatsBomb artifact) sitting between minute-10 and minute-20 events.
    rows = [
        [0, 0, "Pass", 1, False, None, np.nan, None],
        [1, 5, "Shot", 1, True, "Goal", 0.4, None],       # home goal at 5'
        [2, 8, "Shot", 2, False, "Saved", 0.1, None],     # away SOT at 8'
        [3, 10, "Foul Committed", 2, False, None, np.nan, "Red Card"],  # away red 10'
        [4, 10, "Pass", 1, False, None, np.nan, None],
        [5, 0, "Pass", 2, False, None, np.nan, None],     # corrupted minute
        [6, 20, "Shot", 1, True, "Goal", 0.6, None],      # home goal at 20'
        [7, 40, "Own Goal For", 2, False, None, np.nan, None],  # away benefits 40'
    ]
    return _events(rows), home, away


def test_effective_minute_is_monotonic():
    events, _, _ = _match()
    eff = effective_minute(events.sort_values("index"))
    assert (np.diff(eff) >= 0).all(), "effective minute must be non-decreasing"


def test_corrupted_minute_not_pulled_early():
    events, _, _ = _match()
    eff = effective_minute(events.sort_values("index"))
    # the corrupted 0 at index 5 must take the running max (10), not 0
    assert eff[5] == 10, "corrupted minute was not repaired by the running max"


def test_cut_is_prefix_and_nested():
    events, _, _ = _match()
    eff = effective_minute(events.sort_values("index"))
    previous = 0
    for minute in SNAPSHOT_MINUTES:
        length = prefix_length_at(eff, minute)
        assert_prefix(eff, length, minute)
        assert length >= previous, "prefix lengths must be non-decreasing in t"
        previous = length


def test_no_future_event_leaks_into_snapshot():
    events, home, away = _match()
    eff = effective_minute(events.sort_values("index"))
    length = prefix_length_at(eff, 5)
    prefix = events.sort_values("index").iloc[:length]
    assert prefix["minute"].max() <= 5, "an event after t leaked into the t=5 snapshot"


def test_score_matches_prefix_only():
    events, home, away = _match()
    snaps = build_match_snapshots(events, home, away).set_index("snapshot_minute")
    # at 5': only the home goal has occurred
    assert snaps.loc[5, "inplay_home_goals"] == 1
    assert snaps.loc[5, "inplay_away_goals"] == 0
    # at 15': home 1, away 0 still (second home goal is at 20')
    assert snaps.loc[15, "inplay_goal_diff"] == 1
    # at 20': home 2
    assert snaps.loc[20, "inplay_home_goals"] == 2
    # at 40': away own-goal-for credited
    assert snaps.loc[40, "inplay_away_goals"] == 1


def test_man_advantage_sign():
    events, home, away = _match()
    snaps = build_match_snapshots(events, home, away).set_index("snapshot_minute")
    # away team sent off at 10' -> home has +1 man advantage from 10' onward
    assert snaps.loc[5, "inplay_man_advantage"] == 0
    assert snaps.loc[10, "inplay_man_advantage"] == 1
    assert snaps.loc[90, "inplay_man_advantage"] == 1


def test_all_snapshot_minutes_present():
    events, home, away = _match()
    snaps = build_match_snapshots(events, home, away)
    assert list(snaps["snapshot_minute"]) == SNAPSHOT_MINUTES, "missing scheduled snapshot"


if __name__ == "__main__":
    tests = [v for k, v in sorted(globals().items()) if k.startswith("test_")]
    passed = 0
    for test in tests:
        test()
        passed += 1
        print(f"PASS {test.__name__}")
    print(f"\n{passed}/{len(tests)} in-play cut tests passed")
