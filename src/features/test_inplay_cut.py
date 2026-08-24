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


def test_prefix_counters_match_hand_count():
    events, home, away = _match()
    snaps = build_match_snapshots(events, home, away).set_index("snapshot_minute")
    # away's foul at 10' carries the red card: foul and card diffs go to -1
    # from 10' onward and never earlier.
    assert snaps.loc[5, "inplay_foul_diff"] == 0
    assert snaps.loc[10, "inplay_foul_diff"] == -1
    assert snaps.loc[90, "inplay_card_diff"] == -1
    # no pressures or corners in the fixture: the optional-column path must
    # produce zeros, not crash.
    assert (snaps["inplay_pressure_diff"] == 0).all()
    assert (snaps["inplay_corner_diff"] == 0).all()


def test_recent_window_counts_and_momentum():
    events, home, away = _match()
    snaps = build_match_snapshots(events, home, away).set_index("snapshot_minute")
    # at 25': recent window (15, 25] holds only the 20' home shot (xG 0.6);
    # the previous window [5, 15) holds the 5' home shot (0.4) and the 8'
    # away shot (0.1), so momentum = 0.6 - (0.4 - 0.1) = 0.3.
    assert abs(snaps.loc[25, "inplay_recent_xg_diff"] - 0.6) < 1e-9
    assert abs(snaps.loc[25, "inplay_momentum_xg_diff"] - 0.3) < 1e-9
    assert snaps.loc[25, "inplay_recent_shot_diff"] == 1
    # one home event in a 10-minute window -> 0.1 events per minute, all home.
    assert abs(snaps.loc[25, "inplay_recent_event_rate_home"] - 0.1) < 1e-9
    assert snaps.loc[25, "inplay_recent_event_rate_away"] == 0.0
    assert snaps.loc[25, "inplay_recent_event_share_home"] == 1.0
    # an empty window reports the neutral share, not a division error.
    assert snaps.loc[60, "inplay_recent_event_share_home"] == 0.5


def test_corner_counts_use_pass_type():
    rows = pd.DataFrame({
        "index": [0, 1, 2],
        "minute": [3, 30, 50],
        "type": ["Pass", "Pass", "Pass"],
        "team_id": [1, 1, 2],
        "is_goal": [False, False, False],
        "shot_outcome": [None, None, None],
        "shot_xg": [np.nan, np.nan, np.nan],
        "card": [None, None, None],
        "pass_type": ["Corner", None, "Corner"],
    })
    snaps = build_match_snapshots(rows, 1, 2).set_index("snapshot_minute")
    assert snaps.loc[0, "inplay_corner_diff"] == 0
    assert snaps.loc[5, "inplay_corner_diff"] == 1
    assert snaps.loc[50, "inplay_corner_diff"] == 0   # 1 home - 1 away
    # the recent window sees the 50' corner at 50' but not at 70'.
    assert snaps.loc[50, "inplay_recent_corner_diff"] == -1
    assert snaps.loc[70, "inplay_recent_corner_diff"] == 0


def test_new_features_are_prefix_only():
    events, home, away = _match()
    baseline = build_match_snapshots(events, home, away)
    tampered = events.copy()
    # rewrite the 40' event into a late flurry of fouls, cards and shots
    tampered.loc[tampered["index"] == 7,
                 ["type", "card", "minute"]] = ["Foul Committed", "Red Card", 80]
    after = build_match_snapshots(tampered, home, away)
    # the tampered event sits after 35' in both versions, so every snapshot
    # at or before 35' must be identical in every feature column.
    early_before = baseline[baseline["snapshot_minute"] <= 35].reset_index(drop=True)
    early_after = after[after["snapshot_minute"] <= 35].reset_index(drop=True)
    pd.testing.assert_frame_equal(early_before, early_after)


if __name__ == "__main__":
    tests = [v for k, v in sorted(globals().items()) if k.startswith("test_")]
    passed = 0
    for test in tests:
        test()
        passed += 1
        print(f"PASS {test.__name__}")
    print(f"\n{passed}/{len(tests)} in-play cut tests passed")
