"""Tests for the pre-match feature builder (Phase 5A).

Every expected value below is computed by hand from the fixture, not from the
code under test.

    python src/features/test_prematch_features.py
"""

import numpy as np
import pandas as pd

from build_prematch_features import (ROLLING_WINDOW, add_event_aggregates,
                                     add_head_to_head,
                                     add_rolling_form_features,
                                     build_event_aggregates,
                                     build_prematch_features,
                                     build_team_match_long_table)


# --- Fixture ----------------------------------------------------------------
# Team 1 plays six matches; team 2 is the opponent every time, so the fixture
# also exercises the home/away pivot. Scores chosen so no two windows coincide.
FIXTURE_SCORES = [(3, 0), (1, 1), (0, 2), (2, 2), (4, 1), (0, 0), (1, 0)]
FIXTURE_XG = [(2.5, 0.4), (1.1, 1.3), (0.6, 1.9), (1.8, 2.1), (3.2, 0.9),
              (0.7, 0.8), (1.4, 0.5)]
FIXTURE_DAYS = [0, 7, 14, 28, 35, 36, 43]


def make_fixture():
    rows = []
    for position, ((home_score, away_score), (home_xg, away_xg), day) in \
            enumerate(zip(FIXTURE_SCORES, FIXTURE_XG, FIXTURE_DAYS)):
        rows.append({
            "match_id": 100 + position,
            "match_date": pd.Timestamp("2015-08-01") + pd.Timedelta(days=day),
            "home_team_id": 1, "away_team_id": 2,
            "home_score": home_score, "away_score": away_score,
            "home_xg": home_xg, "away_xg": away_xg})
    frame = pd.DataFrame(rows)
    matches = frame.drop(columns=["home_xg", "away_xg"])
    events_index = frame[["match_id", "home_xg", "away_xg"]]
    return matches, events_index


def _event(match_id, team_id, type_name, x=None, end_x=None,
           shot_outcome=None, pass_outcome=None, pass_type=None):
    return {"match_id": match_id, "team_id": team_id, "type": type_name,
            "x": x, "end_x": end_x, "shot_outcome": shot_outcome,
            "pass_outcome": pass_outcome, "pass_type": pass_type}


def make_events_fixture():
    """Hand-countable events for match 100. Team 1:
    3 passes (2 complete; zones def/mid/fin), 1 corner among them, 2 shots
    (1 on target), 1 pressure, 1 carry into the final third, 1 clearance.
    Team 2: 1 complete pass, 1 interception."""
    rows = [
        _event(100, 1, "Pass", x=20.0),                       # def, complete
        _event(100, 1, "Pass", x=60.0, pass_outcome="Incomplete"),  # mid
        _event(100, 1, "Pass", x=90.0, pass_type="Corner"),   # fin, complete
        _event(100, 1, "Shot", shot_outcome="Saved"),
        _event(100, 1, "Shot", shot_outcome="Off T"),
        _event(100, 1, "Pressure"),
        _event(100, 1, "Carry", x=60.0, end_x=85.0),
        _event(100, 1, "Clearance"),
        _event(100, 2, "Pass", x=30.0),
        _event(100, 2, "Interception"),
    ]
    return pd.DataFrame(rows)


def home_rows(team_match):
    return (team_match[(team_match["team_id"] == 1)
                       & (team_match["venue"] == "home")]
            .sort_values("match_date").reset_index(drop=True))


# --- Tests ------------------------------------------------------------------
def test_long_table_has_two_rows_per_match_with_mirrored_scores():
    matches, events_index = make_fixture()
    team_match = build_team_match_long_table(matches, events_index)
    assert len(team_match) == 2 * len(matches), \
        "the long table must carry one row per team per match"
    for match_id in matches["match_id"]:
        pair = team_match[team_match["match_id"] == match_id]
        assert set(pair["venue"]) == {"home", "away"}, "venue not mirrored"
        home = pair[pair["venue"] == "home"].iloc[0]
        away = pair[pair["venue"] == "away"].iloc[0]
        assert home["gf"] == away["ga"] and home["ga"] == away["gf"], \
            "goals for and against are not mirrored across the two rows"
        assert home["xgf"] == away["xga"] and home["xga"] == away["xgf"], \
            "xG is not mirrored across the two rows"
    print("ok  long table mirrors goals and xG across the two team rows")


def test_points_and_win_follow_the_scoreline():
    matches, events_index = make_fixture()
    team_match = build_team_match_long_table(matches, events_index)
    rows = home_rows(team_match)
    # win, draw, loss, draw, win, draw, win.
    assert rows["points"].tolist() == [3, 1, 0, 1, 3, 1, 3], \
        "points do not follow the scoreline"
    assert rows["win"].tolist() == [1, 0, 0, 0, 1, 0, 1], \
        "win flag does not follow the scoreline"
    print("ok  points and win flag follow the scoreline")


def test_first_match_carries_no_form():
    matches, events_index = make_fixture()
    team_match = add_rolling_form_features(
        build_team_match_long_table(matches, events_index))
    first = team_match.sort_values(["team_id", "match_date", "match_id"]) \
        .groupby("team_id").head(1)
    assert first[["form_gf", "form_ga", "form_xgf", "form_xga",
                  "form_points", "form_win"]].isna().all().all(), \
        "a team's first match must carry no rolling form"
    assert first["rest_days"].isna().all(), \
        "a team's first match has no previous match to rest from"
    assert (first["played_prior"] == 0).all(), \
        "a team's first match must report zero prior appearances"
    print("ok  the first match of every team carries no prior information")


def test_rolling_form_is_the_hand_computed_prior_mean():
    matches, events_index = make_fixture()
    team_match = add_rolling_form_features(
        build_team_match_long_table(matches, events_index))
    rows = home_rows(team_match)
    goals_for = [s[0] for s in FIXTURE_SCORES]

    # Prior-only mean over at most ROLLING_WINDOW previous matches.
    expected = []
    for position in range(len(goals_for)):
        window = goals_for[max(0, position - ROLLING_WINDOW):position]
        expected.append(float(np.mean(window)) if window else np.nan)

    observed = rows["form_gf"].tolist()
    assert np.isnan(observed[0]) and np.isnan(expected[0]), \
        "the first entry must be missing"
    assert np.allclose(observed[1:], expected[1:]), \
        f"rolling form mismatch: {observed} != {expected}"
    # Spelled out for the fourth match: mean of 3, 1, 0.
    assert abs(observed[3] - (3 + 1 + 0) / 3) < 1e-12, \
        "the fourth match must average exactly the first three"
    print(f"ok  rolling form equals the hand-computed prior mean "
          f"(window {ROLLING_WINDOW})")


def test_window_never_exceeds_its_length():
    matches, events_index = make_fixture()
    team_match = add_rolling_form_features(
        build_team_match_long_table(matches, events_index))
    rows = home_rows(team_match)
    goals_for = [s[0] for s in FIXTURE_SCORES]
    # The seventh match must ignore the first: the window holds five matches.
    expected = float(np.mean(goals_for[1:6]))
    assert abs(rows["form_gf"].iloc[6] - expected) < 1e-12, \
        "the rolling window did not drop the oldest match"
    print("ok  the window drops the oldest match once it is full")


def test_rest_days_are_calendar_gaps_between_consecutive_matches():
    matches, events_index = make_fixture()
    team_match = add_rolling_form_features(
        build_team_match_long_table(matches, events_index))
    rows = home_rows(team_match)
    expected = [np.nan] + [FIXTURE_DAYS[i] - FIXTURE_DAYS[i - 1]
                           for i in range(1, len(FIXTURE_DAYS))]
    observed = rows["rest_days"].tolist()
    assert np.isnan(observed[0]), "the first match has no rest gap"
    assert observed[1:] == expected[1:], \
        f"rest days mismatch: {observed} != {expected}"
    print("ok  rest days equal the calendar gap to the previous match")


def test_played_prior_counts_only_earlier_matches():
    matches, events_index = make_fixture()
    team_match = add_rolling_form_features(
        build_team_match_long_table(matches, events_index))
    rows = home_rows(team_match)
    assert rows["played_prior"].tolist() == list(range(len(FIXTURE_SCORES))), \
        "played_prior must count strictly earlier matches"
    print("ok  played_prior counts strictly earlier matches")


def test_pivot_produces_one_row_per_match_with_consistent_differences():
    matches, events_index = make_fixture()
    team_match = add_rolling_form_features(
        build_team_match_long_table(matches, events_index))
    features = build_prematch_features(team_match)
    assert features["match_id"].is_unique, "the pivot must be one row per match"
    assert len(features) == len(matches), "the pivot lost or duplicated matches"
    for column in ["form_gf", "form_xgf", "form_points", "rest_days"]:
        left = features[f"home_{column}"] - features[f"away_{column}"]
        right = features[f"diff_{column}"]
        both_present = left.notna() & right.notna()
        assert np.allclose(left[both_present], right[both_present]), \
            f"diff_{column} is not home minus away"
    print("ok  the pivot is one row per match and every diff is home minus away")


def test_event_aggregates_match_the_hand_count():
    aggregates = build_event_aggregates(make_events_fixture())
    team1 = aggregates[aggregates["team_id"] == 1].iloc[0]
    assert team1["shots"] == 2 and team1["shots_on_target"] == 1, \
        "shot counts do not match the hand count"
    assert team1["pressures"] == 1, "pressure count wrong"
    assert team1["passes"] == 3, "pass count wrong"
    assert abs(team1["pass_completion"] - 2 / 3) < 1e-12, \
        "completion must be completed / attempted (no-outcome = complete)"
    assert (team1["passes_def"], team1["passes_mid"], team1["passes_fin"]) \
        == (1, 1, 1), "passes not split by pitch third on x"
    assert team1["carries_final_third"] == 1, \
        "carry ending at end_x >= 80 must count as into the final third"
    assert team1["corners"] == 1 and team1["free_kicks"] == 0 \
        and team1["throw_ins"] == 0, "set-piece counts wrong"
    assert team1["defensive_actions"] == 1, "defensive action count wrong"
    # Team 1 played 3 of the 4 passes in the match.
    assert abs(team1["possession_share"] - 3 / 4) < 1e-12, \
        "possession share must be the team's share of the match's passes"
    team2 = aggregates[aggregates["team_id"] == 2].iloc[0]
    assert team2["defensive_actions"] == 1 and team2["shots"] == 0, \
        "team 2 aggregates wrong"
    print("ok  event aggregates equal the hand count for the fixture match")


def test_event_form_is_rolled_and_prior_only():
    matches, events_index = make_fixture()
    team_match = build_team_match_long_table(matches, events_index)
    team_match = add_event_aggregates(
        team_match, build_event_aggregates(make_events_fixture()))
    team_match = add_rolling_form_features(team_match)
    rows = home_rows(team_match)
    # Only match 100 has events, so its own row carries no form (prior-only)
    # and the second match's form equals match 100's raw aggregate.
    assert np.isnan(rows.loc[0, "form_shots"]), \
        "a match's own events leaked into its own form"
    assert rows.loc[1, "form_shots"] == 2, \
        "the second match must carry the first match's shot count as form"
    print("ok  event aggregates become prior-only rolling form")


def test_h2h_is_the_expanding_mean_of_prior_meetings():
    matches, events_index = make_fixture()
    team_match = add_head_to_head(add_rolling_form_features(
        build_team_match_long_table(matches, events_index)))
    rows = home_rows(team_match)
    margins = [h - a for h, a in FIXTURE_SCORES]
    expected = [np.nan] + [float(np.mean(margins[:i]))
                           for i in range(1, len(margins))]
    observed = rows["h2h_margin"].tolist()
    assert np.isnan(observed[0]), "a pair's first meeting has no head-to-head"
    assert np.allclose(observed[1:], expected[1:]), \
        f"h2h margin mismatch: {observed} != {expected}"
    wins = [1 if h > a else 0 for h, a in FIXTURE_SCORES]
    expected_win = [float(np.mean(wins[:i])) for i in range(1, len(wins))]
    assert np.allclose(rows["h2h_win"].tolist()[1:], expected_win), \
        "h2h win rate is not the expanding mean of prior meetings"
    print("ok  head-to-head is the expanding mean over strictly prior meetings")


def _full_chain(matches, events_index, events):
    team_match = build_team_match_long_table(matches, events_index)
    team_match = add_event_aggregates(team_match,
                                      build_event_aggregates(events))
    team_match = add_rolling_form_features(team_match)
    team_match = add_head_to_head(team_match)
    return build_prematch_features(team_match)


def test_no_feature_column_uses_the_current_match():
    """The graded barrier: perturbing a match's own result or its own events
    must not move any of its features."""
    matches, events_index = make_fixture()
    events = make_events_fixture()
    baseline = _full_chain(matches, events_index, events)

    tampered = matches.copy()
    last = tampered.index[-1]
    tampered.loc[last, "home_score"] = 9
    tampered.loc[last, "away_score"] = 0
    last_match_id = int(matches.loc[last, "match_id"])
    tampered_events = pd.concat([
        events,
        pd.DataFrame([_event(last_match_id, 1, "Shot", x=50.0, end_x=50.0,
                             shot_outcome="Goal") for _ in range(5)]),
    ], ignore_index=True)
    tampered_features = _full_chain(tampered, events_index, tampered_events)

    before = baseline[baseline["match_id"] == last_match_id] \
        .drop(columns="match_id")
    after = tampered_features[tampered_features["match_id"] == last_match_id] \
        .drop(columns="match_id")
    pd.testing.assert_frame_equal(before, after, check_exact=False)
    print("ok  rewriting a match's own result or events leaves its own "
          "features unchanged")


def main():
    tests = [test_long_table_has_two_rows_per_match_with_mirrored_scores,
             test_points_and_win_follow_the_scoreline,
             test_first_match_carries_no_form,
             test_rolling_form_is_the_hand_computed_prior_mean,
             test_window_never_exceeds_its_length,
             test_rest_days_are_calendar_gaps_between_consecutive_matches,
             test_played_prior_counts_only_earlier_matches,
             test_pivot_produces_one_row_per_match_with_consistent_differences,
             test_event_aggregates_match_the_hand_count,
             test_event_form_is_rolled_and_prior_only,
             test_h2h_is_the_expanding_mean_of_prior_meetings,
             test_no_feature_column_uses_the_current_match]
    for test in tests:
        test()
    print(f"\n{len(tests)}/{len(tests)} passed")


if __name__ == "__main__":
    main()
