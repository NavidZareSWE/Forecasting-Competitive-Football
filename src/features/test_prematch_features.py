"""Tests for the pre-match feature builder (Phase 5A).

Every expected value below is computed by hand from the fixture, not from the
code under test.

    python src/features/test_prematch_features.py
"""

import numpy as np
import pandas as pd

from build_prematch_features import (ROLLING_WINDOW, add_rolling_form_features,
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


def test_no_feature_column_uses_the_current_match():
    """The graded barrier: perturbing a match's own result must not move it."""
    matches, events_index = make_fixture()
    baseline = build_prematch_features(add_rolling_form_features(
        build_team_match_long_table(matches, events_index)))

    tampered = matches.copy()
    last = tampered.index[-1]
    tampered.loc[last, "home_score"] = 9
    tampered.loc[last, "away_score"] = 0
    tampered_features = build_prematch_features(add_rolling_form_features(
        build_team_match_long_table(tampered, events_index)))

    match_id = int(matches.loc[last, "match_id"])
    before = baseline[baseline["match_id"] == match_id].drop(columns="match_id")
    after = tampered_features[tampered_features["match_id"] == match_id] \
        .drop(columns="match_id")
    pd.testing.assert_frame_equal(before, after, check_exact=False)
    print("ok  rewriting a match's own result leaves its own features unchanged")


def main():
    tests = [test_long_table_has_two_rows_per_match_with_mirrored_scores,
             test_points_and_win_follow_the_scoreline,
             test_first_match_carries_no_form,
             test_rolling_form_is_the_hand_computed_prior_mean,
             test_window_never_exceeds_its_length,
             test_rest_days_are_calendar_gaps_between_consecutive_matches,
             test_played_prior_counts_only_earlier_matches,
             test_pivot_produces_one_row_per_match_with_consistent_differences,
             test_no_feature_column_uses_the_current_match]
    for test in tests:
        test()
    print(f"\n{len(tests)}/{len(tests)} passed")


if __name__ == "__main__":
    main()
