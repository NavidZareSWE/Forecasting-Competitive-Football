"""Run from the repository root with:

    python src/features/test_event_aggregates.py
"""

from pathlib import Path
import sys

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))

from event_aggregates import (ATTACKING_THIRD_X, QUANTITIES, dismissals,
                              match_frame_aggregates, possession_owners,
                              team_aggregates)

HOME, AWAY = 1, 2


def event(index, team, kind, x=60.0, possession=1, pattern="Regular Play",
          under_pressure=False, outcome=None, xg=None, goal=False, card=None):
    return {"index": index, "team_id": team, "type": kind, "x": x, "y": 40.0,
            "possession": possession, "play_pattern": pattern,
            "under_pressure": under_pressure, "shot_outcome": outcome,
            "shot_xg": xg, "is_goal": goal, "card": card, "period": 1,
            "minute": index}


def make_fixture():
    rows = [
        # --- chain 1, home, from a corner ---
        event(1, HOME, "Pass", x=85.0, pattern="From Corner"),
        event(2, HOME, "Ball Receipt*", x=88.0, pattern="From Corner"),
        event(3, HOME, "Carry", x=88.0, pattern="From Corner"),
        event(4, HOME, "Pass", x=90.0, pattern="From Corner",
              under_pressure=True),
        event(5, HOME, "Shot", x=105.0, pattern="From Corner",
              outcome="Goal", xg=0.55, goal=True),
        event(6, AWAY, "Pressure", x=30.0, pattern="From Corner"),
        # --- chain 2, away in regular play ---
        event(7, AWAY, "Pass", x=50.0, possession=2),
        event(8, AWAY, "Ball Receipt*", x=55.0, possession=2),
        event(9, AWAY, "Carry", x=55.0, possession=2),
        event(10, AWAY, "Pass", x=82.0, possession=2),
        event(11, HOME, "Pressure", x=35.0, possession=2),
        event(12, HOME, "Interception", x=30.0, possession=2),
        # --- loose home events in chain 2 ---
        event(13, HOME, "Pass", x=20.0, possession=2),
        event(14, HOME, "Carry", x=25.0, possession=2),
        event(15, HOME, "Miscontrol", x=25.0, possession=2),
        event(16, AWAY, "Foul Committed", x=60.0, possession=2,
              card="Yellow Card"),
    ]
    return pd.DataFrame(rows)


def home_totals():
    events = make_fixture()
    return team_aggregates(events, HOME, AWAY)


# --- Tests ------------------------------------------------------------------
def test_every_declared_quantity_is_produced():
    totals = home_totals()
    missing = [q for q in QUANTITIES if q not in totals]
    assert not missing, f"QUANTITIES declares {missing} but they are not built"
    extra = [k for k in totals if k not in QUANTITIES]
    assert not extra, f"aggregates produced undeclared keys {extra}"
    print(f"ok  all {len(QUANTITIES)} declared quantities are produced")


def test_shot_quantities_match_the_hand_tally():
    totals = home_totals()
    assert totals["shots"] == 1, totals["shots"]
    assert totals["shots_on_target"] == 1, totals["shots_on_target"]
    assert totals["shots_in_box"] == 1, totals["shots_in_box"]
    assert abs(totals["xg"] - 0.55) < 1e-9, totals["xg"]
    assert totals["goals"] == 1, totals["goals"]
    print("ok  shot counts, xG and goals match the hand tally")


def test_zone_counts_use_the_acting_team_convention():
    totals = home_totals()
    assert totals["passes"] == 3, totals["passes"]
    assert totals["passes_attacking_third"] == 2, \
        totals["passes_attacking_third"]
    assert totals["passes_defensive_third"] == 1, \
        totals["passes_defensive_third"]
    assert abs(totals["pass_share_attacking_third"] - 2 / 3) < 1e-9
    assert totals["carries"] == 2, totals["carries"]
    assert totals["carries_attacking_third"] == 1, \
        totals["carries_attacking_third"]
    print(f"ok  zone counts split on x >= {ATTACKING_THIRD_X:.0f} as tallied")


def test_pressure_quantities():
    totals = home_totals()
    assert totals["pressures"] == 1, totals["pressures"]
    assert totals["pressure_rate_attacking_third"] == 0.0
    assert totals["events_under_pressure"] == 1, \
        totals["events_under_pressure"]
    assert totals["events"] == 10, totals["events"]
    assert abs(totals["under_pressure_share"] - 1 / 10) < 1e-9, \
        totals["under_pressure_share"]
    print("ok  pressure applied and pressure received are counted separately")


def test_defensive_actions_and_turnovers():
    totals = home_totals()
    assert totals["defensive_actions"] == 1, totals["defensive_actions"]
    assert totals["defensive_actions_own_third"] == 1, \
        totals["defensive_actions_own_third"]
    assert totals["turnovers"] == 1, totals["turnovers"]
    print("ok  defensive actions and turnovers match the hand tally")


def test_possession_shares_sum_to_one_and_follow_on_ball_work():
    events = make_fixture()
    owners = possession_owners(events)
    assert owners.loc[1] == HOME, "chain 1 has more home on-ball events"
    assert owners.loc[2] == AWAY, "chain 2 has more away on-ball events"
    home, away = match_frame_aggregates(events, HOME, AWAY)
    assert home["possession_chains"] == 1 and away["possession_chains"] == 1
    assert abs(home["possession_share"] + away["possession_share"] - 1.0) < 1e-9
    print("ok  chains go to the team doing the on-ball work; shares sum to one")


def test_set_pieces_attach_to_the_chain_owner():
    home, away = match_frame_aggregates(make_fixture(), HOME, AWAY)
    assert home["set_piece_corner"] == 1, home["set_piece_corner"]
    assert away["set_piece_corner"] == 0, away["set_piece_corner"]
    print("ok  a corner chain is credited to the team that owned it")


def test_own_goal_is_credited_to_the_benefiting_team():
    events = pd.DataFrame(make_fixture().to_dict("records")
                          + [event(17, AWAY, "Own Goal Against", x=10.0),
                             event(18, HOME, "Own Goal For", x=110.0)])
    home, away = match_frame_aggregates(events, HOME, AWAY)
    assert home["goals"] == 2, f"home should hold both goals, got {home['goals']}"
    assert away["goals"] == 0, f"away conceded an own goal, got {away['goals']}"
    print("ok  Own Goal For credits the benefiting team, not the conceding one")


def test_prefix_aggregates_are_monotone_in_the_prefix_length():
    events = make_fixture()
    counting = ["shots", "passes", "carries", "pressures", "events",
                "defensive_actions", "turnovers", "goals"]
    previous = None
    for length in range(1, len(events) + 1):
        totals = team_aggregates(events.iloc[:length], HOME, AWAY)
        if previous is not None:
            for quantity in counting:
                assert totals[quantity] >= previous[quantity], \
                    (f"{quantity} fell from {previous[quantity]} to "
                     f"{totals[quantity]} when the prefix grew")
        previous = totals
    print("ok  counting quantities never fall as the prefix grows")


def test_empty_frame_is_all_zero_rather_than_an_error():
    empty = make_fixture().iloc[:0]
    totals = team_aggregates(empty, HOME, AWAY)
    assert all(float(totals[q]) == 0.0 for q in QUANTITIES), \
        "an empty prefix must produce zeros, not missing values"
    print("ok  the minute-0 snapshot produces zeros rather than failing")


def test_missing_optional_columns_do_not_raise():
    reduced = make_fixture().drop(columns=["possession", "play_pattern",
                                           "under_pressure"])
    totals = team_aggregates(reduced, HOME, AWAY)
    assert totals["possession_chains"] == 0
    assert totals["set_piece_corner"] == 0
    assert totals["passes"] == 3, "present columns must still be counted"
    print("ok  absent optional columns fall back to zero, not an exception")


def test_dismissals_count_only_the_named_cards():
    events = pd.DataFrame(make_fixture().to_dict("records")
                          + [event(17, HOME, "Foul Committed", card="Red Card"),
                             event(18, HOME, "Bad Behaviour",
                                   card="Second Yellow"),
                             event(19, AWAY, "Foul Committed",
                                   card="Yellow Card")])
    assert dismissals(events, HOME) == 2, dismissals(events, HOME)
    assert dismissals(events, AWAY) == 0, dismissals(events, AWAY)
    print("ok  only red and second-yellow cards count as dismissals")


def main():
    tests = [test_every_declared_quantity_is_produced,
             test_shot_quantities_match_the_hand_tally,
             test_zone_counts_use_the_acting_team_convention,
             test_pressure_quantities,
             test_defensive_actions_and_turnovers,
             test_possession_shares_sum_to_one_and_follow_on_ball_work,
             test_set_pieces_attach_to_the_chain_owner,
             test_own_goal_is_credited_to_the_benefiting_team,
             test_prefix_aggregates_are_monotone_in_the_prefix_length,
             test_empty_frame_is_all_zero_rather_than_an_error,
             test_missing_optional_columns_do_not_raise,
             test_dismissals_count_only_the_named_cards]
    for test in tests:
        test()
    print(f"\n{len(tests)}/{len(tests)} passed")


if __name__ == "__main__":
    main()
