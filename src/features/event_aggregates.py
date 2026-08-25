# StatsBomb coordinates are stated from the acting team's point of view,
# attacking left to right on a 120 by 80 pitch, so x >= 80 is that team's
# attacking third with no venue flip.

import numpy as np
import pandas as pd

ATTACKING_THIRD_X = 80.0
DEFENSIVE_THIRD_X = 40.0
PENALTY_BOX_X = 102.0

ON_BALL_TYPES = {"Pass", "Carry", "Ball Receipt*", "Dribble", "Shot"}
DEFENSIVE_ACTION_TYPES = {"Ball Recovery", "Interception", "Clearance",
                          "Block", "Duel"}
ON_TARGET_OUTCOMES = {"Goal", "Saved", "Saved To Post"}
DISMISSAL_CARDS = {"Red Card", "Second Yellow"}
SET_PIECE_PATTERNS = {"From Corner": "corner",
                      "From Free Kick": "free_kick",
                      "From Throw In": "throw_in"}

QUANTITIES = [
    "goals", "shots", "shots_on_target", "xg", "shots_in_box",
    "passes", "passes_attacking_third", "passes_defensive_third",
    "pass_share_attacking_third", "carries", "carries_attacking_third",
    "dribbles", "mean_x", "touches_attacking_third",
    "pressures", "pressure_rate_attacking_third", "events_under_pressure",
    "under_pressure_share", "defensive_actions", "defensive_actions_own_third",
    "fouls_committed", "fouls_won", "turnovers",
    "possession_chains", "possession_share",
    "set_piece_corner", "set_piece_free_kick", "set_piece_throw_in",
    "events",
]


OPTIONAL_QUANTITIES = ["passes_completed", "pass_completion",
                       "carries_into_attacking_third"]


def optional_quantities(events, team_id):
    if "pass_outcome" not in events.columns and \
            "carry_end_x" not in events.columns:
        return {}
    team_events = events[events["team_id"] == team_id]
    types = _column(team_events, "type")
    result = {}
    if "pass_outcome" in events.columns:
        passes = team_events[types == "Pass"]
        completed = int(_column(passes, "pass_outcome").isna().sum())
        result["passes_completed"] = completed
        result["pass_completion"] = _safe_ratio(completed, len(passes))
    if "carry_end_x" in events.columns:
        carries = team_events[types == "Carry"]
        start = pd.to_numeric(_column(carries, "x"), errors="coerce")
        end = pd.to_numeric(_column(carries, "carry_end_x"), errors="coerce")
        result["carries_into_attacking_third"] = int(
            ((start < ATTACKING_THIRD_X) & (end >= ATTACKING_THIRD_X)).sum())
    return result


def _safe_ratio(numerator, denominator):
    return float(numerator) / float(denominator) if denominator else 0.0


def _boolean(series):
    if series.dtype == bool:
        return series
    values = np.zeros(len(series), dtype=bool)
    present = series.notna().to_numpy()
    if present.any():
        values[present] = series.to_numpy()[present].astype(bool)
    return pd.Series(values, index=series.index)


def _column(frame, name):
    if name in frame.columns:
        return frame[name]
    return pd.Series(np.nan, index=frame.index)


def possession_owners(events):
    if "possession" not in events.columns:
        return pd.Series(dtype="float64")
    on_ball = events[_column(events, "type").isin(ON_BALL_TYPES)]
    on_ball = on_ball[on_ball["possession"].notna()
                      & on_ball["team_id"].notna()]
    if on_ball.empty:
        return pd.Series(dtype="float64")
    counts = (on_ball.groupby(["possession", "team_id"]).size()
              .rename("n").reset_index()
              .sort_values(["possession", "n"], ascending=[True, False]))
    return counts.drop_duplicates("possession").set_index("possession")["team_id"]


def team_aggregates(events, team_id, opponent_id, owners=None):
    if owners is None:
        owners = possession_owners(events)

    types = _column(events, "type")
    team_mask = events["team_id"] == team_id
    team_events = events[team_mask]
    team_types = _column(team_events, "type")
    x = pd.to_numeric(_column(team_events, "x"), errors="coerce")

    shots = team_events[team_types == "Shot"]
    shot_x = pd.to_numeric(_column(shots, "x"), errors="coerce")
    outcomes = _column(shots, "shot_outcome")
    goals = int(_boolean(_column(team_events, "is_goal")).sum())
    goals += int((team_types == "Own Goal For").sum())

    passes = team_events[team_types == "Pass"]
    pass_x = pd.to_numeric(_column(passes, "x"), errors="coerce")
    passes_attacking = int((pass_x >= ATTACKING_THIRD_X).sum())
    passes_defensive = int((pass_x <= DEFENSIVE_THIRD_X).sum())

    carries = team_events[team_types == "Carry"]
    carry_x = pd.to_numeric(_column(carries, "x"), errors="coerce")

    pressures = team_events[team_types == "Pressure"]
    pressure_x = pd.to_numeric(_column(pressures, "x"), errors="coerce")

    under_pressure = _boolean(_column(team_events, "under_pressure"))

    defensive = team_events[team_types.isin(DEFENSIVE_ACTION_TYPES)]
    defensive_x = pd.to_numeric(_column(defensive, "x"), errors="coerce")

    turnovers = int((team_types.isin({"Dispossessed", "Miscontrol"})).sum())

    if len(owners):
        owned = int((owners == team_id).sum())
        total_chains = int(owners.notna().sum())
    else:
        owned, total_chains = 0, 0

    patterns = _column(events, "play_pattern")
    set_pieces = {}
    for pattern, label in SET_PIECE_PATTERNS.items():
        if len(owners) and "possession" in events.columns:
            chains = events.loc[patterns == pattern, "possession"].dropna()
            matched = owners.reindex(chains.unique()).dropna()
            set_pieces[f"set_piece_{label}"] = int((matched == team_id).sum())
        else:
            set_pieces[f"set_piece_{label}"] = 0

    touches_attacking = int((x >= ATTACKING_THIRD_X).sum())

    return {
        "goals": goals,
        "shots": int(len(shots)),
        "shots_on_target": int(outcomes.isin(ON_TARGET_OUTCOMES).sum()),
        "xg": float(pd.to_numeric(_column(shots, "shot_xg"),
                                  errors="coerce").fillna(0.0).sum()),
        "shots_in_box": int((shot_x >= PENALTY_BOX_X).sum()),
        "passes": int(len(passes)),
        "passes_attacking_third": passes_attacking,
        "passes_defensive_third": passes_defensive,
        "pass_share_attacking_third": _safe_ratio(passes_attacking,
                                                  len(passes)),
        "carries": int(len(carries)),
        "carries_attacking_third": int((carry_x >= ATTACKING_THIRD_X).sum()),
        "dribbles": int((team_types == "Dribble").sum()),
        "mean_x": float(x.mean()) if x.notna().any() else 0.0,
        "touches_attacking_third": touches_attacking,
        "pressures": int(len(pressures)),
        "pressure_rate_attacking_third": _safe_ratio(
            int((pressure_x >= ATTACKING_THIRD_X).sum()), len(pressures)),
        "events_under_pressure": int(under_pressure.sum()),
        "under_pressure_share": _safe_ratio(int(under_pressure.sum()),
                                            len(team_events)),
        "defensive_actions": int(len(defensive)),
        "defensive_actions_own_third": int(
            (defensive_x <= DEFENSIVE_THIRD_X).sum()),
        "fouls_committed": int((team_types == "Foul Committed").sum()),
        "fouls_won": int((team_types == "Foul Won").sum()),
        "turnovers": turnovers,
        "possession_chains": owned,
        "possession_share": _safe_ratio(owned, total_chains),
        **set_pieces,
        "events": int(len(team_events)),
        **optional_quantities(events, team_id),
    }


def match_frame_aggregates(events, home_team_id, away_team_id):
    owners = possession_owners(events)
    home = team_aggregates(events, home_team_id, away_team_id, owners)
    away = team_aggregates(events, away_team_id, home_team_id, owners)
    return home, away


def dismissals(events, team_id):
    cards = _column(events[events["team_id"] == team_id], "card")
    return int(cards.isin(DISMISSAL_CARDS).sum())
