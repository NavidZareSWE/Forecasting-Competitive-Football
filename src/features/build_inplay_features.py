from pathlib import Path

import numpy as np
import pandas as pd


HERE = Path(__file__).resolve().parent
PROJECT = HERE.parent
PROCESSED_DIR = PROJECT / "reports" / "processed"
FEATURE_DIR = PROJECT / "reports" / "features"

SNAPSHOT_MINUTES = list(range(0, 91, 5))
RECENT_WINDOW_MINUTES = 10
DISMISSAL_CARDS = {"Red Card", "Second Yellow"}
ON_TARGET_OUTCOMES = {"Goal", "Saved", "Saved To Post"}


def effective_minute(events_sorted):
    # Leakage invariant: accumulate only raises, so eff[i] <= t implies minute[i] <= t.
    # Side effect: period 2 restarts at 45', so its opening events wait until t=50.
    return np.maximum.accumulate(events_sorted["minute"].to_numpy())


def prefix_length_at(effective_minutes, snapshot_minute):
    return int(np.searchsorted(effective_minutes, snapshot_minute, side="right"))


def assert_prefix(effective_minutes, prefix_len, snapshot_minute):
    if prefix_len > 0:
        assert effective_minutes[prefix_len - 1] <= snapshot_minute, \
            "cut included an event after time t"
    if prefix_len < len(effective_minutes):
        assert effective_minutes[prefix_len] > snapshot_minute, \
            "cut excluded an event at or before time t; not a clean prefix"


def goals_for(prefix, team_id):
    scored = int(((prefix["team_id"] == team_id) & prefix["is_goal"]).sum())
    own = int(((prefix["team_id"] == team_id) &
               (prefix["type"] == "Own Goal For")).sum())
    return scored + own


def _series(frame, column):
    # Optional columns may be absent from small test fixtures; treat a missing
    # column as all-missing so every count it feeds is zero.
    if column in frame.columns:
        return frame[column]
    return pd.Series([None] * len(frame), index=frame.index, dtype=object)


def _team_counts(frame, mask, home_team_id, away_team_id):
    home = int((mask & (frame["team_id"] == home_team_id)).sum())
    away = int((mask & (frame["team_id"] == away_team_id)).sum())
    return home, away


def snapshot_state(prefix, home_team_id, away_team_id, snapshot_minute,
                   prefix_effective_minutes):
    home_goals = goals_for(prefix, home_team_id)
    away_goals = goals_for(prefix, away_team_id)

    shots = prefix[prefix["type"] == "Shot"]
    home_shots = int((shots["team_id"] == home_team_id).sum())
    away_shots = int((shots["team_id"] == away_team_id).sum())
    on_target = shots[shots["shot_outcome"].isin(ON_TARGET_OUTCOMES)]
    home_sot = int((on_target["team_id"] == home_team_id).sum())
    away_sot = int((on_target["team_id"] == away_team_id).sum())

    home_xg = float(shots.loc[shots["team_id"] == home_team_id, "shot_xg"].sum())
    away_xg = float(shots.loc[shots["team_id"] == away_team_id, "shot_xg"].sum())

    dismissals = prefix[prefix["card"].isin(DISMISSAL_CARDS)]
    home_red = int((dismissals["team_id"] == home_team_id).sum())
    away_red = int((dismissals["team_id"] == away_team_id).sum())

    # Full-prefix counts beyond shots (brief 5B: event counts, not only xG).
    pressure_mask = prefix["type"] == "Pressure"
    corner_mask = _series(prefix, "pass_type") == "Corner"
    foul_mask = prefix["type"] == "Foul Committed"
    card_mask = _series(prefix, "card").notna()
    home_pressures, away_pressures = _team_counts(
        prefix, pressure_mask, home_team_id, away_team_id)
    home_corners, away_corners = _team_counts(
        prefix, corner_mask, home_team_id, away_team_id)
    home_fouls, away_fouls = _team_counts(
        prefix, foul_mask, home_team_id, away_team_id)
    home_cards, away_cards = _team_counts(
        prefix, card_mask, home_team_id, away_team_id)

    # Windowed on effective minute; the raw column is wrong for exactly the rows
    # the repair exists for.
    recent = prefix[prefix_effective_minutes >=
                    snapshot_minute - RECENT_WINDOW_MINUTES]
    recent_shots = recent[recent["type"] == "Shot"]
    home_recent_xg = float(
        recent_shots.loc[recent_shots["team_id"] == home_team_id, "shot_xg"].sum())
    away_recent_xg = float(
        recent_shots.loc[recent_shots["team_id"] == away_team_id, "shot_xg"].sum())

    # Recent-window counts and per-minute rates (brief 5B). The divisor is the
    # portion of the window actually played, so an early snapshot is not
    # deflated by minutes that never happened.
    window_minutes = max(1, min(RECENT_WINDOW_MINUTES, snapshot_minute))
    home_recent_shots, away_recent_shots = _team_counts(
        recent, recent["type"] == "Shot", home_team_id, away_team_id)
    home_recent_pressures, away_recent_pressures = _team_counts(
        recent, recent["type"] == "Pressure", home_team_id, away_team_id)
    home_recent_corners, away_recent_corners = _team_counts(
        recent, _series(recent, "pass_type") == "Corner",
        home_team_id, away_team_id)
    home_recent_events, away_recent_events = _team_counts(
        recent, pd.Series(True, index=recent.index),
        home_team_id, away_team_id)

    # Momentum: this window's xG against the immediately preceding window's.
    previous = prefix[
        (prefix_effective_minutes >=
         snapshot_minute - 2 * RECENT_WINDOW_MINUTES)
        & (prefix_effective_minutes < snapshot_minute - RECENT_WINDOW_MINUTES)]
    previous_shots = previous[previous["type"] == "Shot"]
    home_prev_xg = float(previous_shots.loc[
        previous_shots["team_id"] == home_team_id, "shot_xg"].sum())
    away_prev_xg = float(previous_shots.loc[
        previous_shots["team_id"] == away_team_id, "shot_xg"].sum())
    recent_total = home_recent_events + away_recent_events
    recent_event_share = (home_recent_events / recent_total
                          if recent_total else 0.5)

    return {
        "snapshot_minute": snapshot_minute,
        "inplay_goal_diff": home_goals - away_goals,
        "inplay_home_goals": home_goals,
        "inplay_away_goals": away_goals,
        "inplay_man_advantage": away_red - home_red,
        "inplay_shot_diff": home_shots - away_shots,
        "inplay_sot_diff": home_sot - away_sot,
        "inplay_xg_diff": round(home_xg - away_xg, 4),
        "inplay_home_xg": round(home_xg, 4),
        "inplay_away_xg": round(away_xg, 4),
        "inplay_pressure_diff": home_pressures - away_pressures,
        "inplay_corner_diff": home_corners - away_corners,
        "inplay_foul_diff": home_fouls - away_fouls,
        "inplay_card_diff": home_cards - away_cards,
        "inplay_recent_xg_diff": round(home_recent_xg - away_recent_xg, 4),
        "inplay_recent_shot_diff": home_recent_shots - away_recent_shots,
        "inplay_recent_pressure_diff":
            home_recent_pressures - away_recent_pressures,
        "inplay_recent_corner_diff": home_recent_corners - away_recent_corners,
        "inplay_recent_shot_rate_home":
            round(home_recent_shots / window_minutes, 4),
        "inplay_recent_shot_rate_away":
            round(away_recent_shots / window_minutes, 4),
        "inplay_recent_event_rate_home":
            round(home_recent_events / window_minutes, 4),
        "inplay_recent_event_rate_away":
            round(away_recent_events / window_minutes, 4),
        "inplay_momentum_xg_diff": round(
            (home_recent_xg - away_recent_xg)
            - (home_prev_xg - away_prev_xg), 4),
        "inplay_recent_event_share_home": round(recent_event_share, 4),
        "inplay_events_so_far": int(len(prefix)),
    }


def build_match_snapshots(events, home_team_id, away_team_id):
    # No-op on clean data (index is already globally chronological); keeps
    # periods separated if an index is corrupted. Test fixtures carry no period.
    sort_keys = ["period", "index"] if "period" in events.columns else ["index"]
    events_sorted = events.sort_values(
        sort_keys, kind="stable").reset_index(drop=True)
    eff = effective_minute(events_sorted)
    rows = []
    for minute in SNAPSHOT_MINUTES:
        prefix_len = prefix_length_at(eff, minute)
        assert_prefix(eff, prefix_len, minute)
        prefix = events_sorted.iloc[:prefix_len]
        rows.append(snapshot_state(prefix, home_team_id, away_team_id, minute,
                                   eff[:prefix_len]))
    return pd.DataFrame(rows)


def main():
    events_path = PROCESSED_DIR / "clean_events.csv"
    match_path = PROCESSED_DIR / "match_store.csv"
    plan_path = PROCESSED_DIR / "snapshot_split_plan.csv"
    for path in [events_path, match_path, plan_path]:
        if not path.exists():
            raise FileNotFoundError(
                f"Missing {path}. Run the Phase 1-4 pipeline before Phase 5B.")

    events = pd.read_csv(events_path, encoding="utf-8")
    matches = pd.read_csv(match_path, encoding="utf-8")
    plan = pd.read_csv(plan_path, encoding="utf-8")

    team_ids = matches.set_index("match_id")[["home_team_id", "away_team_id"]]
    frames = []
    for match_id, group in events.groupby("match_id"):
        if match_id not in team_ids.index:
            continue
        home_id = int(team_ids.loc[match_id, "home_team_id"])
        away_id = int(team_ids.loc[match_id, "away_team_id"])
        snapshots = build_match_snapshots(group, home_id, away_id)
        snapshots.insert(0, "match_id", match_id)
        frames.append(snapshots)

    features = pd.concat(frames, ignore_index=True)
    labelled = features.merge(
        plan[["match_id", "snapshot_minute", "split",
              "label_result", "label_margin"]],
        on=["match_id", "snapshot_minute"], how="inner", validate="one_to_one")

    assert labelled["split"].isin({"train", "validation", "test"}).all(), \
        "snapshot inherited an invalid split"
    assert labelled.groupby("match_id")["split"].nunique().eq(1).all(), \
        "a match's snapshots span more than one split"

    FEATURE_DIR.mkdir(parents=True, exist_ok=True)
    output_path = FEATURE_DIR / "inplay_features.csv"
    labelled.to_csv(output_path, index=False, encoding="utf-8")
    print(f"In-play snapshot rows: {len(labelled)} "
          f"({labelled['match_id'].nunique()} matches x {len(SNAPSHOT_MINUTES)} minutes)")
    print(f"Wrote {output_path}")


if __name__ == "__main__":
    main()
