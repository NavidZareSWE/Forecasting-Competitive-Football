from pathlib import Path
import sys

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

from event_aggregates import QUANTITIES, match_frame_aggregates

PROJECT = HERE.parent
PROCESSED_DIR = PROJECT / "reports" / "processed"
FEATURE_DIR = PROJECT / "reports" / "features"

SNAPSHOT_MINUTES = list(range(0, 91, 5))
RECENT_WINDOW_MINUTES = 10
DISMISSAL_CARDS = {"Red Card", "Second Yellow"}
ON_TARGET_OUTCOMES = {"Goal", "Saved", "Saved To Post"}
CHUNK_ROWS = 400_000
EVENT_COLUMNS = ["match_id", "index", "period", "minute", "type", "team_id",
                 "x", "y", "possession", "play_pattern", "under_pressure",
                 "shot_outcome", "shot_xg", "is_goal", "card"]


def effective_minute(events_sorted):
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

    recent = prefix[prefix_effective_minutes >=
                    snapshot_minute - RECENT_WINDOW_MINUTES]
    recent_shots = recent[recent["type"] == "Shot"]
    home_recent_xg = float(
        recent_shots.loc[recent_shots["team_id"] == home_team_id, "shot_xg"].sum())
    away_recent_xg = float(
        recent_shots.loc[recent_shots["team_id"] == away_team_id, "shot_xg"].sum())

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
        "inplay_recent_xg_diff": round(home_recent_xg - away_recent_xg, 4),
        "inplay_events_so_far": int(len(prefix)),
        **aggregate_state(prefix, recent, home_team_id, away_team_id,
                          snapshot_minute),
    }


def aggregate_state(prefix, recent, home_team_id, away_team_id,
                    snapshot_minute):
    home, away = match_frame_aggregates(prefix, home_team_id, away_team_id)
    recent_home, recent_away = match_frame_aggregates(recent, home_team_id,
                                                      away_team_id)
    elapsed = max(float(snapshot_minute), 1.0)

    state = {}
    for quantity in QUANTITIES:
        state[f"inplay_home_{quantity}"] = round(float(home[quantity]), 4)
        state[f"inplay_away_{quantity}"] = round(float(away[quantity]), 4)
        state[f"inplay_diff_{quantity}"] = round(
            float(home[quantity]) - float(away[quantity]), 4)
        state[f"inplay_recent_diff_{quantity}"] = round(
            float(recent_home[quantity]) - float(recent_away[quantity]), 4)
        state[f"inplay_rate_diff_{quantity}"] = round(
            (float(home[quantity]) - float(away[quantity])) / elapsed, 5)
    state["inplay_minutes_remaining"] = max(90 - int(snapshot_minute), 0)
    return state


def build_match_snapshots(events, home_team_id, away_team_id):
    sort_keys = ["period", "index"] if "period" in events.columns else ["index"]
    events_sorted = events.sort_values(
        sort_keys, kind="stable").reset_index(drop=True)
    effective_minutes = effective_minute(events_sorted)
    rows = []
    for minute in SNAPSHOT_MINUTES:
        prefix_len = prefix_length_at(effective_minutes, minute)
        assert_prefix(effective_minutes, prefix_len, minute)
        prefix = events_sorted.iloc[:prefix_len]
        rows.append(snapshot_state(prefix, home_team_id, away_team_id, minute,
                                   effective_minutes[:prefix_len]))
    return pd.DataFrame(rows)


def main():
    events_path = PROCESSED_DIR / "clean_events.csv"
    match_path = PROCESSED_DIR / "match_store.csv"
    plan_path = PROCESSED_DIR / "snapshot_split_plan.csv"
    for path in [events_path, match_path, plan_path]:
        if not path.exists():
            raise FileNotFoundError(
                f"Missing {path}. Run the Phase 1-4 pipeline before Phase 5B.")

    matches = pd.read_csv(match_path, encoding="utf-8")
    plan = pd.read_csv(plan_path, encoding="utf-8")

    team_ids = matches.set_index("match_id")[["home_team_id", "away_team_id"]]
    frames = []
    completed = set()

    def process(group):
        match_id = int(group["match_id"].iloc[0])
        if match_id in completed:
            raise AssertionError(
                f"match {match_id} is not contiguous in the event store")
        completed.add(match_id)
        if match_id not in team_ids.index:
            return
        snapshots = build_match_snapshots(
            group,
            int(team_ids.loc[match_id, "home_team_id"]),
            int(team_ids.loc[match_id, "away_team_id"]))
        snapshots.insert(0, "match_id", match_id)
        frames.append(snapshots)

    pending = None
    for chunk in pd.read_csv(events_path, usecols=EVENT_COLUMNS,
                             chunksize=CHUNK_ROWS, encoding="utf-8"):
        if pending is not None:
            chunk = pd.concat([pending, chunk], ignore_index=True)
        groups = list(chunk.groupby("match_id", sort=False))
        for _, group in groups[:-1]:
            process(group)
        pending = groups[-1][1] if groups else None
        if len(frames) and len(frames) % 200 == 0:
            print(f"  {len(frames)} matches snapshotted")
    if pending is not None and len(pending):
        process(pending)

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
