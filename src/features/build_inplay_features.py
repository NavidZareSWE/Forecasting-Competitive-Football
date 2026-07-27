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


# --- Effective-minute cut (the time-t leakage boundary) --------------------
def effective_minute(events_sorted):
    # Some events carry corrupted 00:00:00 timestamps while their integer index
    # position is correct. Ordering by index and taking the running maximum of
    # the minute column yields a non-decreasing "effective minute", so a stray
    # low minute can never pull an event earlier than its true index position.
    return np.maximum.accumulate(events_sorted["minute"].to_numpy())


def prefix_length_at(effective_minutes, snapshot_minute):
    # Number of leading events whose effective minute is <= t. Because the
    # effective minute is non-decreasing, this set is always a contiguous
    # index-ordered prefix of the match's events.
    return int(np.searchsorted(effective_minutes, snapshot_minute, side="right"))


def assert_prefix(effective_minutes, prefix_len, snapshot_minute):
    if prefix_len > 0:
        assert effective_minutes[prefix_len - 1] <= snapshot_minute, \
            "cut included an event after time t"
    if prefix_len < len(effective_minutes):
        assert effective_minutes[prefix_len] > snapshot_minute, \
            "cut excluded an event at or before time t; not a clean prefix"


# --- Snapshot state features (events strictly up to and including t) --------
def goals_for(prefix, team_id):
    scored = int(((prefix["team_id"] == team_id) & prefix["is_goal"]).sum())
    own = int(((prefix["team_id"] == team_id) &
               (prefix["type"] == "Own Goal For")).sum())
    return scored + own


def snapshot_state(prefix, home_team_id, away_team_id, snapshot_minute):
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

    recent = prefix[prefix["minute"] >= snapshot_minute - RECENT_WINDOW_MINUTES]
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
    }


def build_match_snapshots(events, home_team_id, away_team_id):
    events_sorted = events.sort_values("index", kind="stable").reset_index(drop=True)
    eff = effective_minute(events_sorted)
    rows = []
    for minute in SNAPSHOT_MINUTES:
        prefix_len = prefix_length_at(eff, minute)
        assert_prefix(eff, prefix_len, minute)
        prefix = events_sorted.iloc[:prefix_len]
        rows.append(snapshot_state(prefix, home_team_id, away_team_id, minute))
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
