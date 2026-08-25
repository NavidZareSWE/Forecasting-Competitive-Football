"""Run from the repository root with:

    python src/features/build_team_match_aggregates.py
"""

from pathlib import Path
import sys

import pandas as pd

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

from event_aggregates import QUANTITIES, dismissals, match_frame_aggregates

PROJECT = HERE.parent
PROCESSED_DIR = PROJECT / "reports" / "processed"

CHUNK_ROWS = 400_000
USE_COLUMNS = ["match_id", "index", "period", "minute", "type", "team_id",
               "x", "y", "possession", "play_pattern", "under_pressure",
               "shot_outcome", "shot_xg", "is_goal", "card"]


def aggregate_match(events, match_id, home_team_id, away_team_id):
    home, away = match_frame_aggregates(events, home_team_id, away_team_id)
    rows = []
    for venue, team_id, opponent_id, own, other in [
            ("home", home_team_id, away_team_id, home, away),
            ("away", away_team_id, home_team_id, away, home)]:
        row = {"match_id": match_id, "team_id": team_id, "venue": venue,
               "red_cards": dismissals(events, team_id)}
        for quantity in QUANTITIES:
            row[f"agg_{quantity}"] = own[quantity]
            row[f"agg_opp_{quantity}"] = other[quantity]
        rows.append(row)
    return rows


def stream_aggregates(events_path, matches):
    team_ids = matches.set_index("match_id")[["home_team_id", "away_team_id"]]
    pending = None
    completed = set()
    rows = []

    def flush(frame):
        match_id = int(frame["match_id"].iloc[0])
        if match_id in completed:
            raise AssertionError(
                f"match {match_id} is not contiguous in the event store; "
                "the streaming reader assumes events of one match are adjacent")
        completed.add(match_id)
        if match_id not in team_ids.index:
            return
        rows.extend(aggregate_match(
            frame, match_id,
            int(team_ids.loc[match_id, "home_team_id"]),
            int(team_ids.loc[match_id, "away_team_id"])))

    for chunk in pd.read_csv(events_path, usecols=USE_COLUMNS,
                             chunksize=CHUNK_ROWS, encoding="utf-8"):
        if pending is not None:
            chunk = pd.concat([pending, chunk], ignore_index=True)
        boundaries = list(chunk.groupby("match_id", sort=False))
        for match_id, frame in boundaries[:-1]:
            flush(frame)
        pending = boundaries[-1][1] if boundaries else None
        if len(completed) % 200 == 0 and completed:
            print(f"  aggregated {len(completed)} matches")
    if pending is not None and len(pending):
        flush(pending)
    print(f"  aggregated {len(completed)} matches in total")
    return pd.DataFrame(rows)


def main():
    events_path = PROCESSED_DIR / "clean_events.csv"
    match_path = PROCESSED_DIR / "match_store.csv"
    for path in [events_path, match_path]:
        if not path.exists():
            raise FileNotFoundError(
                f"Missing {path}. Run the Phase 1-4 pipeline first.")

    matches = pd.read_csv(match_path, encoding="utf-8")
    aggregates = stream_aggregates(events_path, matches)

    assert not aggregates.empty, "no aggregates produced"
    assert aggregates.groupby("match_id").size().eq(2).all(), \
        "every match must produce exactly one row per team"

    home_rows = aggregates[aggregates["venue"] == "home"].set_index("match_id")
    scores = matches.set_index("match_id")[["home_score", "away_score"]]
    joined = home_rows.join(scores, how="inner")
    mismatched = joined[(joined["agg_goals"] != joined["home_score"])
                        | (joined["agg_opp_goals"] != joined["away_score"])]
    print(f"Scoreline agreement with the match store: "
          f"{len(joined) - len(mismatched)}/{len(joined)} matches")
    if len(mismatched):
        print(f"  {len(mismatched)} matches disagree; the label store remains "
              f"authoritative and these aggregates are used only as features")

    output_path = PROCESSED_DIR / "team_match_aggregates.csv"
    aggregates.to_csv(output_path, index=False, encoding="utf-8")
    print(f"Rows: {len(aggregates)} "
          f"({aggregates['match_id'].nunique()} matches x 2 teams), "
          f"{aggregates.shape[1]} columns")
    print(f"Wrote {output_path}")


if __name__ == "__main__":
    main()
