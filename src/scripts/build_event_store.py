import os
import json
import urllib.request
from pathlib import Path

import pandas as pd

# --- Configuration ---
RAW_BASE = "https://raw.githubusercontent.com/statsbomb/open-data/master/data"

HERE = Path(__file__).resolve().parent
PROJECT = HERE.parent
LOCAL_SB_DIR = Path(
    os.environ.get("STATSBOMB_LOCAL", PROJECT / "data" / "statsbomb_open_data" / "data")
)
PROCESSED_DIR = PROJECT / "reports" / "processed"
PROCESSED_DIR.mkdir(parents=True, exist_ok=True)


def load_json(local_path: Path, url: str):
    if local_path.exists():
        return json.loads(local_path.read_text(encoding="utf-8"))
    with urllib.request.urlopen(url, timeout=60) as response:
        data = json.load(response)
    try:
        local_path.parent.mkdir(parents=True, exist_ok=True)
        local_path.write_text(json.dumps(data), encoding="utf-8")
    except OSError:
        pass
    return data


def load_events(match_id: int) -> pd.DataFrame:
    path = LOCAL_SB_DIR / "events" / f"{match_id}.json"
    url = f"{RAW_BASE}/events/{match_id}.json"
    events = load_json(path, url)

    rows = []
    for event in events:
        location = event.get("location") or [None, None]
        shot = event.get("shot") or {}
        foul = event.get("foul_committed") or {}
        bad_behaviour = event.get("bad_behaviour") or {}
        card = (foul.get("card") or bad_behaviour.get("card") or {}).get("name")
        shot_outcome = (shot.get("outcome") or {}).get("name")
        rows.append({
            "match_id": match_id,
            "event_id": event["id"],
            "index": event["index"],
            "period": event["period"],
            "timestamp": event["timestamp"],
            "minute": event["minute"],
            "second": event["second"],
            "type": event["type"]["name"],
            "team": (event.get("team") or {}).get("name"),
            "player": (event.get("player") or {}).get("name"),
            "x": location[0],
            "y": location[1],
            "possession": event.get("possession"),
            "play_pattern": (event.get("play_pattern") or {}).get("name"),
            "under_pressure": bool(event.get("under_pressure", False)),
            "duration": event.get("duration"),
            "shot_outcome": shot_outcome,
            "shot_xg": shot.get("statsbomb_xg"),
            "is_goal": shot_outcome == "Goal",
            "card": card,
        })

    events_table = pd.DataFrame(rows).sort_values("index").reset_index(drop=True)
    assert events_table["index"].is_unique, f"Duplicate event index in match {match_id}"
    assert events_table["index"].is_monotonic_increasing, f"Events unordered in match {match_id}"
    return events_table


def goals_from_events(events_table: pd.DataFrame) -> dict:
    scored = events_table[events_table["is_goal"]].groupby("team").size()
    own_goals = events_table[events_table["type"] == "Own Goal For"].groupby("team").size()
    total = scored.add(own_goals, fill_value=0)
    return {team: int(count) for team, count in total.items()}


def build_event_store(store: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for _, match in store.iterrows():
        events = load_events(int(match["match_id"]))
        goals = goals_from_events(events)
        home_goals = goals.get(match["home_team"], 0)
        away_goals = goals.get(match["away_team"], 0)
        rows.append({
            "match_id": int(match["match_id"]),
            "n_events": len(events),
            "home_goals_events": home_goals,
            "away_goals_events": away_goals,
            "score_ok": bool(
                home_goals == match["home_score"] and away_goals == match["away_score"]
            ),
            "home_xg": round(float(events[events["team"] == match["home_team"]]["shot_xg"].sum()), 3),
            "away_xg": round(float(events[events["team"] == match["away_team"]]["shot_xg"].sum()), 3),
        })

    index_table = pd.DataFrame(rows)
    reconstructed = int(index_table["score_ok"].sum())
    print(f"Score cross-check: {reconstructed}/{len(index_table)} matches reconstruct exactly")
    output_path = PROCESSED_DIR / "events_index.csv"
    index_table.to_csv(output_path, index=False, encoding="utf-8")
    print(f"Wrote event index -> {output_path}")
    return index_table


if __name__ == "__main__":
    store = pd.read_csv(PROCESSED_DIR / "match_store.csv", encoding="utf-8")
    limit = int(os.environ.get("SB_MAX_MATCHES", 0)) or None
    build_event_store(store.head(limit) if limit else store)
