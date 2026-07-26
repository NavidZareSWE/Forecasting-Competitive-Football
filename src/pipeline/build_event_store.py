import os
import json
import time
import urllib.request
from http.client import IncompleteRead
from concurrent.futures import ThreadPoolExecutor, as_completed
from urllib.error import URLError
from pathlib import Path

import pandas as pd

# --- Configuration ---
RAW_BASE = "https://raw.githubusercontent.com/statsbomb/open-data/master/data"
DOWNLOAD_ATTEMPTS = 5
DOWNLOAD_TIMEOUT_SECONDS = 120
DOWNLOAD_WORKERS = int(os.environ.get("SB_DOWNLOAD_WORKERS", 6))

HERE = Path(__file__).resolve().parent
PROJECT = HERE.parent
LOCAL_SB_DIR = Path(
    os.environ.get("STATSBOMB_LOCAL", PROJECT / "data" /
                   "statsbomb_open_data" / "data")
)
PROCESSED_DIR = PROJECT / "reports" / "processed"
PROCESSED_DIR.mkdir(parents=True, exist_ok=True)


def load_json(local_path: Path, url: str):
    if local_path.exists():
        return json.loads(local_path.read_text(encoding="utf-8"))

    request = urllib.request.Request(
        url, headers={"User-Agent": "football-forecast-course-project"})
    for attempt in range(1, DOWNLOAD_ATTEMPTS + 1):
        try:
            with urllib.request.urlopen(request, timeout=DOWNLOAD_TIMEOUT_SECONDS) as response:
                data = json.load(response)
            break
        except (TimeoutError, IncompleteRead, json.JSONDecodeError, URLError, OSError) as error:
            if attempt == DOWNLOAD_ATTEMPTS:
                raise RuntimeError(
                    f"Could not download {url} after {DOWNLOAD_ATTEMPTS} attempts. "
                    "Check your network connection and rerun; cached files will be reused."
                ) from error
            wait_seconds = attempt * 2
            print(
                f"Download attempt {attempt}/{DOWNLOAD_ATTEMPTS} failed for {local_path.name}: "
                f"{error}. Retrying in {wait_seconds}s..."
            )
            time.sleep(wait_seconds)
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
        team = event.get("team") or {}
        player = event.get("player") or {}
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
            "team_id": team.get("id"),
            "team": team.get("name"),
            "player_id": player.get("id"),
            "player": player.get("name"),
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

    events_table = pd.DataFrame(rows).sort_values(
        "index").reset_index(drop=True)
    assert events_table["index"].is_unique, f"Duplicate event index in match {match_id}"
    assert events_table[
        "index"].is_monotonic_increasing, f"Events unordered in match {match_id}"
    return events_table


def goals_from_events(events_table: pd.DataFrame) -> dict:
    scored = events_table[events_table["is_goal"]].groupby("team_id").size()
    own_goals = events_table[events_table["type"]
                             == "Own Goal For"].groupby("team_id").size()
    total = scored.add(own_goals, fill_value=0)
    return {team_id: int(count) for team_id, count in total.items()}


def summarise_events(match, events):
    goals = goals_from_events(events)
    home_goals = goals.get(match["home_team_id"], 0)
    away_goals = goals.get(match["away_team_id"], 0)
    return {
        "match_id": int(match["match_id"]),
        "n_events": len(events),
        "home_goals_events": home_goals,
        "away_goals_events": away_goals,
        "score_ok": bool(
            home_goals == match["home_score"] and away_goals == match["away_score"]
        ),
        "home_xg": round(float(events[events["team_id"] == match["home_team_id"]]["shot_xg"].sum()), 3),
        "away_xg": round(float(events[events["team_id"] == match["away_team_id"]]["shot_xg"].sum()), 3),
    }


def build_event_store(store: pd.DataFrame) -> pd.DataFrame:
    rows = []
    match_rows = [match for _, match in store.iterrows()]
    workers = max(1, min(DOWNLOAD_WORKERS, len(match_rows)))
    print(f"Loading {len(match_rows)} event files with {workers} worker(s)...")

    with ThreadPoolExecutor(max_workers=workers) as executor:
        futures = {
            executor.submit(load_events, int(match["match_id"])): match
            for match in match_rows
        }
        for completed, future in enumerate(as_completed(futures), start=1):
            match = futures[future]
            rows.append(summarise_events(match, future.result()))
            if completed % 100 == 0 or completed == len(futures):
                print(f"  Events loaded: {completed}/{len(futures)}")

    index_table = pd.DataFrame(rows).sort_values(
        "match_id").reset_index(drop=True)
    reconstructed = int(index_table["score_ok"].sum())
    print(
        f"Score cross-check: {reconstructed}/{len(index_table)} matches reconstruct exactly")
    output_path = PROCESSED_DIR / "events_index.csv"
    index_table.to_csv(output_path, index=False, encoding="utf-8")
    print(f"Wrote event index -> {output_path}")
    return index_table


if __name__ == "__main__":
    store = pd.read_csv(PROCESSED_DIR / "match_store.csv", encoding="utf-8")
    limit = int(os.environ.get("SB_MAX_MATCHES", 0)) or None
    build_event_store(store.head(limit) if limit else store)
