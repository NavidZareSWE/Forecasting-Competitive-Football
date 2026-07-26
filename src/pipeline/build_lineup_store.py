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


def load_lineup(match_id: int) -> pd.DataFrame:
    path = LOCAL_SB_DIR / "lineups" / f"{match_id}.json"
    url = f"{RAW_BASE}/lineups/{match_id}.json"
    teams = load_json(path, url)

    rows = []
    for team in teams:
        for player in team["lineup"]:
            positions = player.get("positions", [])
            first_position = positions[0] if positions else {}
            started = (
                bool(positions)
                and first_position.get("from") == "00:00"
                and first_position.get("from_period") == 1
            )
            rows.append({
                "match_id": match_id,
                "team_id": team["team_id"],
                "team_name": team["team_name"],
                "player_id": player["player_id"],
                "player_name": player["player_name"],
                "jersey_number": player.get("jersey_number"),
                "country": (player.get("country") or {}).get("name"),
                "primary_position": first_position.get("position"),
                "started": started,
                "n_cards": len(player.get("cards", [])),
            })
    return pd.DataFrame(rows)


def build_lineup_store(match_ids) -> pd.DataFrame:
    table = pd.concat([load_lineup(mid) for mid in match_ids], ignore_index=True)
    assert table["player_id"].notna().all(), "Missing player_id in lineups!"
    output_path = PROCESSED_DIR / "lineups.csv"
    table.to_csv(output_path, index=False, encoding="utf-8")
    print(
        f"Wrote {len(table)} player-rows for {table['match_id'].nunique()} "
        f"matches -> {output_path}"
    )
    return table


if __name__ == "__main__":
    store = pd.read_csv(PROCESSED_DIR / "match_store.csv", encoding="utf-8")
    match_ids = store["match_id"].tolist()
    limit = int(os.environ.get("SB_MAX_MATCHES", 0)) or None
    build_lineup_store(match_ids[:limit] if limit else match_ids)
