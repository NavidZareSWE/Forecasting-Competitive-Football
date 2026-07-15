import os
import json
import urllib.request
from pathlib import Path

import pandas as pd

# --- Configuration ---
LEAGUES = {
    (2, 27):  "Premier League",
    (7, 27):  "Ligue 1",
    (11, 27): "La Liga",
    (12, 27): "Serie A",
}
RAW_BASE = "https://raw.githubusercontent.com/statsbomb/open-data/master/data"

HERE = Path(__file__).resolve().parent
PROJECT = HERE.parent
LOCAL_SB_DIR = Path(
    os.environ.get("STATSBOMB_LOCAL", PROJECT / "data" /
                   "statsbomb_open_data" / "data")
)
OUT_DIR = PROJECT / "reports" / "processed"
OUT_DIR.mkdir(parents=True, exist_ok=True)


def matches_path(comp_id: int, season_id: int) -> Path:
    return LOCAL_SB_DIR / "matches" / str(comp_id) / f"{season_id}.json"


def load_matches(comp_id: int, season_id: int) -> tuple[list, str]:
    path = matches_path(comp_id, season_id)
    if path.exists():
        return json.loads(path.read_text(encoding="utf-8")), "local"

    url = f"{RAW_BASE}/matches/{comp_id}/{season_id}.json"
    with urllib.request.urlopen(url, timeout=60) as response:
        data = json.load(response)

    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(data), encoding="utf-8")
        source = "downloaded + cached"
    except OSError:
        source = "downloaded (cache write failed)"

    return data, source


def flatten_match(match: dict, comp_name: str) -> dict:
    home = match["home_score"]
    away = match["away_score"]
    if home > away:
        result = "H"
    elif home < away:
        result = "A"
    else:
        result = "D"
    margin = home - away
    return {
        "match_id": match["match_id"],
        "competition_id": match["competition"]["competition_id"],
        "competition_name": comp_name,
        "season_id": match["season"]["season_id"],
        "match_date": match["match_date"],
        "kick_off": match.get("kick_off"),
        "home_team_id": match["home_team"]["home_team_id"],
        "home_team": match["home_team"]["home_team_name"],
        "away_team_id": match["away_team"]["away_team_id"],
        "away_team": match["away_team"]["away_team_name"],
        "home_score": home,
        "away_score": away,
        "result": result,
        "margin_raw": margin,
        "margin": max(-5, min(5, margin)),
        "match_status": match.get("match_status"),
        "match_status_360": match.get("match_status_360"),
    }


def main() -> pd.DataFrame:
    rows = []
    for (comp_id, season_id), comp_name in LEAGUES.items():
        matches, source = load_matches(comp_id, season_id)
        for match in matches:
            rows.append(flatten_match(match, comp_name))
        print(
            f"  {comp_name:15s} comp={comp_id:<3} season={season_id}: "
            f"{len(matches):>3} matches  [{source}]"
        )

    df = pd.DataFrame(rows)
    df["match_date"] = pd.to_datetime(df["match_date"])
    df = df.sort_values(["match_date", "match_id"]).reset_index(drop=True)

    # --- Self-checks ---
    assert df["match_id"].is_unique, "Duplicate match_id across leagues!"
    assert set(df["competition_name"]) == set(
        LEAGUES.values()), "League set drifted!"
    assert df["result"].isin(["H", "D", "A"]).all(), "Bad result label!"
    assert (df["margin"].between(-5, 5)).all(), "Margin not clipped!"
    assert df[["home_score", "away_score"]
              ].notna().all().all(), "Missing score!"

    output_path = OUT_DIR / "match_store.csv"
    df.to_csv(output_path, index=False, encoding="utf-8")
    print(f"\nWrote {len(df)} matches -> {output_path}")
    return df


if __name__ == "__main__":
    print(f"Local dataset dir: {LOCAL_SB_DIR}")
    df = main()
    print("\n=== Match store summary ===")
    print(df.groupby("competition_name").size().to_string())
    print("\nColumns:", list(df.columns))
