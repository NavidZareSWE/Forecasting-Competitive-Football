"""Run from the repository root with:

    python src/pipeline/download_extended_data.py
"""
from pathlib import Path
import hashlib
import sqlite3
import sys
import urllib.request

import pandas as pd

HERE = Path(__file__).resolve().parent
PROJECT = HERE.parent
DATA_DIR = PROJECT / "data"
PROCESSED_DIR = PROJECT / "reports" / "processed"

ESD_URL = ("https://huggingface.co/datasets/julien-c/kaggle-hugomathien-soccer"
           "/resolve/main/database.sqlite")
ESD_PATH = DATA_DIR / "european_soccer_db" / "database.sqlite"
ESD_MATCH_COUNT = 25979
ESD_LEAGUE_COUNT = 11

FOOTBALL_DATA_URL = "https://www.football-data.co.uk/mmz4281/{season}/{div}.csv"
FOOTBALL_DATA_DIR = DATA_DIR / "Football_Data"
FOOTBALL_DATA_DIVS = ["E0", "SP1", "I1", "F1", "D1", "N1", "P1", "B1", "SC0"]
# Complete seasons only. A season still in progress must NOT be added: it
# would enter the store as an undersized (league, season) cell, break the
# split contract's "two full seasons per holdout" shape, and give a test set
# weighted towards August fixtures. 2026/27 is excluded for exactly that
# reason - it was ~10 matches old when 2025/26 was added.
FOOTBALL_DATA_SEASONS = ["1617", "1718", "1819", "1920", "2021",
                         "2122", "2223", "2324", "2425", "2526"]
FOOTBALL_DATA_REQUIRED = ["Div", "Date", "HomeTeam", "AwayTeam",
                          "FTHG", "FTAG", "B365H", "B365D", "B365A"]

FIFA_LEGACY_URL = ("https://huggingface.co/datasets/jsulz/FIFA23"
                   "/resolve/main/male_players%20(legacy).csv")
FIFA_DIR = DATA_DIR / "fifa_ratings"
FIFA_LEGACY_PATH = FIFA_DIR / "male_players_legacy.csv"

FIFA_FULL_URL = ("https://huggingface.co/datasets/jsulz/FIFA23"
                 "/resolve/main/male_players.csv")
FIFA_FILTERED_PATH = FIFA_DIR / "fifa_players_filtered.csv"
FIFA_LEAGUE_IDS = {13: "Premier League", 53: "La Liga", 31: "Serie A",
                   16: "Ligue 1", 19: "Bundesliga", 10: "Eredivisie",
                   308: "Primeira Liga", 4: "Pro League",
                   50: "Scottish Premiership", 66: "Ekstraklasa",
                   189: "Swiss Super League"}
FIFA_KEEP_COLUMNS = ["player_id", "fifa_version", "fifa_update",
                     "fifa_update_date", "short_name", "long_name",
                     "player_positions", "overall", "potential", "age", "dob",
                     "league_id", "league_name", "club_team_id", "club_name"]
FIFA_OPTIONAL = {
    FIFA_DIR / "fc24_players.csv":
        "kagglehub: stefanoleone992/ea-sports-fc-24-complete-player-dataset "
        "(male_players.csv)",
    FIFA_DIR / "fc25_players.csv":
        "kagglehub: any EA FC 25 sofifa player dataset with sofifa_id + "
        "overall + club columns",
}


def download(url, path, label):
    if path.exists() and path.stat().st_size > 0:
        print(f"  cached  {label}: {path.relative_to(PROJECT)}")
        return True
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".part")
    print(f"  fetch   {label}: {url}")
    try:
        request = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
        with urllib.request.urlopen(request, timeout=120) as response, \
                open(tmp, "wb") as sink:
            while True:
                block = response.read(1 << 20)
                if not block:
                    break
                sink.write(block)
        tmp.rename(path)
        print(f"          -> {path.relative_to(PROJECT)} "
              f"({path.stat().st_size / 1e6:.1f} MB)")
        return True
    except Exception as error:
        tmp.unlink(missing_ok=True)
        print(f"  FAILED  {label}: {error}")
        return False


def sha8(path):
    digest = hashlib.sha256()
    with open(path, "rb") as source:
        for block in iter(lambda: source.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()[:8]


def verify_esd():
    with sqlite3.connect(ESD_PATH) as connection:
        matches = connection.execute("SELECT COUNT(*) FROM Match").fetchone()[0]
        leagues = connection.execute("SELECT COUNT(*) FROM League").fetchone()[0]
    assert matches == ESD_MATCH_COUNT, \
        f"ESD Match table has {matches} rows, expected {ESD_MATCH_COUNT}"
    assert leagues == ESD_LEAGUE_COUNT, \
        f"ESD League table has {leagues} rows, expected {ESD_LEAGUE_COUNT}"
    print(f"  verified ESD: {matches} matches, {leagues} leagues")


def read_football_data_csv(path):
    """Season CSVs vary: some carry a UTF-8 BOM, some are latin-1."""
    try:
        return pd.read_csv(path, encoding="utf-8-sig", on_bad_lines="skip")
    except UnicodeDecodeError:
        return pd.read_csv(path, encoding="latin-1", on_bad_lines="skip")


def verify_football_data(path):
    frame = read_football_data_csv(path)
    frame = frame.dropna(subset=["Date"]) if "Date" in frame.columns else frame
    missing = [c for c in FOOTBALL_DATA_REQUIRED if c not in frame.columns]
    assert not missing, f"{path.name}: missing columns {missing}"
    assert len(frame) >= 150, \
        f"{path.name}: only {len(frame)} rows; not a plausible season file"
    return len(frame)


def stream_filter_fifa_full():
    """Stream male_players.csv and keep only our leagues' rows."""
    if FIFA_FILTERED_PATH.exists() and FIFA_FILTERED_PATH.stat().st_size > 0:
        print(f"  cached  FIFA all-updates filtered: "
              f"{FIFA_FILTERED_PATH.relative_to(PROJECT)}")
        return True
    FIFA_DIR.mkdir(parents=True, exist_ok=True)
    tmp = FIFA_FILTERED_PATH.with_suffix(".csv.part")
    print(f"  stream  FIFA all-updates (5.6 GB upstream, filtered in flight)")
    try:
        request = urllib.request.Request(
            FIFA_FULL_URL, headers={"User-Agent": "Mozilla/5.0"})
        with urllib.request.urlopen(request, timeout=300) as response:
            first = True
            kept = 0
            for chunk in pd.read_csv(response, usecols=FIFA_KEEP_COLUMNS,
                                     chunksize=200_000, low_memory=False):
                subset = chunk[chunk["league_id"].isin(FIFA_LEAGUE_IDS)]
                if len(subset):
                    subset.to_csv(tmp, index=False, mode="w" if first else "a",
                                  header=first, encoding="utf-8")
                    first = False
                    kept += len(subset)
        tmp.rename(FIFA_FILTERED_PATH)
        print(f"          -> {FIFA_FILTERED_PATH.relative_to(PROJECT)} "
              f"({kept} rows, {FIFA_FILTERED_PATH.stat().st_size / 1e6:.0f} MB)")
        return True
    except Exception as error:
        tmp.unlink(missing_ok=True)
        print(f"  FAILED  FIFA all-updates: {error}")
        return False


def main():
    manifest = []
    failures = []

    print("European Soccer Database:")
    if download(ESD_URL, ESD_PATH, "ESD sqlite"):
        verify_esd()
        manifest.append({"path": str(ESD_PATH.relative_to(PROJECT)),
                         "rows": ESD_MATCH_COUNT, "sha256_8": sha8(ESD_PATH)})
    else:
        failures.append(f"ESD sqlite: download {ESD_URL} manually to {ESD_PATH}")

    print(f"football-data.co.uk seasons 2016/17-"
          f"20{FOOTBALL_DATA_SEASONS[-1][:2]}/{FOOTBALL_DATA_SEASONS[-1][2:]}:")
    for season in FOOTBALL_DATA_SEASONS:
        for div in FOOTBALL_DATA_DIVS:
            path = FOOTBALL_DATA_DIR / season / f"{div}.csv"
            url = FOOTBALL_DATA_URL.format(season=season, div=div)
            if download(url, path, f"{season}/{div}"):
                rows = verify_football_data(path)
                manifest.append({"path": str(path.relative_to(PROJECT)),
                                 "rows": rows, "sha256_8": sha8(path)})
            else:
                failures.append(f"football-data {season}/{div}: {url}")

    print("FIFA player ratings:")
    if download(FIFA_LEGACY_URL, FIFA_LEGACY_PATH, "FIFA 15-23 legacy"):
        header = pd.read_csv(FIFA_LEGACY_PATH, nrows=5)
        for column in ["player_id", "overall", "fifa_version"]:
            assert column in header.columns, \
                f"FIFA legacy CSV missing expected column {column!r}; " \
                f"columns start: {list(header.columns)[:12]}"
        manifest.append({"path": str(FIFA_LEGACY_PATH.relative_to(PROJECT)),
                         "rows": -1, "sha256_8": sha8(FIFA_LEGACY_PATH)})
    else:
        failures.append(f"FIFA legacy: {FIFA_LEGACY_URL}")

    if stream_filter_fifa_full():
        frame = pd.read_csv(FIFA_FILTERED_PATH,
                            usecols=["fifa_version"], low_memory=False)
        versions = sorted(frame["fifa_version"].unique())
        assert versions[-1] >= 23, \
            f"filtered FIFA file tops out at version {versions[-1]}"
        manifest.append({"path": str(FIFA_FILTERED_PATH.relative_to(PROJECT)),
                         "rows": len(frame),
                         "sha256_8": sha8(FIFA_FILTERED_PATH)})
    else:
        failures.append(f"FIFA all-updates: {FIFA_FULL_URL}")

    for path, hint in FIFA_OPTIONAL.items():
        if path.exists():
            print(f"  present {path.name}")
            manifest.append({"path": str(path.relative_to(PROJECT)),
                             "rows": -1, "sha256_8": sha8(path)})
        else:
            print(f"  absent  {path.name} (optional) — {hint}; "
                  "FIFA 23 ratings will be carried forward for later seasons")

    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    manifest_path = PROCESSED_DIR / "download_manifest.csv"
    pd.DataFrame(manifest).to_csv(manifest_path, index=False, encoding="utf-8")
    print(f"Wrote {manifest_path} ({len(manifest)} files)")

    if failures:
        print("\nRequired downloads failed:")
        for failure in failures:
            print(f"  - {failure}")
        sys.exit(1)
    print("All required inputs present.")


if __name__ == "__main__":
    main()
