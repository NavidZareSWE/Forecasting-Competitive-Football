from pathlib import Path

import pandas as pd

from build_event_store import load_events


HERE = Path(__file__).resolve().parent
PROJECT = HERE.parent
PROCESSED_DIR = PROJECT / "reports" / "processed"

MATCH_COLUMNS = {
    "match_id", "competition_id", "season_id", "home_team_id", "home_team",
    "away_team_id", "away_team",
}
LINEUP_COLUMNS = {"match_id", "team_id",
                  "team_name", "player_id", "player_name"}


def drop_log_rows(frame, mask, source, reason):
    dropped = frame.loc[mask].copy()
    if dropped.empty:
        return pd.DataFrame(columns=["source", "match_id", "record_id", "reason"])
    record_id = dropped["event_id"] if "event_id" in dropped else dropped.index.astype(
        str)
    return pd.DataFrame({
        "source": source,
        "match_id": dropped.get("match_id"),
        "record_id": record_id,
        "reason": reason,
    })


def clean_events(match_store):
    cleaned_frames, drop_frames, ordering_rows = [], [], []
    for match_id in match_store["match_id"]:
        events = load_events(int(match_id)).copy()
        events["match_id"] = int(match_id)
        events["event_time"] = pd.to_timedelta(
            events["timestamp"], errors="coerce")

        invalid = (
            events["event_id"].isna()
            | events["index"].isna()
            | events["period"].isna()
            | events["event_time"].isna()
        )
        drop_frames.append(drop_log_rows(
            events, invalid, "events", "missing required event identity or timestamp"))
        events = events.loc[~invalid].copy()

        ordered = events.sort_values(
            ["period", "event_time", "index"], kind="stable")
        assert ordered["event_id"].is_unique, f"Duplicate event_id in match {match_id}!"

        ordering_rows.append({
            "scope": "events", "field": "period,timestamp,index", "affected_rows": int(
                (events["event_id"].to_numpy() !=
                 ordered["event_id"].to_numpy()).sum()
            ),
            "reason": "events reordered into canonical event-time order",
        })

        ordered["event_time_seconds"] = ordered["event_time"].dt.total_seconds()
        cleaned_frames.append(ordered.drop(columns="event_time"))

    clean_table = pd.concat(cleaned_frames, ignore_index=True)
    drop_log = pd.concat(drop_frames, ignore_index=True)
    return clean_table, drop_log, pd.DataFrame(ordering_rows)


def clean_lineups(lineups):
    invalid = lineups[["match_id", "team_id", "player_id"]].isna().any(axis=1)
    drop_log = drop_log_rows(lineups, invalid, "lineups",
                             "missing match, team, or player identifier")
    return lineups.loc[~invalid].copy(), drop_log


def canonical_map(observations, id_column, name_column, entity):
    observations = observations.dropna(subset=[id_column, name_column]).copy()
    observations["name_normalized"] = observations[name_column].str.strip(
    ).str.casefold()
    counts = observations.groupby(
        [id_column, "name_normalized"], as_index=False).size()
    counts = counts.sort_values(
        [id_column, "size", "name_normalized"], ascending=[True, False, True])
    canonical = counts.drop_duplicates(id_column).rename(columns={
        "name_normalized": f"canonical_{entity}_name",
        "size": "canonical_name_observations",
    })
    variants = counts.groupby(id_column).size().rename(
        "name_variants").reset_index()
    canonical = canonical.merge(variants, on=id_column, how="left")
    return canonical.rename(columns={id_column: f"{entity}_id"})


def quality_log(match_store, events, lineups, ordering_audit):
    rows = ordering_audit.to_dict("records")
    for column in ["team_id", "player_id", "x", "y", "shot_xg", "card"]:
        rows.append({
            "scope": "events", "field": column, "affected_rows": int(events[column].isna().sum()),
            "reason": "optional attribute absent; retained as missing",
        })
    for column in ["primary_position", "country", "jersey_number"]:
        rows.append({
            "scope": "lineups", "field": column, "affected_rows": int(lineups[column].isna().sum()),
            "reason": "optional attribute absent; retained as missing",
        })
    lineup_match_ids = set(lineups["match_id"])
    for match_id in match_store.loc[~match_store["match_id"].isin(lineup_match_ids), "match_id"]:
        rows.append({
            "scope": "lineups", "field": "match_id", "affected_rows": 1,
            "reason": f"no lineup rows available for match_id={match_id}; match retained",
        })
    return pd.DataFrame(rows)


def main():
    match_path = PROCESSED_DIR / "match_store.csv"
    lineup_path = PROCESSED_DIR / "lineups.csv"
    for path in [match_path, lineup_path]:
        if not path.exists():
            raise FileNotFoundError(
                f"Missing {path}. Run the Phase 1 stores before cleaning.")

    match_store = pd.read_csv(match_path, encoding="utf-8")
    lineups = pd.read_csv(lineup_path, encoding="utf-8")
    assert MATCH_COLUMNS <= set(
        match_store), "Rebuild match_store.csv to include stable team IDs."
    assert LINEUP_COLUMNS <= set(
        lineups), "lineups.csv is missing required identity columns."

    clean_event_table, event_drops, ordering_audit = clean_events(match_store)
    clean_lineup_table, lineup_drops = clean_lineups(lineups)

    team_observations = pd.concat([
        match_store[["home_team_id", "home_team"]].rename(
            columns={"home_team_id": "team_id", "home_team": "team_name"}),
        match_store[["away_team_id", "away_team"]].rename(
            columns={"away_team_id": "team_id", "away_team": "team_name"}),
        clean_lineup_table[["team_id", "team_name"]],
        clean_event_table[["team_id", "team"]].rename(
            columns={"team": "team_name"}),
    ], ignore_index=True)
    player_observations = pd.concat([
        clean_lineup_table[["player_id", "player_name"]],
        clean_event_table[["player_id", "player"]].rename(
            columns={"player": "player_name"}),
    ], ignore_index=True)

    team_map = canonical_map(team_observations, "team_id", "team_name", "team")
    player_map = canonical_map(
        player_observations, "player_id", "player_name", "player")
    drops = pd.concat([event_drops, lineup_drops], ignore_index=True)
    quality = quality_log(match_store, clean_event_table,
                          clean_lineup_table, ordering_audit)

    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    clean_event_table.to_csv(
        PROCESSED_DIR / "clean_events.csv", index=False, encoding="utf-8")
    clean_lineup_table.to_csv(
        PROCESSED_DIR / "clean_lineups.csv", index=False, encoding="utf-8")
    team_map.to_csv(PROCESSED_DIR / "team_identity_map.csv",
                    index=False, encoding="utf-8")
    player_map.to_csv(PROCESSED_DIR / "player_identity_map.csv",
                      index=False, encoding="utf-8")
    drops.to_csv(PROCESSED_DIR / "cleaning_drops.csv",
                 index=False, encoding="utf-8")
    quality.to_csv(PROCESSED_DIR / "data_quality_log.csv",
                   index=False, encoding="utf-8")

    print(
        f"Clean events: {len(clean_event_table)} | clean lineups: {len(clean_lineup_table)}")
    print(
        f"Canonical teams: {len(team_map)} | canonical players: {len(player_map)}")
    print(
        f"Dropped rows logged: {len(drops)} | quality findings logged: {len(quality)}")
    print("Wrote clean_events.csv, clean_lineups.csv, identity maps, cleaning_drops.csv, data_quality_log.csv")


if __name__ == "__main__":
    main()
