"""Run from the repository root with:

    python src/features/build_rating_features.py
"""
from pathlib import Path
import sqlite3

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
PROJECT = HERE.parent
PROCESSED_DIR = PROJECT / "reports" / "processed"
ESD_PATH = PROJECT / "data" / "european_soccer_db" / "database.sqlite"

SQUAD_WINDOW_MONTHS = 12

XI_AGGREGATES = ["xi_overall_mean", "xi_overall_max", "xi_gk", "xi_def_mean",
                 "xi_mid_mean", "xi_att_mean", "xi_potential_mean",
                 "xi_age_mean", "xi_n_rated"]
SQUAD_AGGREGATES = ["squad_overall_top11", "squad_overall_top18", "squad_star",
                    "squad_gk_best", "squad_def_mean", "squad_mid_mean",
                    "squad_att_mean", "squad_age_mean", "squad_n_rated"]


def line_from_y(y):
    if y == 1:
        return "GK"
    if y <= 4:
        return "DEF"
    if y <= 8:
        return "MID"
    return "ATT"


def load_xi_long():
    """(esd_match_api_id, side, slot, player_api_id, line) per lineup slot."""
    slots = []
    for side in ["home", "away"]:
        ids = ", ".join(f"{side}_player_{i}" for i in range(1, 12))
        ys = ", ".join(f"{side}_player_Y{i}" for i in range(1, 12))
        with sqlite3.connect(ESD_PATH) as connection:
            frame = pd.read_sql(
                f"SELECT match_api_id, {ids}, {ys} FROM Match "
                f"WHERE {side}_player_1 IS NOT NULL", connection)
        long = pd.wide_to_long(
            frame,
            stubnames=[f"{side}_player_", f"{side}_player_Y"],
            i="match_api_id", j="slot").reset_index()
        long = long.rename(columns={f"{side}_player_": "player_api_id",
                                    f"{side}_player_Y": "y"})
        long["side"] = side
        slots.append(long[["match_api_id", "side", "slot",
                           "player_api_id", "y"]])
    xi = pd.concat(slots, ignore_index=True).dropna(subset=["player_api_id"])
    xi["player_api_id"] = xi["player_api_id"].astype(int)
    xi["line"] = xi["y"].map(
        lambda y: "MID" if pd.isna(y) else line_from_y(int(y)))
    return xi


def asof_ratings(long, ratings, date_column):
    """Attach the latest rating strictly before date_column, per player."""
    long = long.sort_values(date_column).reset_index(drop=True)
    ratings = ratings.sort_values("effective_date")
    merged = pd.merge_asof(
        long, ratings,
        left_on=date_column, right_on="effective_date",
        by="player_key", allow_exact_matches=False)
    assert ((merged["effective_date"] < merged[date_column])
            | merged["effective_date"].isna()).all(), \
        "as-of join leaked a same-day-or-later rating"
    return merged


def aggregate_side(group, prefix):
    rated = group.dropna(subset=["overall"])
    lines = rated.groupby("line")["overall"].mean()
    return {
        f"{prefix}_overall_mean": rated["overall"].mean(),
        f"{prefix}_overall_max": rated["overall"].max(),
        f"{prefix}_gk": lines.get("GK", np.nan),
        f"{prefix}_def_mean": lines.get("DEF", np.nan),
        f"{prefix}_mid_mean": lines.get("MID", np.nan),
        f"{prefix}_att_mean": lines.get("ATT", np.nan),
        f"{prefix}_potential_mean": rated["potential"].mean(),
        f"{prefix}_age_mean": rated["age"].mean(),
        f"{prefix}_n_rated": len(rated),
    }


def build_xi_features(store, xi, esd_ratings, player_names):
    lineup_rows = store.dropna(subset=["esd_match_api_id"]).copy()
    lineup_rows["esd_match_api_id"] = lineup_rows["esd_match_api_id"].astype(int)
    joined = xi.merge(
        lineup_rows[["match_id", "esd_match_api_id", "match_date", "era"]],
        left_on="match_api_id", right_on="esd_match_api_id", how="inner")
    joined["player_key"] = "esd:" + joined["player_api_id"].astype(str)
    joined = asof_ratings(joined, esd_ratings, "match_date")

    grouped = joined.groupby(["match_id", "side"])
    records = []
    for (match_id, side), group in grouped:
        row = aggregate_side(group, "xi")
        row["match_id"], row["side"] = match_id, side
        records.append(row)
    frame = pd.DataFrame(records)
    home = frame[frame["side"] == "home"].set_index("match_id").add_prefix("home_")
    away = frame[frame["side"] == "away"].set_index("match_id").add_prefix("away_")
    features = home.join(away, how="inner").drop(
        columns=["home_side", "away_side"])
    for name in XI_AGGREGATES:
        features[f"diff_{name}"] = (features[f"home_{name}"]
                                    - features[f"away_{name}"])
    display = joined.merge(player_names, on="player_api_id", how="left")
    return features.reset_index(), display


def esd_squad_snapshots(xi, store, esd_ratings):
    """Monthly (club, snapshot) squad aggregates from lineup appearances."""
    lineup_matches = store.dropna(subset=["esd_match_api_id"]).copy()
    lineup_matches["esd_match_api_id"] = \
        lineup_matches["esd_match_api_id"].astype(int)
    joined = xi.merge(
        lineup_matches[["esd_match_api_id", "match_date", "home_team_id",
                        "away_team_id"]],
        left_on="match_api_id", right_on="esd_match_api_id", how="inner")
    joined["club"] = np.where(joined["side"] == "home",
                              joined["home_team_id"], joined["away_team_id"])

    modal_line = (joined.groupby(["player_api_id", "line"]).size()
                  .rename("n").reset_index()
                  .sort_values("n", ascending=False)
                  .drop_duplicates("player_api_id")
                  .set_index("player_api_id")["line"])

    appearances = joined[["club", "player_api_id", "match_date"]].copy()
    appearances["month"] = appearances["match_date"].dt.to_period("M")
    appearances = appearances.drop_duplicates(
        ["club", "player_api_id", "month"])

    offsets = np.arange(1, SQUAD_WINDOW_MONTHS + 1)
    expanded = appearances.loc[
        appearances.index.repeat(len(offsets))].reset_index(drop=True)
    expanded["snapshot_month"] = (expanded["month"].to_numpy()
                                  + np.tile(offsets, len(appearances)))
    members = expanded.drop_duplicates(
        ["club", "player_api_id", "snapshot_month"])
    members = members.assign(
        snapshot_date=members["snapshot_month"].dt.to_timestamp(),
        player_key="esd:" + members["player_api_id"].astype(str),
        line=members["player_api_id"].map(modal_line).fillna("MID"))

    rated = asof_ratings(
        members[["club", "player_key", "line", "snapshot_date"]],
        esd_ratings, "snapshot_date")
    return aggregate_squads(rated, "snapshot_date")


def sofifa_squad_snapshots(sofifa_ratings):
    frame = sofifa_ratings.dropna(subset=["club_team_id"]).copy()
    frame["club"] = frame["club_team_id"].astype(int)
    frame["line"] = frame["position_bucket"].fillna("MID")
    frame = frame.rename(columns={"effective_date": "snapshot_date"})
    return aggregate_squads(frame, "snapshot_date")


def aggregate_squads(rated, date_column):
    rated = rated.dropna(subset=["overall"])
    records = []
    for (club, snapshot), group in rated.groupby(["club", date_column]):
        overall = group["overall"].sort_values(ascending=False)
        lines = group.groupby("line")["overall"].mean()
        gk = group[group["line"] == "GK"]["overall"]
        records.append({
            "club": club, "snapshot_date": snapshot,
            "squad_overall_top11": overall.head(11).mean(),
            "squad_overall_top18": overall.head(18).mean(),
            "squad_star": overall.max(),
            "squad_gk_best": gk.max() if len(gk) else np.nan,
            "squad_def_mean": lines.get("DEF", np.nan),
            "squad_mid_mean": lines.get("MID", np.nan),
            "squad_att_mean": lines.get("ATT", np.nan),
            "squad_age_mean": group["age"].mean(),
            "squad_n_rated": len(group),
        })
    return pd.DataFrame(records)


def build_squad_features(store, squad_snapshots):
    snapshots = squad_snapshots.sort_values("snapshot_date")
    sides = []
    for side in ["home", "away"]:
        frame = store[["match_id", "match_date", f"{side}_team_id"]].rename(
            columns={f"{side}_team_id": "club"}).sort_values("match_date")
        merged = pd.merge_asof(frame, snapshots,
                               left_on="match_date", right_on="snapshot_date",
                               by="club", allow_exact_matches=False)
        assert ((merged["snapshot_date"] < merged["match_date"])
                | merged["snapshot_date"].isna()).all()
        merged = merged.set_index("match_id")[SQUAD_AGGREGATES]
        sides.append(merged.add_prefix(f"{side}_"))
    features = sides[0].join(sides[1], how="outer")
    for name in SQUAD_AGGREGATES:
        features[f"diff_{name}"] = (features[f"home_{name}"]
                                    - features[f"away_{name}"])
    return features.reset_index()


def build_lineup_display(store, splits, xi_display, sofifa_ratings):
    """Real XIs (StatsBomb era) + strongest sofifa XI (test-era fixtures)."""
    rows = []
    sb = xi_display[xi_display["era"] == "statsbomb"]
    for r in sb.itertuples():
        rows.append({"match_id": r.match_id, "side": r.side, "slot": r.slot,
                     "player_name": r.player_name, "position_bucket": r.line,
                     "overall": r.overall, "age": r.age, "kind": "actual_xi"})

    test_ids = set(splits.loc[splits["split"] == "test", "match_id"])
    fixtures = store[store["match_id"].isin(test_ids)
                     & (store["era"] == "football_data")]
    roster = sofifa_ratings.dropna(subset=["club_team_id"]).copy()
    roster["club"] = roster["club_team_id"].astype(int)
    latest = (roster.sort_values("effective_date")
              .groupby(["club", "player_key"]).tail(1))
    club_latest = latest.groupby("club")["effective_date"].transform("max")
    latest = latest[latest["effective_date"]
                    >= club_latest - pd.Timedelta(days=400)]
    top11 = (latest.sort_values("overall", ascending=False)
             .groupby("club").head(11))
    by_club = dict(tuple(top11.groupby("club")))
    for r in fixtures.itertuples():
        for side, club in [("home", r.home_team_id), ("away", r.away_team_id)]:
            squad = by_club.get(club)
            if squad is None:
                continue
            for slot, p in enumerate(squad.itertuples(), start=1):
                rows.append({"match_id": r.match_id, "side": side,
                             "slot": slot, "player_name": p.player_name,
                             "position_bucket": p.position_bucket,
                             "overall": p.overall, "age": p.age,
                             "kind": "expected_squad"})
    return pd.DataFrame(rows)


def main():
    store = pd.read_csv(PROCESSED_DIR / "extended_match_store.csv",
                        encoding="utf-8", parse_dates=["match_date"])
    splits = pd.read_csv(PROCESSED_DIR / "temporal_match_splits_extended.csv",
                         encoding="utf-8", usecols=["match_id", "split"])
    ratings = pd.read_csv(PROCESSED_DIR / "player_ratings.csv",
                          encoding="utf-8", parse_dates=["effective_date"])
    esd_ratings = ratings[ratings["source"] == "esd"][
        ["player_key", "effective_date", "overall", "potential", "age"]]
    sofifa_ratings = ratings[ratings["source"] == "sofifa"]

    with sqlite3.connect(ESD_PATH) as connection:
        player_names = pd.read_sql(
            "SELECT player_api_id, player_name FROM Player", connection)
    xi = load_xi_long()

    print("XI features...")
    xi_features, xi_display = build_xi_features(store, xi, esd_ratings,
                                                player_names)
    print(f"  {len(xi_features)} matches with XI aggregates")

    print("Squad snapshots (ESD lineups, monthly)...")
    esd_snapshots = esd_squad_snapshots(xi, store, esd_ratings)
    print(f"  {len(esd_snapshots)} club-month snapshots")
    if len(sofifa_ratings):
        print("Squad snapshots (sofifa rosters)...")
        sofifa_snapshots = sofifa_squad_snapshots(sofifa_ratings)
        print(f"  {len(sofifa_snapshots)} club-update snapshots")
        snapshots = pd.concat([esd_snapshots, sofifa_snapshots],
                              ignore_index=True)
    else:
        print("No sofifa ratings present; squad features stop at 2016.")
        snapshots = esd_snapshots
    squad_features = build_squad_features(store, snapshots)

    features = store[["match_id"]].merge(
        xi_features, on="match_id", how="left").merge(
        squad_features, on="match_id", how="left")
    features.insert(1, "xi_available",
                    features["home_xi_overall_mean"].notna().astype(int))

    xi_cov = features["xi_available"].mean()
    squad_cov = features["home_squad_overall_top11"].notna().mean()
    era = store.set_index("match_id")["era"]
    big5 = store[store["league"].isin(
        {"Premier League", "La Liga", "Serie A", "Ligue 1", "Bundesliga"})]
    big5_esd = big5[big5["era"].isin({"esd", "statsbomb"})]["match_id"]
    big5_xi = features.set_index("match_id").loc[big5_esd, "xi_available"].mean()
    assert big5_xi >= 0.90, f"big-5 XI coverage only {big5_xi:.1%}"
    print(f"Coverage: XI {xi_cov:.1%} of all matches (big-5 ESD-era "
          f"{big5_xi:.1%}), squads {squad_cov:.1%}")

    output = PROCESSED_DIR / "rating_features.csv"
    features.to_csv(output, index=False, encoding="utf-8")
    print(f"Wrote {output} ({features.shape[1]} columns)")

    display = build_lineup_display(store, splits, xi_display, sofifa_ratings)
    display_path = PROCESSED_DIR / "lineup_display.csv"
    display.to_csv(display_path, index=False, encoding="utf-8")
    print(f"Wrote {display_path} ({len(display)} rows)")


if __name__ == "__main__":
    main()
