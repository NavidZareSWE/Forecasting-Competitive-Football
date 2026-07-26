from pathlib import Path

import pandas as pd


HERE = Path(__file__).resolve().parent
PROJECT = HERE.parent
PROCESSED_DIR = PROJECT / "reports" / "processed"

SPLIT_RATIOS = {"train": 0.60, "validation": 0.20, "test": 0.20}
SNAPSHOT_MINUTES = list(range(0, 91, 5))
REQUIRED_COLUMNS = {
    "match_id", "competition_id", "competition_name", "season_id", "match_date",
    "kick_off", "home_team", "away_team", "label_result", "label_margin",
}


def date_split_sizes(n_dates):
    assert n_dates >= 3, "Need at least three match dates for train/validation/test."
    n_train = max(1, int(n_dates * SPLIT_RATIOS["train"]))
    n_validation = max(1, int(n_dates * SPLIT_RATIOS["validation"]))
    if n_train + n_validation >= n_dates:
        n_train, n_validation = n_dates - 2, 1
    return n_train, n_validation


def build_match_splits(targets):
    missing = REQUIRED_COLUMNS - set(targets.columns)
    assert not missing, f"model_targets is missing required columns: {sorted(missing)}"
    assert targets["match_id"].is_unique, "A parent match may have only one split assignment!"

    table = targets.copy()
    table["match_date"] = pd.to_datetime(table["match_date"], errors="coerce")
    assert table["match_date"].notna().all(), "Every match needs a valid match_date!"

    # A whole date is assigned to one split. This prevents same-day fixtures
    # from crossing a temporal boundary when kick-off times are incomplete.
    dates = pd.DataFrame({"match_date": sorted(table["match_date"].unique())})
    n_train, n_validation = date_split_sizes(len(dates))
    dates["split"] = "test"
    dates.loc[:n_train - 1, "split"] = "train"
    dates.loc[n_train:n_train + n_validation - 1, "split"] = "validation"

    splits = table.merge(dates, on="match_date", how="left", validate="many_to_one")
    splits = splits.sort_values(["match_date", "match_id"]).reset_index(drop=True)

    assert splits["split"].isin(SPLIT_RATIOS).all(), "Invalid split name!"
    assert splits["split"].notna().all(), "Unassigned match!"
    assert splits.groupby("match_id").size().eq(1).all(), "Duplicate match split!"
    assert splits.loc[splits["split"] == "train", "match_date"].max() < splits.loc[
        splits["split"] == "validation", "match_date"
    ].min(), "Train and validation dates overlap!"
    assert splits.loc[splits["split"] == "validation", "match_date"].max() < splits.loc[
        splits["split"] == "test", "match_date"
    ].min(), "Validation and test dates overlap!"

    return splits


def build_snapshot_plan(match_splits):
    snapshots = match_splits.loc[match_splits.index.repeat(len(SNAPSHOT_MINUTES))].copy()
    snapshots["snapshot_minute"] = SNAPSHOT_MINUTES * len(match_splits)
    snapshots = snapshots.sort_values(["match_date", "match_id", "snapshot_minute"]).reset_index(drop=True)

    expected_rows = len(match_splits) * len(SNAPSHOT_MINUTES)
    assert len(snapshots) == expected_rows, "Unexpected snapshot count!"
    assert snapshots.groupby("match_id")["split"].nunique().eq(1).all(), (
        "Snapshots from one match crossed a split boundary!"
    )
    assert snapshots.groupby("match_id")["snapshot_minute"].nunique().eq(
        len(SNAPSHOT_MINUTES)
    ).all(), "A match is missing a scheduled snapshot!"
    return snapshots


def main():
    source_path = PROCESSED_DIR / "model_targets.csv"
    if not source_path.exists():
        raise FileNotFoundError(
            f"Missing {source_path}. Run build_label_store.py before build_temporal_splits.py."
        )

    targets = pd.read_csv(source_path, encoding="utf-8")
    match_splits = build_match_splits(targets)
    snapshot_plan = build_snapshot_plan(match_splits)

    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    match_splits.to_csv(PROCESSED_DIR / "temporal_match_splits.csv", index=False, encoding="utf-8")
    snapshot_plan.to_csv(PROCESSED_DIR / "snapshot_split_plan.csv", index=False, encoding="utf-8")

    print("Chronological split by whole match date (60% train / 20% validation / 20% test)")
    print(match_splits.groupby("split").agg(
        matches=("match_id", "size"), first_date=("match_date", "min"), last_date=("match_date", "max")
    ).reindex(["train", "validation", "test"]).to_string())
    print(f"Snapshot plan: {len(snapshot_plan)} rows at minutes {SNAPSHOT_MINUTES}")
    print("Wrote temporal_match_splits.csv and snapshot_split_plan.csv")


if __name__ == "__main__":
    main()
