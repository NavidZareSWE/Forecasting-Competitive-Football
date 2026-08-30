"""Run from the repository root with:

    python src/pipeline/build_extended_splits.py
"""
from pathlib import Path

import pandas as pd

HERE = Path(__file__).resolve().parent
PROJECT = HERE.parent
PROCESSED_DIR = PROJECT / "reports" / "processed"

# Season-boundary split. The shape is fixed - the two most recent complete
# seasons are the test set, the two before them are validation, and
# everything earlier trains - so the contract stays "predict forward,
# never backward".
#
# The four constants are pinned rather than derived from whatever seasons
# happen to sit on disk. Downloading one more season must not silently
# reshape train/validation/test underneath a finished set of results:
# every number in the report would move and nothing would say so. To roll
# forward, edit all four together and re-run the whole model layer.
TRAIN_SEASONS_THROUGH = "2020/2021"
VALIDATION_SEASONS = {"2021/2022", "2022/2023"}
TEST_SEASONS = {"2023/2024", "2024/2025"}
OUT_OF_SCOPE_SEASONS = {"2025/2026"}


def season_split(season):
    if season in OUT_OF_SCOPE_SEASONS:
        return "out_of_scope"
    if season in TEST_SEASONS:
        return "test"
    if season in VALIDATION_SEASONS:
        return "validation"
    assert season <= TRAIN_SEASONS_THROUGH, (
        f"season {season} is newer than the pinned train boundary and is "
        f"listed in none of VALIDATION_SEASONS, TEST_SEASONS or "
        f"OUT_OF_SCOPE_SEASONS. Roll the split constants forward "
        f"deliberately and re-run the model layer.")
    return "train"


def main():
    store = pd.read_csv(PROCESSED_DIR / "extended_match_store.csv",
                        encoding="utf-8", parse_dates=["match_date"])
    inplay = pd.read_csv(PROCESSED_DIR / "temporal_match_splits.csv",
                         encoding="utf-8")
    holdout_ids = set(
        inplay.loc[inplay["split"].isin({"validation", "test"}), "match_id"])

    frame = store.copy()
    frame["split"] = frame["season"].map(season_split)
    excluded_mask = frame["match_id"].isin(holdout_ids)
    frame.loc[excluded_mask, "split"] = "excluded"
    frame = frame.rename(columns={"result": "label_result",
                                  "margin": "label_margin"})

    assert frame["match_id"].is_unique
    assert set(frame.loc[excluded_mask, "match_id"]) == holdout_ids, \
        "in-play holdout ids missing from the extended store"
    assert frame.loc[frame["split"] == "excluded", "era"].eq("statsbomb").all()
    train_max = frame.loc[frame["split"] == "train", "match_date"].max()
    val_dates = frame.loc[frame["split"] == "validation", "match_date"]
    test_min = frame.loc[frame["split"] == "test", "match_date"].min()
    assert train_max < val_dates.min(), "train dates overlap validation"
    assert val_dates.max() < test_min, "validation dates overlap test"

    output = PROCESSED_DIR / "temporal_match_splits_extended.csv"
    frame.to_csv(output, index=False, encoding="utf-8")
    print(frame.groupby("split").size().reindex(
        ["train", "validation", "test", "excluded",
         "out_of_scope"]).to_string())
    print(f"train through {train_max.date()}, validation "
          f"{val_dates.min().date()}..{val_dates.max().date()}, "
          f"test from {test_min.date()}")
    print(f"Wrote {output}")


if __name__ == "__main__":
    main()
