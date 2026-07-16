from pathlib import Path
import pandas as pd


HERE = Path(__file__).resolve().parent
PROJECT = HERE.parent
PROCESSED_DIR = PROJECT / "reports" / "processed"

REQUIRED_COLUMNS = {
    "match_id", "competition_id", "competition_name", "season_id",
    "match_date", "kick_off", "home_team", "away_team",
    "home_score", "away_score", "result", "margin",
}


def build_label_store(match_store):
    missing = REQUIRED_COLUMNS - set(match_store.columns)
    assert not missing, f"match_store is missing required columns: {sorted(missing)}"
    assert match_store["match_id"].is_unique, "Each target must have one match_id!"
    assert match_store[["home_score", "away_score"]].notna().all().all(), "Missing final score!"

    expected_result = pd.Series("D", index=match_store.index)
    expected_result = expected_result.mask(match_store["home_score"] > match_store["away_score"], "H")
    expected_result = expected_result.mask(match_store["home_score"] < match_store["away_score"], "A")
    expected_margin = (match_store["home_score"] - match_store["away_score"]).clip(-5, 5)

    assert match_store["result"].equals(expected_result), "Result label disagrees with final score!"
    assert match_store["margin"].equals(expected_margin), "Margin label disagrees with final score!"

    labels = match_store[[
        "match_id", "competition_id", "competition_name", "season_id",
        "match_date", "kick_off", "home_team", "away_team",
    ]].copy()
    # These are targets only. Feature builders must not read them as predictors.
    labels["label_result"] = expected_result
    labels["label_margin"] = expected_margin

    assert labels["label_result"].isin(["H", "D", "A"]).all(), "Invalid outcome label!"
    assert labels["label_margin"].between(-5, 5).all(), "Margin label outside [-5, 5]!"
    return labels.sort_values(["match_date", "match_id"]).reset_index(drop=True)


def main():
    source_path = PROCESSED_DIR / "match_store.csv"
    if not source_path.exists():
        raise FileNotFoundError(
            f"Missing {source_path}. Run build_match_store.py before build_label_store.py."
        )

    match_store = pd.read_csv(source_path, parse_dates=["match_date"], encoding="utf-8")
    labels = build_label_store(match_store)

    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    output_path = PROCESSED_DIR / "model_targets.csv"
    labels.to_csv(output_path, index=False, encoding="utf-8")
    print(f"Wrote {len(labels)} canonical targets -> {output_path}")
    print("Task C/L label_result: H (home win), D (draw), A (away win)")
    print("Task R/L label_margin: clipped home_score - away_score in [-5, 5]")


if __name__ == "__main__":
    main()
