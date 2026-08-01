"""Task C models re-scored against the de-vigged market on the identical
odds-tagged test subset, with the tagging coverage rate.

    python src/models/market_comparison.py
"""

from pathlib import Path

import numpy as np
import pandas as pd

from modeling_common import (CLASS_ORDER, RESULTS_DIR, classification_metrics,
                             per_class_metrics)


PROJECT = Path(__file__).resolve().parent.parent
PROCESSED_DIR = PROJECT / "reports" / "processed"

# H/D/A and home/draw/away are the same three columns.
MARKET_COLUMNS = {"H": "p_home", "D": "p_draw", "A": "p_away"}


def load_predictions(task="C"):
    path = RESULTS_DIR / f"predictions_{task}.csv"
    if not path.exists():
        raise FileNotFoundError(f"Missing {path}. Run run_models.py first.")
    return pd.read_csv(path, encoding="utf-8")


def load_market():
    path = PROCESSED_DIR / "market_baseline.csv"
    if not path.exists():
        raise FileNotFoundError(f"Missing {path}. Run build_market_baseline.py first.")
    market = pd.read_csv(path, encoding="utf-8")
    probabilities = market[[MARKET_COLUMNS[c] for c in CLASS_ORDER]].to_numpy()

    # De-vigging contract re-asserted at the point of use.
    assert np.all(market["overround"] > 1.0), \
        "market rows with overround <= 1 found; de-vigging is not meaningful"
    assert np.allclose(probabilities.sum(axis=1), 1.0, atol=1e-9), \
        "de-vigged market probabilities do not sum to 1"
    return market, probabilities


def main():
    predictions = load_predictions("C")
    market, market_probabilities = load_market()
    market = market.assign(**{f"m_{c}": market_probabilities[:, i]
                              for i, c in enumerate(CLASS_ORDER)})

    test_matches = predictions["match_id"].unique()
    tagged = market[market["match_id"].isin(test_matches)]
    tagged_ids = set(tagged["match_id"])

    coverage = len(tagged_ids) / len(test_matches)
    print(f"Test matches:            {len(test_matches)}")
    print(f"Tagged with odds:        {len(tagged_ids)}")
    print(f"Tagging coverage:        {coverage:.4f}")
    if coverage < 1.0:
        print(f"Untagged and therefore excluded from this comparison: "
              f"{len(test_matches) - len(tagged_ids)} matches")

    per_league = (market[market["match_id"].isin(test_matches)]
                  .groupby("competition_name")["match_id"].nunique())
    test_per_league = (predictions.drop_duplicates("match_id")
                       .groupby("competition_name")["match_id"].nunique()
                       if "competition_name" in predictions.columns
                       else pd.Series(dtype=int))
    coverage_rows = []
    for league, total in test_per_league.items():
        covered = int(per_league.get(league, 0))
        coverage_rows.append({"competition_name": league,
                              "test_matches": int(total),
                              "tagged_with_odds": covered,
                              "coverage": round(covered / total, 4)})
    coverage_rows.append({"competition_name": "ALL",
                          "test_matches": len(test_matches),
                          "tagged_with_odds": len(tagged_ids),
                          "coverage": round(coverage, 4)})
    coverage_frame = pd.DataFrame(coverage_rows)

    rows = []
    market_lookup = tagged.set_index("match_id")[[f"m_{c}" for c in CLASS_ORDER]]
    for model_name, group in predictions.groupby("model"):
        subset = group[group["match_id"].isin(tagged_ids)]
        if subset.empty:
            continue
        proba = subset[[f"p_{c}" for c in CLASS_ORDER]].to_numpy()
        y_true = subset["y_true"].to_numpy()
        row = {"model": model_name, "n_matches": len(subset),
               **{k: round(v, 5)
                  for k, v in classification_metrics(proba, y_true).items()}}
        row.update(per_class_metrics(proba, y_true))
        rows.append(row)

    reference = predictions[predictions["model"] == predictions["model"].iloc[0]]
    reference = reference[reference["match_id"].isin(tagged_ids)]
    aligned = market_lookup.loc[reference["match_id"]].to_numpy()
    market_row = {"model": "MARKET_devigged", "n_matches": len(reference),
                  **{k: round(v, 5)
                     for k, v in classification_metrics(
                         aligned, reference["y_true"].to_numpy()).items()}}
    market_row.update(per_class_metrics(aligned, reference["y_true"].to_numpy()))
    rows.append(market_row)

    results = pd.DataFrame(rows).sort_values("rps").reset_index(drop=True)
    market_rps = float(results.loc[results["model"] == "MARKET_devigged", "rps"].iloc[0])
    results["rps_gap_vs_market"] = (results["rps"] - market_rps).round(5)
    results["beats_market"] = results["rps"] < market_rps
    results["tagging_coverage"] = round(coverage, 4)

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    results.to_csv(RESULTS_DIR / "market_comparison.csv", index=False,
                   encoding="utf-8")
    coverage_frame.to_csv(RESULTS_DIR / "market_coverage.csv", index=False,
                          encoding="utf-8")

    print("\nRPS on the tagged test matches (lower is better):")
    print(results[["model", "n_matches", "rps", "log_loss", "brier", "ece",
                   "rps_gap_vs_market", "beats_market"]].to_string(index=False))
    beaters = results[results["beats_market"] & (results["model"] != "MARKET_devigged")]
    print(f"\nModels beating the de-vigged market: {len(beaters)} of "
          f"{len(results) - 1}")
    print(f"Wrote -> {RESULTS_DIR / 'market_comparison.csv'}")
    print(f"Wrote -> {RESULTS_DIR / 'market_coverage.csv'}")


if __name__ == "__main__":
    main()
