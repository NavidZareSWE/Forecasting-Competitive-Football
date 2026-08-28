"""Run from the repository root with:

    python src/pipeline/tune_ratings.py
"""
from pathlib import Path
import json
import sys

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

import build_team_ratings as ratings_module  # noqa: E402

PROJECT = HERE.parent
PROCESSED_DIR = PROJECT / "reports" / "processed"

ELO_K_GRID = [10.0, 20.0, 32.0]
ELO_HOME_ADV_GRID = [50.0, 70.0, 90.0]
PI_LAMBDA_GRID = [0.025, 0.035, 0.06, 0.09]
CLASS_ORDER = ["H", "D", "A"]
PROBE_COLUMNS = ["elo_diff", "elo_expected_home", "pi_expected_gd",
                 "pi_home_home_pre", "pi_home_away_pre",
                 "pi_away_home_pre", "pi_away_away_pre"]


def ranked_probability_score(proba, y_true):
    observed = np.zeros_like(proba)
    index = {c: i for i, c in enumerate(CLASS_ORDER)}
    for row, label in enumerate(y_true):
        observed[row, index[label]] = 1.0
    cum_p = np.cumsum(proba, axis=1)
    cum_o = np.cumsum(observed, axis=1)
    return float((((cum_p - cum_o) ** 2)[:, :-1].sum(axis=1) / 2).mean())


def candidate_ratings(store, k, home_adv, lam):
    ratings_module.ELO_K = k
    ratings_module.ELO_HOME_ADV = home_adv
    ratings_module.PI_LAMBDA = lam
    rows = []
    for _, league_matches in store.groupby("league"):
        ordered = league_matches.sort_values(["match_date", "match_id"])
        rows.extend(ratings_module.run_league(ordered))
    return pd.DataFrame(rows)


def score_candidate(store, splits, k, home_adv, lam):
    frame = candidate_ratings(store, k, home_adv, lam).merge(
        splits, on="match_id")
    train = frame[frame["split"] == "train"]
    validation = frame[frame["split"] == "validation"]
    probe = LogisticRegression(max_iter=2000, C=1.0)
    probe.fit(train[PROBE_COLUMNS], train["label_result"])
    proba = probe.predict_proba(validation[PROBE_COLUMNS])
    aligned = np.zeros((len(validation), 3))
    classes = list(probe.classes_)
    for j, label in enumerate(CLASS_ORDER):
        aligned[:, j] = proba[:, classes.index(label)]
    return ranked_probability_score(aligned,
                                    validation["label_result"].to_numpy())


def main():
    store = pd.read_csv(PROCESSED_DIR / "extended_match_store.csv",
                        encoding="utf-8", parse_dates=["match_date"])
    splits = pd.read_csv(PROCESSED_DIR / "temporal_match_splits_extended.csv",
                         encoding="utf-8",
                         usecols=["match_id", "split", "label_result"])

    results = []
    for k in ELO_K_GRID:
        for home_adv in ELO_HOME_ADV_GRID:
            for lam in PI_LAMBDA_GRID:
                rps = score_candidate(store, splits, k, home_adv, lam)
                results.append({"elo_k": k, "elo_home_adv": home_adv,
                                "pi_lambda": lam, "validation_rps": rps})
                print(f"K={k:5.1f} HA={home_adv:5.1f} lambda={lam:.3f} "
                      f"-> validation RPS {rps:.5f}")

    frame = pd.DataFrame(results).sort_values("validation_rps")
    frame.to_csv(PROCESSED_DIR / "rating_tuning.csv", index=False,
                 encoding="utf-8")
    best = frame.iloc[0]
    params = {"ELO_K": float(best["elo_k"]),
              "ELO_HOME_ADV": float(best["elo_home_adv"]),
              "PI_LAMBDA": float(best["pi_lambda"]),
              "validation_rps": float(best["validation_rps"])}
    with open(PROCESSED_DIR / "rating_params.json", "w",
              encoding="utf-8") as sink:
        json.dump(params, sink, indent=2)
    print(f"Best: {params}")
    print(f"Wrote rating_tuning.csv, rating_params.json")


if __name__ == "__main__":
    main()
