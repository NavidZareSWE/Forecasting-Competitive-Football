import os
import unicodedata
from pathlib import Path

import pandas as pd

# --- Configuration ---
HERE = Path(__file__).resolve().parent
PROJECT = HERE.parent
def resolve_case_insensitive_folder(path):
    path = Path(path)
    if path.exists():
        return path
    if path.parent.is_dir():
        for candidate in path.parent.iterdir():
            if candidate.is_dir() and candidate.name.lower() == path.name.lower():
                return candidate
    return path


FOOTBALL_DATA_DIR = resolve_case_insensitive_folder(
    os.environ.get("FOOTBALL_DATA", PROJECT / "data" / "Football_Data"))
PROCESSED_DIR = PROJECT / "reports" / "processed"
PROCESSED_DIR.mkdir(parents=True, exist_ok=True)

DIV_TO_LEAGUE = {"SP1": "La Liga", "E0": "Premier League", "I1": "Serie A", "F1": "Ligue 1"}
DATE_FORMATS = {"E0": "%d/%m/%Y"}
DEFAULT_DATE_FORMAT = "%d/%m/%y"

MANUAL_OVERRIDES = {"Ath Madrid": "atletico madrid"}


def strip_accents(text: str) -> str:
    return "".join(c for c in unicodedata.normalize("NFKD", str(text)) if not unicodedata.combining(c))


def normalize(name: str) -> str:
    return strip_accents(name).lower().replace("-", " ").strip()


def levenshtein(a: str, b: str) -> int:
    if len(a) < len(b):
        a, b = b, a
    if not b:
        return len(a)
    previous = list(range(len(b) + 1))
    for i, ca in enumerate(a, 1):
        current = [i]
        for j, cb in enumerate(b, 1):
            current.append(min(previous[j] + 1, current[j - 1] + 1, previous[j - 1] + (ca != cb)))
        previous = current
    return previous[-1]


def token_similarity(fd_name: str, sb_name: str):
    fd_tokens, sb_tokens = normalize(fd_name).split(), normalize(sb_name).split()
    matched, distance = 0, 0
    for a in fd_tokens:
        best = None
        for b in sb_tokens:
            if a == b or b.startswith(a) or a.startswith(b):
                cost = 0
            elif levenshtein(a, b) <= 1:
                cost = levenshtein(a, b)
            else:
                continue
            best = cost if best is None else min(best, cost)
        if best is not None:
            matched += 1
            distance += best
    return matched / len(fd_tokens), distance


def resolve_team(name, sb_names):
    by_norm = {normalize(n): n for n in sb_names}
    if name in MANUAL_OVERRIDES:
        return by_norm[MANUAL_OVERRIDES[name]]
    if normalize(name) in by_norm:
        return by_norm[normalize(name)]
    ranked = sorted(sb_names, key=lambda sb: (-token_similarity(name, sb)[0],
                                              token_similarity(name, sb)[1],
                                              levenshtein(normalize(name), normalize(sb))))
    return ranked[0]


def match_method(name, sb_names):
    if name in MANUAL_OVERRIDES:
        return "override"
    if normalize(name) in {normalize(n) for n in sb_names}:
        return "exact"
    return "fuzzy"


def find_football_data_files():
    files = {}
    for path in sorted(FOOTBALL_DATA_DIR.glob("*.csv")):
        div = pd.read_csv(path, encoding="utf-8", nrows=1)["Div"].iloc[0]
        if div in DIV_TO_LEAGUE:
            files[div] = path
    return files


def clean_football_data(div, path):
    df = pd.read_csv(path, encoding="utf-8")
    df = df.dropna(subset=["HomeTeam", "AwayTeam"]).copy()
    date_format = DATE_FORMATS.get(div, DEFAULT_DATE_FORMAT)
    df["fd_date"] = pd.to_datetime(df["Date"], format=date_format)
    df["competition_name"] = DIV_TO_LEAGUE[div]
    return df[["competition_name", "fd_date", "HomeTeam", "AwayTeam", "B365H", "B365D", "B365A"]]


def de_vig(row):
    odds = [row["B365H"], row["B365D"], row["B365A"]]
    if any(pd.isna(o) for o in odds):
        return pd.Series({"p_home": None, "p_draw": None, "p_away": None, "overround": None})
    implied = [1.0 / o for o in odds]
    overround = sum(implied)
    return pd.Series({
        "p_home": implied[0] / overround,
        "p_draw": implied[1] / overround,
        "p_away": implied[2] / overround,
        "overround": overround,
    })


def main():
    store = pd.read_csv(PROCESSED_DIR / "match_store.csv", encoding="utf-8", parse_dates=["match_date"])

    tagged_frames, failure_frames, alias_rows, coverage_rows = [], [], [], []

    for div, path in find_football_data_files().items():
        league = DIV_TO_LEAGUE[div]
        fd = clean_football_data(div, path)
        sb = store[store["competition_name"] == league]
        sb_names = sorted(set(sb["home_team"]) | set(sb["away_team"]))

        fd["home_sb"] = fd["HomeTeam"].map(lambda n: resolve_team(n, sb_names))
        fd["away_sb"] = fd["AwayTeam"].map(lambda n: resolve_team(n, sb_names))

        for fd_name in sorted(set(fd["HomeTeam"]) | set(fd["AwayTeam"])):
            alias_rows.append({"league": league, "fd_name": fd_name,
                               "sb_name": resolve_team(fd_name, sb_names),
                               "method": match_method(fd_name, sb_names)})
        mapped_teams = {resolve_team(n, sb_names) for n in set(fd["HomeTeam"]) | set(fd["AwayTeam"])}
        assert mapped_teams == set(sb_names), f"{league}: alias map is not a bijection onto StatsBomb teams"

        # --- Tag on identity only (home_sb, away_sb are unique per league-season) ---
        merged = fd.merge(
            sb[["match_id", "match_date", "home_team", "away_team"]],
            left_on=["home_sb", "away_sb"], right_on=["home_team", "away_team"], how="left",
        )
        merged["date_matches"] = merged["fd_date"] == merged["match_date"]

        matched = merged[merged["match_id"].notna()].copy()
        unmatched = merged[merged["match_id"].isna()].copy()

        # --- De-vig ---
        matched = pd.concat([matched, matched.apply(de_vig, axis=1)], axis=1)

        tagged_frames.append(matched)
        if len(unmatched):
            unmatched["reason"] = "no StatsBomb match for this fixture (absent from open-data release)"
            failure_frames.append(
                unmatched[["competition_name", "fd_date", "HomeTeam", "AwayTeam", "home_sb", "away_sb", "reason"]]
            )
        missing_odds = matched[matched["p_home"].isna()].copy()
        if len(missing_odds):
            missing_odds["reason"] = "matched but missing Bet365 odds"
            failure_frames.append(
                missing_odds[["competition_name", "fd_date", "HomeTeam", "AwayTeam", "home_sb", "away_sb", "reason"]]
            )

        coverage_rows.append({
            "league": league,
            "fd_matches": len(fd),
            "tagged": len(matched),
            "tagged_with_odds": int(matched["p_home"].notna().sum()),
            "coverage": round(matched["p_home"].notna().sum() / len(fd), 4),
            "date_mismatches": int((~matched["date_matches"]).sum()),
        })

    baseline = pd.concat(tagged_frames, ignore_index=True)
    baseline = baseline[baseline["p_home"].notna()].copy()
    baseline["match_id"] = baseline["match_id"].astype(int)
    baseline_out = baseline[[
        "match_id", "competition_name", "match_date", "home_team", "away_team",
        "B365H", "B365D", "B365A", "p_home", "p_draw", "p_away", "overround",
    ]]

    coverage = pd.DataFrame(coverage_rows)
    alias_map = pd.DataFrame(alias_rows)
    failures = pd.concat(failure_frames, ignore_index=True) if failure_frames else pd.DataFrame()

    baseline_out.to_csv(PROCESSED_DIR / "market_baseline.csv", index=False, encoding="utf-8")
    alias_map.to_csv(PROCESSED_DIR / "alias_map.csv", index=False, encoding="utf-8")
    coverage.to_csv(PROCESSED_DIR / "odds_coverage.csv", index=False, encoding="utf-8")
    failures.to_csv(PROCESSED_DIR / "odds_failures.csv", index=False, encoding="utf-8")

    # --- Self-checks ---
    prob_sums = baseline_out[["p_home", "p_draw", "p_away"]].sum(axis=1)
    assert (prob_sums.sub(1.0).abs() < 1e-9).all(), "De-vigged probabilities do not sum to 1!"
    assert (baseline_out["overround"] > 1.0).all(), "Over-round should exceed 1 before de-vigging!"

    print(coverage.to_string(index=False))
    print(f"\nBaseline rows (tagged + de-vigged): {len(baseline_out)}")
    print(f"Alias entries: {len(alias_map)} | Failures logged: {len(failures)}")
    print("Wrote: market_baseline.csv, alias_map.csv, odds_coverage.csv, odds_failures.csv")


if __name__ == "__main__":
    main()
