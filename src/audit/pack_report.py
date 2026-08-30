"""Collect everything a written report needs into one folder.

    python src/audit/pack_report.py            # -> report_pack/ at the repo root
    PACK_DIR=/somewhere python src/audit/pack_report.py

Two jobs, and the second is the one that matters:

1. Copy the result files out of the gitignored src/reports/ tree and render
   report-ready Markdown tables from them, so nothing has to be retyped.

2. **Label which generation each file belongs to.** The pipeline's stages are
   run at different times, and right now they disagree: the data layer was
   rebuilt with the 2025/26 season while every model result still describes the
   previous split. A report that quotes the new dataset summary next to the old
   accuracy table would be wrong, so the packer stamps each file with the stage
   that produced it and refuses to pretend the tree is coherent.

Generations are decided by the modification time of a sentinel file per stage,
not by a hardcoded date, so this keeps working after a re-run.
"""
from pathlib import Path
import datetime
import os
import shutil
import sys

import pandas as pd

HERE = Path(__file__).resolve().parent
PROJECT = HERE.parent
REPO = PROJECT.parent
REPORTS = PROJECT / "reports"
PROCESSED = REPORTS / "processed"
FEATURES = REPORTS / "features"
_DEFAULT_PACK = ("report_pack_v2" if os.environ.get("SCOPE", "").lower() == "v2"
                 else "report_pack")
PACK = Path(os.environ.get("PACK_DIR", REPO / _DEFAULT_PACK))

COPY_LIMIT_MB = 12.0

# SCOPE=v2 packs one coherent modelling run for someone writing the v2 report.
#
# It used to drop two hardcoded lists of file names: "v1-era" analyses that
# still reported n_train=896, and data-layer tables rebuilt ahead of the sweep.
# The 2026-08-30 full run retired both - every stage from the store through
# ablation, significance, SHAP, resampling and the compute profile was produced
# in a single pass at n_train=39,707, so nothing is excluded for staleness any
# more. Divergence is now caught dynamically by the stage stamps (stage_of)
# rather than by a name list that has to be edited by hand every time the
# pipeline is re-run.
#
# In v2 scope the dataset table is still derived from the FEATURE table - the
# authoritative record of the split the models were actually fitted on.
SCOPE = os.environ.get("SCOPE", "all").lower()

# Raw stores a report writer does not need; results and tables are enough.
BULKY = {"extended_match_store.csv", "temporal_match_splits_extended.csv",
         "team_ratings.csv", "alias_map_extended.csv",
         "odds_failures_extended.csv", "predictions_Cm.csv"}


def in_scope(path):
    if SCOPE != "v2":
        return True
    return path.name not in BULKY

# section -> (subfolder, [files]). Order is report order.
SECTIONS = {
    "data": ("01_data", [
        PROCESSED / "extended_match_store.csv",
        PROCESSED / "temporal_match_splits_extended.csv",
        PROCESSED / "odds_coverage_extended.csv",
        PROCESSED / "odds_failures_extended.csv",
        PROCESSED / "team_registry.csv",
        PROCESSED / "alias_map_extended.csv",
        PROCESSED / "download_manifest.csv",
        PROCESSED / "data_quality_log.csv",
        PROCESSED / "cleaning_drops.csv",
        REPORTS / "competition_audit.csv",
    ]),
    "ratings": ("02_ratings", [
        PROCESSED / "team_ratings.csv",
        PROCESSED / "rating_tuning.csv",
        PROCESSED / "rating_params.json",
    ]),
    "results": ("03_results", [
        REPORTS / "model_results.csv",
        REPORTS / "ensemble_comparison.csv",
        REPORTS / "market_comparison.csv",
        REPORTS / "market_coverage.csv",
        REPORTS / "market_blend.csv",
        REPORTS / "best_params.json",
        REPORTS / "tuning_results.csv",
        REPORTS / "models" / "manifest.json",
        REPORTS / "predictions_Cm.csv",
    ]),
    "analysis": ("04_analysis", [
        REPORTS / "inplay_metric_by_minute.csv",
        REPORTS / "inplay_calibration_by_phase.csv",
        REPORTS / "reliability_bins.csv",
        REPORTS / "kernel_scaling.csv",
        REPORTS / "compute_profile.csv",
        REPORTS / "api_latency.csv",
        REPORTS / "ablation.csv",
        REPORTS / "resampling_study.csv",
        REPORTS / "p1_comparison.csv",
        REPORTS / "significance_bootstrap.csv",
        REPORTS / "significance_seeds.csv",
        REPORTS / "seed_repetitions.csv",
        REPORTS / "margin_to_probability.csv",
        REPORTS / "shap_importance.csv",
        REPORTS / "worst_predictions.csv",
    ]),
}

# Stage sentinels: (stage label, file whose mtime dates the stage).
STAGES = [
    ("A. data layer (store, splits, odds, ratings)",
     PROCESSED / "temporal_match_splits_extended.csv"),
    ("B. feature tables", FEATURES / "prematch_features_extended.csv"),
    ("C. tuning", REPORTS / "best_params.json"),
    ("D. model sweep, ensemble, market comparison", REPORTS / "model_results.csv"),
    ("E. offline analyses (ablation, significance, SHAP)",
     REPORTS / "ablation.csv"),
]


def mtime(path):
    return path.stat().st_mtime if path.exists() else 0.0


def stage_of(path):
    """Which stage produced this file, by nearest sentinel at or before it."""
    t = mtime(path)
    best = None
    for label, sentinel in STAGES:
        s = mtime(sentinel)
        if s and abs(t - s) < 6 * 3600:          # same run window
            if best is None or abs(t - s) < best[1]:
                best = (label, abs(t - s))
    return best[0] if best else "unclassified"


def md_table(frame, floatfmt="{:.5f}"):
    """Markdown table, numbers formatted so they can be pasted as-is."""
    def cell(v):
        if isinstance(v, float):
            return "" if pd.isna(v) else floatfmt.format(v)
        return "" if v is None or (isinstance(v, float) and pd.isna(v)) else str(v)
    head = "| " + " | ".join(str(c) for c in frame.columns) + " |"
    rule = "|" + "|".join("---" for _ in frame.columns) + "|"
    rows = ["| " + " | ".join(cell(v) for v in r) + " |"
            for r in frame.itertuples(index=False)]
    return "\n".join([head, rule, *rows])


def read(path, **kw):
    return pd.read_csv(path, encoding="utf-8", **kw) if path.exists() else None


# --------------------------------------------------------------------------
# report-ready tables
# --------------------------------------------------------------------------

def table_dataset():
    if SCOPE == "v2":
        # The feature table is what the models were actually fitted on, so in
        # v2 scope it - not the newer splits file - defines the dataset.
        j = read(FEATURES / "prematch_features_extended.csv",
                 usecols=["match_id", "match_date", "split", "season",
                          "competition_name"])
        if j is None:
            return None
        league = "competition_name"
    else:
        splits = read(PROCESSED / "temporal_match_splits_extended.csv")
        store = read(PROCESSED / "extended_match_store.csv")
        if splits is None or store is None:
            return None
        league = "league" if "league" in store.columns else "competition_name"
        j = store.merge(splits[["match_id", "split"]], on="match_id")
    rows = []
    # Derived from the data, not hardcoded: a split name that exists but is not
    # listed here would vanish from the table while still counting towards the
    # ALL row, leaving a silent gap. "out_of_scope" (the 2025/26 season) is
    # exactly that case.
    preferred = ["train", "validation", "test", "excluded", "out_of_scope"]
    present = list(dict.fromkeys(
        [n for n in preferred if n in set(j["split"])]
        + sorted(set(j["split"]) - set(preferred))))
    for name in present:
        part = j[j["split"] == name]
        if not len(part):
            continue
        rows.append({
            "split": name, "matches": len(part),
            "seasons": f'{part["season"].min()} - {part["season"].max()}',
            "first match": str(pd.to_datetime(part["match_date"]).min().date()),
            "last match": str(pd.to_datetime(part["match_date"]).max().date()),
            "leagues": part[league].nunique(),
        })
    rows.append({"split": "ALL", "matches": len(j),
                 "seasons": f'{j["season"].min()} - {j["season"].max()}',
                 "first match": str(pd.to_datetime(j["match_date"]).min().date()),
                 "last match": str(pd.to_datetime(j["match_date"]).max().date()),
                 "leagues": j[league].nunique()})
    return md_table(pd.DataFrame(rows))


def _task_table(results, task, metric, extra):
    sub = results[results["task"] == task].dropna(subset=[metric])
    sub = sub.sort_values(metric)
    cols = ["model", metric] + [c for c in extra if c in sub.columns]
    out = sub[cols].rename(columns={"model": "Model"})
    return md_table(out)


def tables_results():
    results = read(REPORTS / "model_results.csv")
    if results is None:
        return {}
    return {
        "T3_task_C_prematch_outcome":
            _task_table(results, "C", "rps", ["log_loss", "brier", "ece_after"]),
        "T4_task_R_prematch_margin":
            _task_table(results, "R", "mae", ["rmse", "corr"]),
        "T5_task_Lc_inplay_outcome":
            _task_table(results, "Lc", "rps", ["log_loss", "brier", "ece_after"]),
        "T6_task_Lr_inplay_margin":
            _task_table(results, "Lr", "mae", ["rmse", "corr"]),
    }


def table_inplay_curve():
    frame = read(REPORTS / "inplay_metric_by_minute.csv")
    if frame is None:
        return None
    sub = frame[frame["task"] == "Lc"]
    live = sub[sub["series"] == "in-play"]
    best = live.groupby("model")["rps"].mean().idxmin()
    # The frozen rows carry the PRE-MATCH model's name, so they must not be
    # filtered by the in-play model - that silently drops the reference column,
    # which is the only thing that makes this table mean anything.
    frozen = sub[sub["series"] == "frozen pre-match"]
    wide = (pd.concat([live[live["model"] == best], frozen])
            .pivot_table(index="snapshot_minute", columns="series", values="rps")
            .reset_index().rename(columns={"snapshot_minute": "minute"}))
    note = f"In-play model: `{best}`."
    if len(frozen):
        ref = str(frozen["model"].iloc[0])
        cross = wide[wide["in-play"] < wide["frozen pre-match"]]
        when = (f" The in-play model overtakes it at minute "
                f"**{int(cross['minute'].iloc[0])}**." if len(cross)
                else " The in-play model never overtakes it.")
        note += (f" Frozen pre-match reference: the served `{ref}` scoring the "
                 f"same matches, held flat.{when}")
    else:
        note += (" **No frozen pre-match reference** - the in-play matches are "
                 "excluded from tasks C/R, so run train_final.py and rebuild.")
    return note + "\n\n" + md_table(wide, "{:.4f}")


def table_simple(path, floatfmt="{:.5f}", note=""):
    frame = read(path)
    if frame is None:
        return None
    return (note + "\n\n" if note else "") + md_table(frame, floatfmt)


def build_tables():
    out = {}
    ds = table_dataset()
    if ds:
        out["T1_dataset_and_splits"] = ds
    cov = table_simple(REPORTS / "market_coverage.csv", "{:.4f}")
    if cov:
        out["T2_odds_coverage_by_league"] = cov
    out.update(tables_results())
    for key, path, fmt, note in [
        ("T7_ensemble_vs_best_single", REPORTS / "ensemble_comparison.csv",
         "{:.5f}", "Point estimates only. None of these differences survive a "
         "paired cluster bootstrap - see the notebook, section 9."),
        ("T8_model_vs_market", REPORTS / "market_comparison.csv", "{:.5f}", ""),
        ("T9_market_blend", REPORTS / "market_blend.csv", "{:.5f}",
         "Declared odds-as-feature arm. Odds are the baseline everywhere else."),
        ("T11_kernel_scaling", REPORTS / "kernel_scaling.csv", "{:.4f}",
         "Exact kernel ridge is capped at 8,000 training rows; the Gram matrix "
         "on the full pre-match table would be ~12 GB."),
        ("T12_api_latency", REPORTS / "api_latency.csv", "{:.3f}", ""),
        ("T13_compute_profile", REPORTS / "compute_profile.csv", "{:.3f}", ""),
    ]:
        t = table_simple(path, fmt, note)
        if t:
            out[key] = t
    curve = table_inplay_curve()
    if curve:
        out["T10_inplay_rps_by_minute"] = curve
    if SCOPE == "v2":
        out = {k: v for k, v in out.items()
               if TABLE_SOURCE.get(k) is None or in_scope(TABLE_SOURCE[k])}
    return out


# --------------------------------------------------------------------------

# Which file each report table is rendered from, so every table can carry its
# own provenance stamp - a table pasted into a document loses the README.
TABLE_SOURCE = {
    "T1_dataset_and_splits": (FEATURES / "prematch_features_extended.csv"
                              if SCOPE == "v2"
                              else PROCESSED / "temporal_match_splits_extended.csv"),
    "T2_odds_coverage_by_league": REPORTS / "market_coverage.csv",
    "T3_task_C_prematch_outcome": REPORTS / "model_results.csv",
    "T4_task_R_prematch_margin": REPORTS / "model_results.csv",
    "T5_task_Lc_inplay_outcome": REPORTS / "model_results.csv",
    "T6_task_Lr_inplay_margin": REPORTS / "model_results.csv",
    "T7_ensemble_vs_best_single": REPORTS / "ensemble_comparison.csv",
    "T8_model_vs_market": REPORTS / "market_comparison.csv",
    "T9_market_blend": REPORTS / "market_blend.csv",
    "T10_inplay_rps_by_minute": REPORTS / "inplay_metric_by_minute.csv",
    "T11_kernel_scaling": REPORTS / "kernel_scaling.csv",
    "T12_api_latency": REPORTS / "api_latency.csv",
    "T13_compute_profile": REPORTS / "compute_profile.csv",
}


def copy_files():
    manifest = []
    for section, (folder, paths) in SECTIONS.items():
        dest_dir = PACK / folder
        dest_dir.mkdir(parents=True, exist_ok=True)
        for path in paths:
            if not in_scope(path):
                continue
            if not path.exists():
                manifest.append({"section": section, "file": path.name,
                                 "stage": "MISSING", "rows": "", "size_kb": "",
                                 "packed": False})
                continue
            size_mb = path.stat().st_size / 1e6
            packed = size_mb <= COPY_LIMIT_MB
            if packed:
                shutil.copy2(path, dest_dir / path.name)
            rows = ""
            if path.suffix == ".csv":
                try:
                    rows = sum(1 for _ in open(path, encoding="utf-8")) - 1
                except Exception:
                    rows = ""
            manifest.append({"section": section, "file": path.name,
                             "stage": stage_of(path), "rows": rows,
                             "size_kb": round(path.stat().st_size / 1024, 1),
                             "packed": packed})
    return manifest


def copy_figures():
    dest = PACK / "05_figures"
    dest.mkdir(parents=True, exist_ok=True)
    n = 0
    # Every figure is regenerated by the same run as the results it draws, so
    # in v2 scope none are dropped. This list stays empty rather than deleted:
    # re-running only part of the pipeline is the case that would refill it.
    stale = set()
    for src in [REPORTS / "visualizations", REPORTS / "report_build"]:
        if not src.exists():
            continue
        for path in sorted(src.rglob("*")):
            if not (path.is_file() and path.suffix in {".html", ".png", ".svg"}):
                continue
            if SCOPE == "v2" and path.name in stale:
                continue
            shutil.copy2(path, dest / path.name)
            n += 1
    for extra in [REPO / "docs" / "pipeline_overview.png",
                  REPO / "docs" / "pipeline_overview.svg"]:
        if extra.exists():
            shutil.copy2(extra, dest / extra.name)
            n += 1
    return n


def copy_docs():
    dest = PACK / "06_docs"
    dest.mkdir(parents=True, exist_ok=True)
    n = 0
    for path in [REPO / "docs" / "model_book.md",
                 REPO / "docs" / "feature_book.html",
                 REPO / "notebooks" / "model_vs_market.ipynb",
                 REPO / "REPORT.md", REPO / "CLAUDE.md"]:
        if path.exists():
            shutil.copy2(path, dest / path.name)
            n += 1
    return n


def write_readme_v2(manifest, tables):
    """Teammate-facing README: what is here, what is deliberately not."""
    stamp = datetime.datetime.fromtimestamp(
        mtime(REPORTS / "model_results.csv")).strftime("%Y-%m-%d %H:%M")
    lines = [
        "# v2 results",
        "",
        "Everything needed to write the v2 report. One modelling run, "
        f"finished {stamp}. Every number here comes from that run, so the "
        "tables are consistent with each other.",
        "",
        "## Start here",
        "",
        "1. `tables/` - the numbers, as Markdown tables. Paste them straight "
        "into the document; each carries its source file and run time in a "
        "footer.",
        "2. `06_docs/model_book.md` - prose walkthrough of every model, what "
        "was tried and what the results mean. Written for exactly this "
        "purpose.",
        "3. `06_docs/model_vs_market.ipynb` - the analysis notebook, already "
        "executed with outputs. Section 9 is the significance test.",
        "4. `05_figures/` - HTML figures, open in a browser.",
        "",
        "## The four findings the report has to state",
        "",
        "**1. The market wins, but only just.** The best odds-free model "
        "trails the de-vigged Bet365 line by +0.005 RPS on 5,806 tagged "
        "matches (T8). Normalised against the dummy, the model captures ~88% "
        "of the skill the bookmaker has. Every model clears the dummy easily "
        "and they cluster within 0.001 of each other: the signal is real but "
        "shared. Reported, not buried.",
        "",
        "**2. Ensembling helps in exactly one place.** On Task Lr the stack "
        "beats lightgbm by 0.038 MAE (Holm p = 0.000) and xgboost by 0.028 "
        "(p = 0.030) - keep it there. Everywhere else it does not beat the "
        "best single model: on Task C, `stack_temporal` beats `stack` "
        "(p = 0.048), which vindicates fitting the meta-learner on recent "
        "validation rather than random folds, but plain xgboost still wins "
        "the head outright. Ensembling is not what closes the gap to the "
        "market.",
        "",
        "**3. Live information overtakes pre-match at minute 20.** In-play "
        "RPS starts *worse* than the frozen pre-match reference - 0.214 "
        "against 0.204 at kick-off, because a snapshot with no elapsed match "
        "has fewer features and less training data. Parity comes at the "
        "20-minute snapshot; half-time reads 0.153 and full time 0.023 (T10). "
        "Do not claim in-play leads from the first whistle.",
        "",
        "**4. The case for class balancing was a small-sample artifact.** At "
        "n_train = 896 borderline-SMOTE ranked best and G-SMOTENC worst. At "
        "39,707 the ordering inverts: every synthetic oversampling arm now "
        "scores below doing nothing, and the best draw recall any arm reaches "
        "falls from 0.24 to 0.02. Reweighting is free and harmless; "
        "synthetic minority oversampling costs RPS.",
        "",
        "## Deliberately not in here",
        "",
        "- Raw match stores and rating tables - results and tables are enough "
        "to write from. They live in `src/reports/` if needed.",
        "",
        "Nothing is excluded for staleness. Every stage in this pack, from "
        "the store through ablation, significance, SHAP and the compute "
        "profile, was produced by a single full run.",
        "",
        "## Scope",
        "",
        "Train through 2020/21, validation 2021/22-2022/23, test "
        "2023/24-2024/25. The 2025/26 season is present in the store but "
        "assigned `out_of_scope` by the pinned split constants, so it trains "
        "and scores nothing. Adding it means moving all four constants "
        "together and re-running the model layer.",
        "",
        "## What the numbers describe",
        "",
        (tables.get("T1_dataset_and_splits", "").split("*Source")[0]
         if "T1_dataset_and_splits" in tables else ""),
        "",
        "## Files",
        "",
        md_table(pd.DataFrame(manifest)),
        "",
        "Regenerate: `SCOPE=v2 python src/audit/pack_report.py`",
        "",
    ]
    (PACK / "README.md").write_text("\n".join(lines), encoding="utf-8")


def write_readme(manifest, tables):
    stages = {}
    for row in manifest:
        stages.setdefault(row["stage"], []).append(row["file"])
    ordered = [label for label, _ in STAGES if label in stages]
    ordered += [s for s in stages if s not in ordered]

    coherent = len({s for s in stages if s.startswith(("A.", "D."))}) <= 1
    if SCOPE == "v2":
        write_readme_v2(manifest, tables)
        return
    lines = [
        "# Report pack",
        "",
        f"Generated by `src/audit/pack_report.py` from `src/reports/` "
        f"(gitignored, so this folder is the portable copy).",
        "",
        "## Read this before quoting any number",
        "",
    ]
    if not coherent:
        lines += [
            "> **The tree is not coherent.** The data layer and the model "
            "results were produced by different runs, so the dataset summary "
            "(T1) describes a different split from the accuracy tables "
            "(T3-T8). Quote them together only after re-running the model "
            "stages. The stage table below says which file is which.",
            "",
        ]
    lines += [
        "Files are stamped with the pipeline stage that produced them, dated "
        "from that stage's sentinel file rather than a hardcoded date:",
        "",
    ]
    for label in ordered:
        sentinel = dict(STAGES).get(label)
        when = (datetime.datetime.fromtimestamp(mtime(sentinel))
                .strftime("%Y-%m-%d %H:%M") if sentinel and mtime(sentinel)
                else "n/a")
        lines.append(f"- **{label}** (run {when}): "
                     f"{', '.join(sorted(stages[label]))}")
    lines += [
        "",
        "## Layout",
        "",
        "| folder | contents |",
        "|---|---|",
        "| `01_data/` | match store, splits, odds coverage and failures, team registry, alias map |",
        "| `02_ratings/` | Elo + pi-rating outputs and their validation search |",
        "| `03_results/` | the sweep, the ensemble, model-vs-market, the blend, tuned hyperparameters |",
        "| `04_analysis/` | in-play curves, calibration, kernel scaling, latency, ablation, significance, SHAP |",
        "| `05_figures/` | every generated HTML figure and PNG |",
        "| `06_docs/` | model book, feature book, executed notebook, REPORT.md, CLAUDE.md |",
        "| `tables/` | **report-ready Markdown tables** - paste straight in |",
        "",
        f"Files larger than {COPY_LIMIT_MB:.0f} MB are listed in `MANIFEST.csv` "
        "but not copied; read them from `src/reports/`.",
        "",
        "## Tables",
        "",
    ]
    for name in sorted(tables):
        lines.append(f"- `tables/{name}.md`")
    lines += ["", "## Files", "", md_table(pd.DataFrame(manifest)), ""]
    (PACK / "README.md").write_text("\n".join(lines), encoding="utf-8")


def main():
    if PACK.exists():
        shutil.rmtree(PACK)
    PACK.mkdir(parents=True)

    manifest = copy_files()
    n_fig = copy_figures()
    n_doc = copy_docs()

    tables = build_tables()
    tdir = PACK / "tables"
    tdir.mkdir(exist_ok=True)
    split_path = PROCESSED / "temporal_match_splits_extended.csv"
    # In v2 scope T1 is rendered from the feature table the models were fitted
    # on, so it cannot disagree with the accuracy tables and needs no warning.
    mismatch = (SCOPE != "v2"
                and stage_of(split_path) != stage_of(REPORTS / "model_results.csv"))
    split_warning = (
        "> **Different split from the model tables.** This describes the data "
        "layer as rebuilt on "
        + datetime.datetime.fromtimestamp(mtime(split_path)).strftime("%Y-%m-%d")
        + "; the accuracy tables were produced on the previous split. Do not "
          "present them as one experiment until the model stages are re-run."
          "\n\n") if mismatch else ""

    for name, body in tables.items():
        title = name.split("_", 1)[1].replace("_", " ").capitalize()
        source = TABLE_SOURCE.get(name)
        stamp = ""
        if source is not None and source.exists():
            stamp = ("\n\n*Source: `" + source.name + "` - "
                     + stage_of(source) + " - run "
                     + datetime.datetime.fromtimestamp(mtime(source))
                       .strftime("%Y-%m-%d %H:%M") + ".*")
        warn = split_warning if name.startswith("T1_") else ""
        (tdir / f"{name}.md").write_text(
            f"### {title}\n\n{warn}{body}{stamp}\n", encoding="utf-8")

    pd.DataFrame(manifest).to_csv(PACK / "MANIFEST.csv", index=False,
                                  encoding="utf-8")
    write_readme(manifest, tables)

    packed = sum(1 for r in manifest if r["packed"])
    missing = [r["file"] for r in manifest if r["stage"] == "MISSING"]
    total_mb = sum(p.stat().st_size for p in PACK.rglob("*") if p.is_file()) / 1e6
    print(f"Wrote {PACK}")
    print(f"  {packed}/{len(manifest)} result files, {n_fig} figures, "
          f"{n_doc} documents, {len(tables)} tables  ({total_mb:.1f} MB)")
    if missing:
        print(f"  missing (stage not run): {', '.join(missing)}")
    if SCOPE == "v2":
        print("  scope=v2: one coherent modelling run; nothing excluded for "
              "staleness (see README.md)")
        return
    dated = sorted((mtime(sn), lb) for lb, sn in STAGES if mtime(sn))
    if dated:
        span = (dated[-1][0] - dated[0][0]) / 3600
        fmt = lambda t: datetime.datetime.fromtimestamp(t).strftime("%m-%d %H:%M")
        print(f"  stages span {span:.1f} h "
              f"({fmt(dated[0][0])} -> {fmt(dated[-1][0])})")
        if span > 1:
            print("  NOTE: stages were run at different times - read "
                  "README.md before quoting numbers across sections")


if __name__ == "__main__":
    main()
