import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from plotly.offline import get_plotlyjs
from pathlib import Path

# --- Configuration ---
HERE = Path(__file__).resolve().parent
PROJECT = HERE.parent
PROCESSED_DIR = PROJECT / "reports" / "processed"
VIS_DIR = PROJECT / "reports" / "visualizations"
VIS_DIR.mkdir(parents=True, exist_ok=True)

LEAGUE_ORDER = ["La Liga", "Premier League", "Serie A", "Ligue 1"]
OUTCOME_COLORS = {"Home win": "#2E7D32", "Draw": "#9E9E9E", "Away win": "#C62828"}
METHOD_COLORS = {"exact": "#2E7D32", "fuzzy": "#1565C0", "override": "#B0791A"}
TEMPLATE = "plotly_white"


def coverage_figure(coverage: pd.DataFrame) -> go.Figure:
    coverage = coverage.set_index("league").reindex(LEAGUE_ORDER).reset_index()
    fig = px.bar(
        coverage, x="league", y="coverage", template=TEMPLATE,
        text=[f"{t}/{n}" for t, n in zip(coverage["tagged_with_odds"], coverage["fd_matches"])],
    )
    fig.update_traces(marker_color="#3D195B", textposition="outside")
    fig.update_layout(
        title="Odds tagging coverage per league",
        xaxis_title="", yaxis_title="Coverage", yaxis_tickformat=".0%",
        yaxis_range=[0.9, 1.02], height=430,
    )
    return fig


def alias_method_figure(alias_map: pd.DataFrame) -> go.Figure:
    counts = alias_map.groupby(["league", "method"]).size().reset_index(name="teams")
    fig = px.bar(
        counts, x="league", y="teams", color="method", barmode="stack",
        category_orders={"league": LEAGUE_ORDER, "method": ["exact", "fuzzy", "override"]},
        color_discrete_map=METHOD_COLORS, template=TEMPLATE,
    )
    fig.update_layout(
        title="How each team name was matched between the two sources",
        xaxis_title="", yaxis_title="Teams", legend_title="Match method", height=430,
    )
    return fig


def overround_figure(baseline: pd.DataFrame) -> go.Figure:
    fig = px.box(
        baseline, x="competition_name", y="overround", template=TEMPLATE,
        category_orders={"competition_name": LEAGUE_ORDER},
    )
    fig.update_traces(marker_color="#C62828")
    fig.add_hline(y=1.0, line_dash="dash", line_color="#9E9E9E",
                  annotation_text="fair (no margin)")
    fig.update_layout(
        title="Bookmaker over-round (the margin removed by de-vigging)",
        xaxis_title="", yaxis_title="Sum of raw implied probabilities", height=430,
    )
    return fig


def reliability_figure(baseline: pd.DataFrame) -> go.Figure:
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=[0, 1], y=[0, 1], mode="lines", name="perfect",
                             line=dict(color="#9E9E9E", dash="dash")))
    for label, prob_col, letter in [("Home win", "p_home", "H"),
                                     ("Draw", "p_draw", "D"),
                                     ("Away win", "p_away", "A")]:
        frame = pd.DataFrame({"predicted": baseline[prob_col],
                              "actual": (baseline["result"] == letter).astype(int)})
        frame["bin"] = np.clip((frame["predicted"] * 10).astype(int), 0, 9)
        grouped = frame.groupby("bin").agg(
            predicted=("predicted", "mean"), observed=("actual", "mean"), n=("actual", "size")
        ).reset_index()
        fig.add_trace(go.Scatter(
            x=grouped["predicted"], y=grouped["observed"], mode="markers+lines",
            name=label, marker=dict(size=grouped["n"] / 12 + 5, color=OUTCOME_COLORS[label]),
        ))
    fig.update_layout(
        title="Market calibration: predicted probability vs how often it happened",
        xaxis_title="Market probability (de-vigged)", yaxis_title="Observed frequency",
        xaxis_range=[0, 1], yaxis_range=[0, 1], height=470,
    )
    return fig


def ternary_figure(baseline: pd.DataFrame) -> go.Figure:
    labelled = baseline.copy()
    labelled["outcome"] = labelled["result"].map({"H": "Home win", "D": "Draw", "A": "Away win"})
    fig = px.scatter_ternary(
        labelled, a="p_home", b="p_draw", c="p_away", color="outcome",
        color_discrete_map=OUTCOME_COLORS, opacity=0.5, template=TEMPLATE,
    )
    fig.update_traces(marker=dict(size=5))
    fig.update_layout(
        title="Every match by its market probabilities, coloured by the actual result",
        legend_title="Actual result", height=520,
    )
    return fig


CAPTIONS = {
    "coverage": "The fraction of odds rows successfully tagged to a StatsBomb match and carrying Bet365 odds. Three leagues reach 100%. Ligue 1 reaches 98.9%: three fixtures are absent from the StatsBomb release and one lacks Bet365 odds. The test set uses only matches shown here, so models and the market are always compared on identical games.",
    "alias": "How the two sources' team names were reconciled. Most teams match exactly after accent-stripping; 32 are resolved by the token edit-distance matcher (handling abbreviations such as Man City to Manchester City); one needs a documented override (Ath Madrid). A bijection check guarantees the map is one-to-one, so a wrong guess cannot pass silently.",
    "overround": "The bookmaker margin baked into the raw odds. The three raw implied probabilities sum to this value rather than 1; de-vigging divides by it to recover fair probabilities. The margin sits around 4 to 5 percent, which is roughly what the market keeps on average.",
    "reliability": "The reason the market is the baseline to beat. Each point is a probability bin: its horizontal position is what the market predicted, its vertical position is how often that actually happened, and marker size is how many matches fell in the bin. Points sitting on the dashed diagonal mean the market's probabilities are well calibrated and trustworthy.",
    "ternary": "Each match placed by its three de-vigged probabilities and coloured by what actually happened. The three corners are certainty; the centre is maximum uncertainty. Draws (grey) sit toward the middle and are almost never the single most likely outcome, while confident home (green) and away (red) predictions sit near their corners and mostly land there.",
}


def build_dashboard():
    baseline = pd.read_csv(PROCESSED_DIR / "market_baseline.csv", encoding="utf-8")
    coverage = pd.read_csv(PROCESSED_DIR / "odds_coverage.csv", encoding="utf-8")
    alias_map = pd.read_csv(PROCESSED_DIR / "alias_map.csv", encoding="utf-8")
    store = pd.read_csv(PROCESSED_DIR / "match_store.csv", encoding="utf-8")
    baseline = baseline.merge(store[["match_id", "result"]], on="match_id")

    sections = [
        ("Integration", "coverage", coverage_figure(coverage)),
        ("Integration", "alias", alias_method_figure(alias_map)),
        ("De-vigging", "overround", overround_figure(baseline)),
        ("Market baseline", "reliability", reliability_figure(baseline)),
        ("Market baseline", "ternary", ternary_figure(baseline)),
    ]

    blocks = []
    for layer, key, fig in sections:
        chart = fig.to_html(full_html=False, include_plotlyjs=False)
        blocks.append(
            f'<section><p class="eyebrow">{layer}</p>{chart}'
            f'<p class="caption">{CAPTIONS[key]}</p></section>'
        )

    page = f"""<!DOCTYPE html>
<html lang="en"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>Market Baseline Overview</title>
<style>
  body{{margin:0;background:#F4F6F4;color:#16211C;
    font-family:-apple-system,Segoe UI,Roboto,Helvetica,Arial,sans-serif;}}
  .wrap{{max-width:900px;margin:0 auto;padding:40px 24px 80px;}}
  h1{{font-size:30px;margin:0 0 6px;}}
  .lead{{color:#4A5751;margin:0 0 8px;font-size:17px;}}
  .scope{{color:#4A5751;font-size:13px;border-top:1px solid #D5DCD6;
    margin-top:16px;padding-top:12px;}}
  section{{background:#fff;border:1px solid #E1E6E1;border-radius:10px;
    padding:14px 16px 8px;margin:22px 0;}}
  .eyebrow{{font-family:ui-monospace,Menlo,Consolas,monospace;font-size:11px;
    letter-spacing:.14em;text-transform:uppercase;color:#4A5751;margin:2px 4px 8px;}}
  .caption{{color:#3A453F;font-size:14.5px;line-height:1.55;margin:6px 6px 12px;}}
</style></head>
<body><div class="wrap">
  <h1>Market baseline - visual overview</h1>
  <p class="lead">From raw Bet365 odds to the de-vigged probabilities every model is judged against.</p>
  <p class="scope">Built from {len(baseline)} tagged matches across the four leagues.
  Sources: Football-Data.co.uk 2015/16 odds, tagged to StatsBomb Open Data by date and team identity.</p>
  <script>{get_plotlyjs()}</script>
  {''.join(blocks)}
</div></body></html>"""

    output_path = VIS_DIR / "market_baseline_overview.html"
    output_path.write_text(page, encoding="utf-8")
    print(f"Wrote dashboard ({len(page)//1024} KB) -> {output_path}")


if __name__ == "__main__":
    build_dashboard()
