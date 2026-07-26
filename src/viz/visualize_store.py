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

LEAGUE_COLORS = {
    "La Liga": "#C8102E",
    "Premier League": "#3D195B",
    "Serie A": "#008FD7",
    "Ligue 1": "#091C3E",
}
TEMPLATE = "plotly_white"


# --- Matches layer ---
def season_timeline(matches: pd.DataFrame) -> go.Figure:
    matches = matches.copy()
    matches["week"] = matches["match_date"].dt.to_period("W").dt.start_time
    weekly = matches.groupby(["week", "competition_name"]).size().reset_index(name="matches")
    fig = px.bar(
        weekly, x="week", y="matches", color="competition_name",
        color_discrete_map=LEAGUE_COLORS, barmode="stack", template=TEMPLATE,
    )
    fig.update_layout(
        title="Matches per week across the 2015/16 season (all four leagues)",
        xaxis_title="", yaxis_title="Matches", legend_title="League", height=430,
    )
    return fig


def margin_distribution(matches: pd.DataFrame) -> go.Figure:
    counts = matches["margin"].value_counts().sort_index().reset_index()
    counts.columns = ["margin", "matches"]
    fig = px.bar(counts, x="margin", y="matches", template=TEMPLATE)
    fig.update_traces(marker_color="#2E7D32")
    fig.update_layout(
        title="Goal margin distribution (home minus away, clipped to [-5, 5]) - the Task R label",
        xaxis_title="Goal margin", yaxis_title="Matches", height=430,
    )
    fig.update_xaxes(dtick=1)
    return fig


# --- Lineups layer ---
def starters_per_match(lineups: pd.DataFrame) -> go.Figure:
    per_match = lineups.groupby("match_id")["started"].sum().reset_index(name="starters")
    fig = px.histogram(per_match, x="starters", template=TEMPLATE, nbins=30)
    fig.update_traces(marker_color="#1565C0")
    fig.add_vline(x=22, line_dash="dash", line_color="#B23A48",
                  annotation_text="expected 22 (11 per team)")
    fig.update_layout(
        title=f"Starters counted per match ({per_match['match_id'].nunique()} matches) - validates the started flag",
        xaxis_title="Starters in the match", yaxis_title="Matches", height=430,
    )
    return fig


# --- Events layer ---
def xg_vs_goals(events_index: pd.DataFrame) -> go.Figure:
    home = events_index[["home_xg", "home_goals_events", "score_ok"]].rename(
        columns={"home_xg": "xg", "home_goals_events": "goals"})
    home["side"] = "Home"
    away = events_index[["away_xg", "away_goals_events", "score_ok"]].rename(
        columns={"away_xg": "xg", "away_goals_events": "goals"})
    away["side"] = "Away"
    long = pd.concat([home, away], ignore_index=True)

    rng = np.random.default_rng(0)
    long["goals_jitter"] = long["goals"] + rng.uniform(-0.15, 0.15, len(long))

    top = float(max(long["xg"].max(), long["goals"].max())) + 0.5
    fig = px.scatter(
        long, x="xg", y="goals_jitter", color="side",
        color_discrete_map={"Home": "#2E7D32", "Away": "#C62828"},
        opacity=0.7, template=TEMPLATE,
    )
    fig.add_trace(go.Scatter(
        x=[0, top], y=[0, top], mode="lines", name="xG = goals",
        line=dict(color="#9E9E9E", dash="dash"),
    ))
    fig.update_layout(
        title=f"Expected goals (xG) vs actual goals per team ({len(events_index)} matches)",
        xaxis_title="Team xG in the match", yaxis_title="Actual goals (jittered)", height=460,
    )
    return fig


def event_file_health(events_index: pd.DataFrame) -> go.Figure:
    events_index = events_index.copy()
    events_index["status"] = np.where(
        events_index["score_ok"], "score_ok = True", "score_ok = False")
    fig = px.histogram(
        events_index, x="n_events", color="status", template=TEMPLATE, nbins=40,
        color_discrete_map={"score_ok = True": "#2E7D32", "score_ok = False": "#B23A48"},
    )
    fig.update_layout(
        title=f"Events per match file, coloured by the score cross-check ({len(events_index)} matches)",
        xaxis_title="Events in the match file", yaxis_title="Matches",
        legend_title="", height=430,
    )
    return fig


# --- Assembly ---
CAPTIONS = {
    "timeline": "Every match sits on a real calendar date. This is the shape the temporal train/test split will cut through: earlier weeks become training, later weeks become the held-out test set.",
    "margin": "The regression label. It is bell-shaped around a draw (margin 0) and leans slightly positive, which is home advantage. Clipping to [-5, 5] only affects a handful of extreme scorelines.",
    "starters": "Each match should have exactly 22 starters (11 per team). The sharp single spike at 22 confirms the started flag behaves as intended across the sample.",
    "xg": "Each point is one team in one match. Points above the dashed line scored more than their chances were worth (finishing or luck); points below scored fewer. The cloud rising left to right shows xG tracks goals without being identical to them - which is exactly why it is a useful, more stable signal.",
    "health": "A per-match health check. n_events clusters in the normal few-thousand range (no truncated files). Bars are coloured by whether goals reconstructed from the events matched the official score; any red bar is a match to inspect (here, one match flagged only because of a team-name spelling difference between StatsBomb files, not a missing goal).",
}


def build_dashboard():
    matches = pd.read_csv(PROCESSED_DIR / "match_store.csv", encoding="utf-8",
                          parse_dates=["match_date"])
    lineups = pd.read_csv(PROCESSED_DIR / "lineups.csv", encoding="utf-8")
    events_index = pd.read_csv(PROCESSED_DIR / "events_index.csv", encoding="utf-8")

    sections = [
        ("Matches layer", "timeline", season_timeline(matches)),
        ("Matches layer", "margin", margin_distribution(matches)),
        ("Lineups layer", "starters", starters_per_match(lineups)),
        ("Events layer", "xg", xg_vs_goals(events_index)),
        ("Events layer", "health", event_file_health(events_index)),
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
<title>Relational Store Overview</title>
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
  <h1>Relational store - visual overview</h1>
  <p class="lead">The three built layers (matches, lineups, events), keyed by match_id.</p>
  <p class="scope">Matches: full store ({len(matches)} matches). Lineups and events: sample of
  {events_index['match_id'].nunique()} matches built in this session; re-run the builders
  uncapped for all {len(matches)}. Data: StatsBomb Open Data 2015/16.</p>
  <script>{get_plotlyjs()}</script>
  {''.join(blocks)}
</div></body></html>"""

    output_path = VIS_DIR / "store_overview.html"
    output_path.write_text(page, encoding="utf-8")
    print(f"Wrote dashboard ({len(page)//1024} KB) -> {output_path}")


if __name__ == "__main__":
    build_dashboard()
