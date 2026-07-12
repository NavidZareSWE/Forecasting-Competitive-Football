from pathlib import Path
import pandas as pd
import plotly.graph_objects as go

HERE = Path(__file__).resolve().parent
PROJECT = HERE.parent
match_store = pd.read_csv(
    PROJECT / "reports" / "processed" / "match_store.csv", encoding="utf-8")

outcome_share_by_league = (
    match_store.groupby("competition_name")["result"]
    .value_counts(normalize=True)
    .unstack()
    .reindex(columns=["H", "D", "A"])
    .sort_index()
)

outcome_labels = {"H": "Home win", "D": "Draw", "A": "Away win"}
outcome_colors = {"H": "#2E7D32", "D": "#9E9E9E", "A": "#C62828"}

fig = go.Figure()
for outcome in ["H", "D", "A"]:
    fig.add_bar(
        name=outcome_labels[outcome],
        x=outcome_share_by_league.index,
        y=outcome_share_by_league[outcome],
        marker_color=outcome_colors[outcome],
        text=[f"{share*100:.1f}%" for share in outcome_share_by_league[outcome]],
        textposition="auto",
    )

fig.update_layout(
    title="Raw match outcomes by league - StatsBomb 2015/16 (n=1,517)",
    barmode="group",
    yaxis_title="Share of matches",
    yaxis_tickformat=".0%",
    xaxis_title="",
    legend_title="Outcome (home team's view)",
    template="plotly_white",
    font=dict(size=13),
)

output_path = PROJECT / "reports" / "visualizations" / \
    "raw_result_distribution.html"
fig.write_html(output_path, include_plotlyjs="cdn")
print("Wrote", output_path)
