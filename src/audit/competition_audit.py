import json
import pathlib
import collections

import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots


BASE_DIR = pathlib.Path(__file__).resolve().parent
ROOT = BASE_DIR.parent / "data" / "statsbomb_open_data" / "data"
REPORTS_DIR = BASE_DIR.parent / "reports"


def build_audit(root: pathlib.Path) -> pd.DataFrame:
    rows = []
    for cs in json.loads((root / "competitions.json").read_text(encoding="utf-8")):
        cid, sid = cs["competition_id"], cs["season_id"]
        f = root / "matches" / str(cid) / f"{sid}.json"
        if not f.exists():
            rows.append(dict(cid=cid, sid=sid, comp=cs["competition_name"],
                             season=cs["season_name"], n_matches=0,
                             note="NO MATCHES FILE"))
            continue

        ms = json.loads(f.read_text(encoding="utf-8"))
        teams = collections.Counter()
        dates, n360 = [], 0
        for m in ms:
            teams[m["home_team"]["home_team_name"]] += 1
            teams[m["away_team"]["away_team_name"]] += 1
            dates.append(m["match_date"])
            if m.get("match_status_360") == "available":
                n360 += 1

        n = len(ms)
        top_team, top_cnt = teams.most_common(1)[0]
        rows.append(dict(
            cid=cid, sid=sid, comp=cs["competition_name"], season=cs["season_name"],
            n_matches=n, n_teams=len(teams), top_team=top_team,
            # 1.0 => every match has this club
            top_team_share=round(top_cnt / n, 3),
            date_min=min(dates), date_max=max(dates),
            pct_360=round(n360 / n, 3),
            gender=cs.get("competition_gender"), country=cs.get("country_name"),
        ))

    return pd.DataFrame(rows).sort_values(
        ["top_team_share", "n_matches"], ascending=[True, False]
    )


# --------------------------------------------------------------------------
# 2. The competition-selection rule (single source of truth)
# --------------------------------------------------------------------------
CONFEDERATIONS = {"International", "Europe", "Africa",
                  "South America", "North and Central America"}
ODDS_SOURCE_COUNTRIES = {"Spain", "England", "Italy", "France", "Germany"}
SINGLE_CLUB_THRESHOLD = 0.5

# Colour + ordering for every decision category
DECISIONS = {
    "Included - full league (2015/2016)":       "#2fbf8f",
    "Excluded - single-club / finals":          "#e5645e",
    "Excluded - tournament (no odds source)":   "#8a93b0",
    "Excluded - women's (no odds source)":      "#b7728f",
    "Excluded - league not in odds source":     "#c9a24b",
}


def categorise(row) -> str:

    if row["top_team_share"] >= SINGLE_CLUB_THRESHOLD:
        # every (or almost every) match involves one club -> longitudinal
        # single-club release, or a one-off final. Opponent-side form
        # features would be undefined for ~19 of 20 teams.
        return "Excluded - single-club / finals"
    if row["gender"] == "female":
        return "Excluded - women's (no odds source)"
    if row["country"] in CONFEDERATIONS:
        # tournaments: World Cups, Euros, continental cups -> not joinable
        # to Football-Data.co.uk, so cannot enter the test set.
        return "Excluded - tournament (no odds source)"
    if row["country"] in ODDS_SOURCE_COUNTRIES:
        return "Included - full league (2015/2016)"
    return "Excluded - league not in odds source"


# --------------------------------------------------------------------------
# 3. Figures - each one backs a specific sentence in the report
# --------------------------------------------------------------------------
TEMPLATE = "plotly_white"
FONT = dict(family="Georgia, 'Times New Roman', serif",
            size=13, color="#1f2430")


def fig_release_map(df: pd.DataFrame) -> go.Figure:

    fig = px.scatter(
        df, x="top_team_share", y="n_matches", color="decision",
        color_discrete_map=DECISIONS, log_y=True,
        size="n_teams", size_max=18, opacity=0.85,
        custom_data=["comp", "season", "top_team", "n_teams"],
        category_orders={"decision": list(DECISIONS)},
    )
    fig.update_traces(hovertemplate=(
        "<b>%{customdata[0]} %{customdata[1]}</b><br>"
        "matches: %{y}<br>teams: %{customdata[3]}<br>"
        "top team: %{customdata[2]} (share %{x})<extra></extra>"
    ))
    # the decision gate
    fig.add_vline(x=SINGLE_CLUB_THRESHOLD, line_dash="dash", line_color="#888",
                  annotation_text="selection gate  (share \u2265 0.5 \u2192 single-club)",
                  annotation_position="top")
    # regime labels
    fig.add_annotation(x=0.10, y=380, ax=0, ay=-46, text="the four usable<br>full leagues",
                       showarrow=True, arrowhead=2, font=dict(color="#1f7a5a", size=12))
    fig.add_annotation(x=1.0, y=36, ax=0, ay=-40, text="single-club<br>biographies",
                       showarrow=True, arrowhead=2, font=dict(color="#a83a35", size=12))
    fig.update_layout(
        title="<b>Figure 1 - Anatomy of the repository.</b> Where each competition-season sits by "
              "single-club share vs. size",
        xaxis_title="top_team_share  \u2014  fraction of the season's matches involving its most-frequent club",
        yaxis_title="matches in season (log scale)",
        template=TEMPLATE, font=FONT, legend_title_text="selection decision",
        height=560, margin=dict(t=90, r=30, b=70, l=70),
    )
    return fig


def fig_share_strip(df: pd.DataFrame) -> go.Figure:

    fig = px.strip(
        df, x="top_team_share", color="decision",
        color_discrete_map=DECISIONS, stripmode="overlay",
        custom_data=["comp", "season"],
        category_orders={"decision": list(DECISIONS)},
    )
    # px.strip builds hidden go.Box traces; their jitter is scaled to box
    # WIDTH, which px leaves as None, so update_traces(jitter=...) alone does
    # nothing (plotly.py issue #4563). Setting an explicit width restores it.
    fig.update_traces(width=0.8, jitter=1.0, pointpos=0,
                      marker=dict(size=9, opacity=0.7),
                      hovertemplate="<b>%{customdata[0]} %{customdata[1]}</b><br>share %{x}<extra></extra>")
    fig.add_vrect(x0=0.3, x1=0.7, fillcolor="#cccccc", opacity=0.18, line_width=0,
                  annotation_text="near-empty middle", annotation_position="top")
    fig.update_layout(
        title="<b>Figure 2 - The split is bimodal.</b> Almost nothing lives between a league (~0.10) "
              "and a single-club release (1.0)",
        xaxis_title="top_team_share", yaxis_title="", showlegend=True,
        template=TEMPLATE, font=FONT, legend_title_text="selection decision",
        height=340, margin=dict(t=90, r=30, b=60, l=40), yaxis=dict(showticklabels=False),
    )
    return fig


def fig_360(df: pd.DataFrame) -> go.Figure:
    """FINDING 2: none of the four included leagues carry 360 data. The 360
    that exists in the repo lives entirely in competitions we exclude, so the
    'impute missing 360' angle does not apply to our training corpus."""
    inc = df[df["decision"].str.startswith("Included")].copy()
    rich = df[df["pct_360"] > 0].copy()
    show = pd.concat([inc, rich]).drop_duplicates(subset=["cid", "sid"])
    show["label"] = show["comp"] + " " + show["season"].astype(str)
    show = show.sort_values("pct_360")
    fig = px.bar(
        show, x="pct_360", y="label", orientation="h", color="decision",
        color_discrete_map=DECISIONS, category_orders={
            "decision": list(DECISIONS)},
        custom_data=["n_matches"],
    )
    fig.update_traces(
        hovertemplate="<b>%{y}</b><br>360 coverage %{x:.0%}<br>matches %{customdata[0]}<extra></extra>")
    fig.update_layout(
        title="<b>Figure 3 - StatsBomb 360 is absent from every league we train on.</b> "
              "The four included leagues sit at 0%",
        xaxis_title="share of matches with 360 freeze-frames", yaxis_title="",
        xaxis_tickformat=".0%", template=TEMPLATE, font=FONT,
        legend_title_text="selection decision",
        height=520, margin=dict(t=90, r=30, b=60, l=260),
    )
    return fig


def fig_decision_totals(df: pd.DataFrame) -> go.Figure:
    """FINDING 3: the selection rule and its cost, in matches. Shows how many
    matches each exclusion reason removes, and how many survive."""
    g = (df.groupby("decision")
           .agg(seasons=("cid", "size"), matches=("n_matches", "sum"))
           .reindex(list(DECISIONS)).dropna().reset_index())
    g["matches"] = g["matches"].astype(int)
    fig = px.bar(
        g, x="matches", y="decision", orientation="h", color="decision",
        color_discrete_map=DECISIONS, text="matches",
        custom_data=["seasons"], category_orders={"decision": list(DECISIONS)},
    )
    fig.update_traces(textposition="outside",
                      hovertemplate="<b>%{y}</b><br>%{x} matches across %{customdata[0]} season(s)<extra></extra>")
    fig.update_layout(
        title="<b>Figure 4 - The selection rule, priced in matches.</b> "
              "1,517 matches survive; every exclusion has a stated reason",
        xaxis_title="matches", yaxis_title="", showlegend=False,
        template=TEMPLATE, font=FONT,
        height=380, margin=dict(t=90, r=70, b=60, l=300),
    )
    return fig


def fig_included_detail(df: pd.DataFrame) -> go.Figure:
    """The four survivors, side by side: match count and home/draw/away is not
    known here (needs labels), so we show size + team count + 360 = 0 as the
    at-a-glance profile of the training corpus."""
    inc = df[df["decision"].str.startswith(
        "Included")].copy().sort_values("n_matches")
    inc["label"] = inc["comp"] + " " + inc["season"].astype(str)
    fig = make_subplots(rows=1, cols=2, shared_yaxes=True,
                        subplot_titles=("matches in season",
                                        "teams in league"),
                        horizontal_spacing=0.08)
    fig.add_trace(go.Bar(x=inc["n_matches"], y=inc["label"], orientation="h",
                         marker_color="#2fbf8f", text=inc["n_matches"],
                         textposition="outside", showlegend=False), row=1, col=1)
    fig.add_trace(go.Bar(x=inc["n_teams"], y=inc["label"], orientation="h",
                         marker_color="#7c6cff", text=inc["n_teams"],
                         textposition="outside", showlegend=False), row=1, col=2)
    fig.update_layout(
        title="<b>Figure 5 - The training corpus.</b> Four full leagues, one season (2015/2016). "
              "Note Ligue 1 has 377, not 380",
        template=TEMPLATE, font=FONT, height=340,
        margin=dict(t=100, r=40, b=40, l=160),
    )
    fig.update_xaxes(title_text="matches", row=1, col=1)
    fig.update_xaxes(title_text="teams", row=1, col=2)
    return fig


# --------------------------------------------------------------------------
# 4. Assemble the standalone HTML report
# --------------------------------------------------------------------------
def write_report(df: pd.DataFrame, out: pathlib.Path) -> None:
    inc = df[df["decision"].str.startswith("Included")]
    total_incl = int(inc["n_matches"].sum())

    figs = [
        fig_release_map(df),
        fig_share_strip(df),
        fig_360(df),
        fig_decision_totals(df),
        fig_included_detail(df),
    ]

    # First figure embeds the full plotly.js source (include_plotlyjs=True ->
    # ~3 MB, fully self-contained, works offline). The rest reference the same
    # embedded copy (=False) so the library is included exactly once.
    blocks = []
    for i, fig in enumerate(figs):
        blocks.append(fig.to_html(
            full_html=False,
            include_plotlyjs=(True if i == 0 else False),
            config={"displayModeBar": False, "responsive": True},
        ))

    intro = f"""
    <header>
      <p class="eyebrow">Machine Learning Final Project &middot; Phase 1 &middot; Competition Selection</p>
      <h1>Which competitions we train on, and why</h1>
      <p class="lede">Read only from the <code>matches/*.json</code> metadata &mdash; no events file was
      opened to make this decision. Of {len(df)} competition-seasons in the StatsBomb open-data repository,
      <b>4</b> survive selection: La&nbsp;Liga, Premier&nbsp;League, Serie&nbsp;A and Ligue&nbsp;1, all
      season <b>2015/2016</b>, for a total of <b>{total_incl:,} matches</b>. Every figure below is generated
      from <code>reports/competition_audit.csv</code>.</p>
    </header>
    """

    css = """
    <style>
      body{margin:0;background:#f4f6fb;color:#1f2430;
           font-family:system-ui,-apple-system,Segoe UI,Roboto,sans-serif;line-height:1.6}
      .wrap{max-width:1060px;margin:0 auto;padding:32px 20px 80px}
      .eyebrow{font-family:ui-monospace,monospace;font-size:.72rem;letter-spacing:.18em;
               text-transform:uppercase;color:#5b45e0;margin:0 0 10px}
      h1{font-family:Georgia,serif;font-size:2rem;margin:0 0 12px;letter-spacing:-.01em}
      .lede{font-size:1.04rem;color:#48506a;max-width:70ch}
      code{background:#e7eaf5;border-radius:5px;padding:1px 5px;font-size:.92em}
      .card{background:#fff;border:1px solid #e0e4f0;border-radius:14px;padding:8px 10px 14px;
            margin:22px 0;box-shadow:0 18px 40px -34px rgba(40,50,90,.5)}
      footer{color:#7982a2;font-size:.85rem;margin-top:40px;border-top:1px dashed #cfd6e6;padding-top:16px}
    </style>
    """

    html = ("<!DOCTYPE html><html lang='en'><head><meta charset='utf-8'>"
            "<meta name='viewport' content='width=device-width,initial-scale=1'>"
            "<title>Competition Selection Audit</title>" +
            css + "</head><body><div class='wrap'>"
            + intro
            + "".join(f"<div class='card'>{b}</div>" for b in blocks)
            + "<footer>Source: StatsBomb Open Data (matches metadata only). "
              "Generated by <code>scripts/00_audit_competitions.py</code>. "
              "Odds coverage for the four selected leagues must be confirmed against "
              "Football-Data.co.uk season CSVs before this selection is final.</footer>"
            + "</div></body></html>")

    out.write_text(html, encoding="utf-8")


# --------------------------------------------------------------------------
# 5. Main
# --------------------------------------------------------------------------
def main() -> None:
    REPORTS_DIR.mkdir(exist_ok=True)

    df = build_audit(ROOT)
    df["decision"] = df.apply(categorise, axis=1)

    # self-check: the rule must select exactly the four 2015/16 leagues
    got = set(map(tuple, df.loc[df["decision"].str.startswith("Included"),
                                ["cid", "sid"]].values.tolist()))
    expected = {(11, 27), (2, 27), (12, 27), (7, 27)}
    assert got == expected, f"Competition selection drifted: {sorted(got)}"

    df.to_csv(REPORTS_DIR / "competition_audit.csv", index=False)
    (REPORTS_DIR / "visualizations").mkdir(parents=True, exist_ok=True)
    write_report(df, REPORTS_DIR / "visualizations" / "competition_audit.html")

    print(df.drop(columns=["date_min", "date_max"]).to_string(index=False))
    print(f"\nIncluded: {len(got)} leagues, "
          f"{int(df.loc[df['decision'].str.startswith('Included'), 'n_matches'].sum()):,} matches")
    print(f"Wrote {REPORTS_DIR/'competition_audit.csv'}")
    print(f"Wrote {REPORTS_DIR / 'visualizations' / 'competition_audit.html'}")


if __name__ == "__main__":
    main()
