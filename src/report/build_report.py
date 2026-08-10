"""Phase 10: render the report content to PDF, and export it for the DOCX build.

All prose and every number live in report_content.py. This file only draws.

    python src/report/build_report.py
"""

import html
import json
import re

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from PIL import Image as PILImage

from reportlab.lib import colors
from reportlab.lib.enums import TA_JUSTIFY
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
from reportlab.lib.units import cm
from reportlab.platypus import (Image, KeepTogether, PageBreak, Paragraph,
                                SimpleDocTemplate, Spacer, Table, TableStyle)

from report_content import BUILD_DIR, RESULTS_DIR, build_blocks

OUTPUT_PDF = RESULTS_DIR / "final_report.pdf"
CONTENT_JSON = BUILD_DIR / "report_content.json"
PAGE_WIDTH = A4[0] - 4 * cm

styles = getSampleStyleSheet()
BODY = ParagraphStyle("body", parent=styles["Normal"], fontName="Times-Roman",
                      fontSize=9.6, leading=13.4, alignment=TA_JUSTIFY,
                      spaceAfter=6)
H1 = ParagraphStyle("h1", parent=styles["Heading1"], fontName="Times-Bold",
                    fontSize=14, leading=17, spaceBefore=13, spaceAfter=7)
H2 = ParagraphStyle("h2", parent=styles["Heading2"], fontName="Times-Bold",
                    fontSize=11.5, leading=14, spaceBefore=10, spaceAfter=5)
H3 = ParagraphStyle("h3", parent=styles["Heading3"], fontName="Times-Bold",
                    fontSize=10.2, leading=13, spaceBefore=8, spaceAfter=4)
CAPTION = ParagraphStyle("caption", parent=BODY, fontSize=8.3, leading=10.6,
                         textColor=colors.HexColor("#333333"), spaceAfter=10)
CELL = ParagraphStyle("cell", parent=BODY, fontSize=7.6, leading=9.4,
                      alignment=0, spaceAfter=0)
MONO = ParagraphStyle("mono", parent=styles["Normal"], fontName="Courier",
                      fontSize=7.2, leading=8.9)
TITLE = ParagraphStyle("title", parent=styles["Title"], fontName="Times-Bold",
                       fontSize=19, leading=23)
SUBTITLE = ParagraphStyle("subtitle", parent=BODY, fontSize=12, leading=16,
                          alignment=1)


# --- Inline markup ----------------------------------------------------------
def inline(text):
    parts = re.split(r"(<b>|</b>|<code>|</code>)", text)
    out = []
    for part in parts:
        if part == "<b>":
            out.append("<b>")
        elif part == "</b>":
            out.append("</b>")
        elif part == "<code>":
            out.append("<font face='Courier' size='8'>")
        elif part == "</code>":
            out.append("</font>")
        else:
            out.append(html.escape(part))
    return "".join(out)


# --- Figures ----------------------------------------------------------------
def scaled(path, max_width=PAGE_WIDTH, max_height=20 * cm):
    with PILImage.open(path) as handle:
        width, height = handle.size
    ratio = min(max_width / width, max_height / height)
    return Image(str(path), width=width * ratio, height=height * ratio)


def render_equation(block):
    BUILD_DIR.mkdir(parents=True, exist_ok=True)
    path = BUILD_DIR / f"eq_{block['name']}.png"
    figure = plt.figure(figsize=(7.2, 0.85))
    figure.text(0.5, 0.5, block["latex"], fontsize=15, ha="center", va="center")
    figure.savefig(path, dpi=220, bbox_inches="tight", transparent=True)
    plt.close(figure)
    block["rendered_path"] = str(path)
    return scaled(path, max_width=PAGE_WIDTH * 0.78, max_height=1.15 * cm)


def render_lineplot(block):
    BUILD_DIR.mkdir(parents=True, exist_ok=True)
    path = BUILD_DIR / f"{block['name']}.png"
    frame = block["frame"]
    figure, axis = plt.subplots(figsize=(7.0, 3.2))
    for key, group in frame.groupby(block["series"]):
        group = group.sort_values(block["x"])
        dashed = block.get("dashed") and block["dashed"] in str(key)
        axis.plot(group[block["x"]], group[block["y"]], marker="o",
                  markersize=2.6, linewidth=1.4,
                  linestyle="--" if dashed else "-", label=str(key))
    if block.get("logx"):
        axis.set_xscale("log")
    if block.get("logy"):
        axis.set_yscale("log")
    axis.set_xlabel(block["xlabel"], fontsize=8)
    axis.set_ylabel(block["ylabel"], fontsize=8)
    axis.set_title(block["title"], fontsize=9)
    axis.tick_params(labelsize=7)
    axis.grid(alpha=0.3)
    axis.legend(fontsize=6, ncol=2)
    figure.tight_layout()
    figure.savefig(path, dpi=170)
    plt.close(figure)
    block["rendered_path"] = str(path)
    return scaled(path)


def render_table(block):
    data = [[Paragraph(f"<b>{html.escape(c)}</b>", CELL)
             for c in block["header"]]]
    for row in block["rows"]:
        data.append([Paragraph(html.escape(str(v)), CELL) for v in row])
    table = Table(data, repeatRows=1, hAlign="LEFT")
    table.setStyle(TableStyle([
        ("GRID", (0, 0), (-1, -1), 0.3, colors.HexColor("#999999")),
        ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#e8e8e8")),
        ("VALIGN", (0, 0), (-1, -1), "TOP"),
        ("LEFTPADDING", (0, 0), (-1, -1), 3),
        ("RIGHTPADDING", (0, 0), (-1, -1), 3),
        ("TOPPADDING", (0, 0), (-1, -1), 2),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 2)]))
    return table


# --- Build ------------------------------------------------------------------
def to_flowables(blocks):
    story = []
    for block in blocks:
        kind = block["type"]
        if kind == "title":
            story.append(Spacer(1, 3.2 * cm))
            story.append(Paragraph(inline(block["text"]), TITLE))
            story.append(Spacer(1, 0.4 * cm))
            story.append(Paragraph(inline(block["subtitle"]), SUBTITLE))
            story.append(Spacer(1, 1.1 * cm))
        elif kind == "h1":
            story.append(Paragraph(inline(block["text"]), H1))
        elif kind == "h2":
            story.append(Paragraph(inline(block["text"]), H2))
        elif kind == "h3":
            story.append(Paragraph(inline(block["text"]), H3))
        elif kind == "para":
            story.append(Paragraph(inline(block["text"]), BODY))
        elif kind == "caption":
            story.append(Paragraph(inline(block["text"]), CAPTION))
        elif kind == "mono":
            story.append(Paragraph(html.escape(block["text"]), MONO))
        elif kind == "pagebreak":
            story.append(PageBreak())
        elif kind == "equation":
            story.append(render_equation(block))
        elif kind == "lineplot":
            story.append(render_lineplot(block))
        elif kind == "image":
            flow = [scaled(block["path"],
                           max_height=block.get("max_height_cm", 11.0) * cm)]
            if block.get("caption"):
                flow.append(Spacer(1, 3))
                flow.append(Paragraph(inline(block["caption"]), CAPTION))
            story.append(KeepTogether(flow))
        elif kind == "table":
            story.append(render_table(block))
            if block.get("caption"):
                story.append(Spacer(1, 3))
                story.append(Paragraph(inline(block["caption"]), CAPTION))
    return story


def _pixel_size(path):
    with PILImage.open(path) as handle:
        return handle.size


def export_json(blocks, body_length):
    payload = []
    for block in blocks:
        if block["type"] in {"equation", "lineplot"}:
            item = {"type": "image", "path": block.get("rendered_path"),
                    "max_height_cm": 1.2 if block["type"] == "equation" else 9.0,
                    "centered": block["type"] == "equation"}
        else:
            item = {k: v for k, v in block.items() if k != "frame"}
        if item.get("type") == "image" and item.get("path"):
            width, height = _pixel_size(item["path"])
            item["width_px"], item["height_px"] = width, height
        payload.append(item)
    CONTENT_JSON.parent.mkdir(parents=True, exist_ok=True)
    CONTENT_JSON.write_text(
        json.dumps({"body_blocks": body_length, "blocks": payload}, indent=1),
        encoding="utf-8")
    return CONTENT_JSON


def _footer(canvas, document):
    canvas.saveState()
    canvas.setFont("Times-Roman", 7.5)
    canvas.setFillColor(colors.HexColor("#555555"))
    canvas.drawRightString(A4[0] - 2 * cm, 1.1 * cm, str(document.page))
    canvas.drawString(2 * cm, 1.1 * cm,
                      "Forecasting Competitive Football - final report")
    canvas.restoreState()


def build():
    blocks, body_length = build_blocks()
    story = to_flowables(blocks)
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    document = SimpleDocTemplate(
        str(OUTPUT_PDF), pagesize=A4,
        leftMargin=2 * cm, rightMargin=2 * cm,
        topMargin=1.8 * cm, bottomMargin=1.8 * cm,
        title="Forecasting Competitive Football",
        author="Final project report")
    document.build(story, onFirstPage=_footer, onLaterPages=_footer)
    path = export_json(blocks, body_length)
    print(f"Wrote -> {OUTPUT_PDF}")
    print(f"Wrote -> {path}  ({body_length} body blocks, {len(blocks)} total)")
    return OUTPUT_PDF


if __name__ == "__main__":
    build()
