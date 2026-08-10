// Phase 10: render the same report content model to DOCX.
//
// Reads src/reports/report_build/report_content.json, which build_report.py
// writes from report_content.py. The PDF and the DOCX therefore carry
// identical prose and identical numbers by construction.
//
//   node src/report/render_docx.js

const fs = require("fs");
const path = require("path");
const {
  AlignmentType, Document, Footer, HeadingLevel, ImageRun, PageBreak,
  PageNumber, Packer, Paragraph, ShadingType, Table, TableCell, TableRow,
  TextRun, WidthType,
} = require("docx");

const HERE = __dirname;
const RESULTS_DIR = path.join(HERE, "..", "reports");
const CONTENT_JSON = path.join(RESULTS_DIR, "report_build",
                               "report_content.json");
const OUTPUT_DOCX = path.join(RESULTS_DIR, "final_report.docx");

// A4 is 11906 DXA wide; 2 cm margins leave 9638 for content.
const CONTENT_WIDTH_DXA = 9638;
const CONTENT_WIDTH_CM = 17.0;
const EMU_PER_CM = 360000;
const SERIF = "Times New Roman";

// --- Inline markup ----------------------------------------------------------
// The content model uses only <b> and <code>, so a small splitter is enough.
function runs(text, base = {}) {
  const pieces = text.split(/(<b>|<\/b>|<code>|<\/code>)/);
  const out = [];
  let bold = false;
  let code = false;
  for (const piece of pieces) {
    if (piece === "<b>") { bold = true; continue; }
    if (piece === "</b>") { bold = false; continue; }
    if (piece === "<code>") { code = true; continue; }
    if (piece === "</code>") { code = false; continue; }
    if (!piece) continue;
    out.push(new TextRun({
      text: piece,
      bold: bold || base.bold === true,
      font: code ? "Courier New" : (base.font || SERIF),
      size: code ? 16 : (base.size || 19),
      color: base.color,
    }));
  }
  if (out.length === 0) out.push(new TextRun({ text: "", font: SERIF }));
  return out;
}

// --- Images -----------------------------------------------------------------
function imageParagraph(block) {
  const widthPx = block.width_px || 1000;
  const heightPx = block.height_px || 400;
  const maxHeightCm = block.max_height_cm || 11.0;
  const ratio = Math.min(CONTENT_WIDTH_CM / (widthPx / 96 * 2.54),
                         maxHeightCm / (heightPx / 96 * 2.54));
  const widthCm = (widthPx / 96 * 2.54) * ratio;
  const heightCm = (heightPx / 96 * 2.54) * ratio;
  return new Paragraph({
    alignment: block.centered ? AlignmentType.CENTER : AlignmentType.LEFT,
    spacing: { before: 120, after: 120 },
    children: [new ImageRun({
      type: "png",
      data: fs.readFileSync(block.path),
      transformation: {
        width: Math.round(widthCm * EMU_PER_CM / 12700),
        height: Math.round(heightCm * EMU_PER_CM / 12700),
      },
    })],
  });
}

// --- Tables -----------------------------------------------------------------
function buildTable(block) {
  const columns = block.header.length;
  const base = Math.floor(CONTENT_WIDTH_DXA / columns);
  const widths = new Array(columns).fill(base);
  widths[columns - 1] = CONTENT_WIDTH_DXA - base * (columns - 1);

  const cell = (text, isHeader, width) => new TableCell({
    width: { size: width, type: WidthType.DXA },
    shading: isHeader
      ? { type: ShadingType.CLEAR, fill: "E8E8E8" }
      : undefined,
    margins: { top: 40, bottom: 40, left: 70, right: 70 },
    children: [new Paragraph({
      spacing: { before: 0, after: 0 },
      children: [new TextRun({
        text: String(text), bold: isHeader, font: SERIF, size: 15,
      })],
    })],
  });

  const rows = [new TableRow({
    tableHeader: true,
    children: block.header.map((h, i) => cell(h, true, widths[i])),
  })];
  for (const row of block.rows) {
    rows.push(new TableRow({
      children: row.map((v, i) => cell(v, false, widths[i])),
    }));
  }
  return new Table({
    columnWidths: widths,
    width: { size: CONTENT_WIDTH_DXA, type: WidthType.DXA },
    rows,
  });
}

// --- Block dispatch ---------------------------------------------------------
function toChildren(blocks) {
  const children = [];
  for (const block of blocks) {
    switch (block.type) {
      case "title":
        children.push(new Paragraph({ spacing: { before: 2400, after: 0 } }));
        children.push(new Paragraph({
          alignment: AlignmentType.CENTER,
          spacing: { after: 200 },
          children: [new TextRun({
            text: block.text, bold: true, font: SERIF, size: 38,
          })],
        }));
        children.push(new Paragraph({
          alignment: AlignmentType.CENTER,
          spacing: { after: 600 },
          children: [new TextRun({
            text: block.subtitle, font: SERIF, size: 24,
          })],
        }));
        break;
      case "h1":
        children.push(new Paragraph({
          heading: HeadingLevel.HEADING_1,
          spacing: { before: 280, after: 140 },
          children: runs(block.text, { bold: true, size: 28, color: "000000" }),
        }));
        break;
      case "h2":
        children.push(new Paragraph({
          heading: HeadingLevel.HEADING_2,
          spacing: { before: 220, after: 110 },
          children: runs(block.text, { bold: true, size: 23, color: "000000" }),
        }));
        break;
      case "h3":
        children.push(new Paragraph({
          heading: HeadingLevel.HEADING_3,
          spacing: { before: 180, after: 90 },
          children: runs(block.text, { bold: true, size: 21, color: "000000" }),
        }));
        break;
      case "para":
        children.push(new Paragraph({
          alignment: AlignmentType.JUSTIFIED,
          spacing: { after: 130, line: 276 },
          children: runs(block.text),
        }));
        break;
      case "caption":
        children.push(new Paragraph({
          alignment: AlignmentType.JUSTIFIED,
          spacing: { before: 60, after: 200 },
          children: runs(block.text, { size: 17, color: "333333" }),
        }));
        break;
      case "mono":
        children.push(new Paragraph({
          spacing: { after: 0 },
          children: [new TextRun({
            text: block.text, font: "Courier New", size: 15,
          })],
        }));
        break;
      case "pagebreak":
        children.push(new Paragraph({ children: [new PageBreak()] }));
        break;
      case "image":
        if (block.path && fs.existsSync(block.path)) {
          children.push(imageParagraph(block));
          if (block.caption) {
            children.push(new Paragraph({
              alignment: AlignmentType.JUSTIFIED,
              spacing: { after: 200 },
              children: runs(block.caption, { size: 17, color: "333333" }),
            }));
          }
        }
        break;
      case "table":
        children.push(buildTable(block));
        if (block.caption) {
          children.push(new Paragraph({
            alignment: AlignmentType.JUSTIFIED,
            spacing: { before: 80, after: 200 },
            children: runs(block.caption, { size: 17, color: "333333" }),
          }));
        } else {
          children.push(new Paragraph({ spacing: { after: 160 } }));
        }
        break;
      default:
        break;
    }
  }
  return children;
}

function main() {
  if (!fs.existsSync(CONTENT_JSON)) {
    console.error(`Missing ${CONTENT_JSON}. Run build_report.py first.`);
    process.exit(1);
  }
  const payload = JSON.parse(fs.readFileSync(CONTENT_JSON, "utf-8"));

  const footer = new Footer({
    children: [new Paragraph({
      children: [
        new TextRun({
          text: "Forecasting Competitive Football - final report    ",
          font: SERIF, size: 15, color: "555555",
        }),
        new TextRun({ children: [PageNumber.CURRENT], font: SERIF, size: 15,
                      color: "555555" }),
      ],
    })],
  });

  const document = new Document({
    creator: "Forecasting Competitive Football",
    title: "Forecasting Competitive Football",
    styles: {
      default: {
        document: { run: { font: SERIF, size: 19 } },
      },
    },
    sections: [{
      properties: {
        page: { margin: { top: 1020, bottom: 1020, left: 1134, right: 1134 } },
      },
      footers: { default: footer },
      children: toChildren(payload.blocks),
    }],
  });

  Packer.toBuffer(document).then((buffer) => {
    fs.writeFileSync(OUTPUT_DOCX, buffer);
    console.log(`Wrote -> ${OUTPUT_DOCX}`);
    console.log(`        ${payload.blocks.length} blocks `
                + `(${payload.body_blocks} in the body)`);
  });
}

main();
