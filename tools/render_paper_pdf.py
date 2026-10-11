"""Render the Vastloom streaming-inference working paper from Markdown to PDF.

Usage:
    python tools/render_paper_pdf.py [input.md] [output.pdf]

Defaults to docs/research/CONSUMER_STREAMING_INFERENCE.md and
docs/research/vastloom-streaming-moe-inference.pdf. Uses the Cambria family
(bundled with Windows) so the paper's mathematical glyphs render unchanged.
"""

from __future__ import annotations

import html
import re
import argparse
from pathlib import Path

from reportlab.lib import colors
from reportlab.lib.enums import TA_CENTER, TA_LEFT, TA_RIGHT
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import ParagraphStyle
from reportlab.lib.units import mm
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont
from reportlab.platypus import (
    HRFlowable,
    Paragraph,
    SimpleDocTemplate,
    Spacer,
    Table,
    TableStyle,
)

REPO_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_INPUT = REPO_ROOT / "docs" / "research" / "CONSUMER_STREAMING_INFERENCE.md"
DEFAULT_OUTPUT = REPO_ROOT / "docs" / "research" / "vastloom-streaming-moe-inference.pdf"

FONT_DIR = Path(r"C:\Windows\Fonts")


def register_fonts() -> None:
    pdfmetrics.registerFont(TTFont("Cambria", str(FONT_DIR / "cambria.ttc"), subfontIndex=0))
    pdfmetrics.registerFont(TTFont("Cambria-Bold", str(FONT_DIR / "cambriab.ttf")))
    pdfmetrics.registerFont(TTFont("Cambria-Italic", str(FONT_DIR / "cambriai.ttf")))
    pdfmetrics.registerFont(TTFont("Cambria-BoldItalic", str(FONT_DIR / "cambriaz.ttf")))
    pdfmetrics.registerFontFamily(
        "Cambria",
        normal="Cambria",
        bold="Cambria-Bold",
        italic="Cambria-Italic",
        boldItalic="Cambria-BoldItalic",
    )


def inline_markup(text: str) -> str:
    text = html.escape(text, quote=False)
    text = re.sub(r"\*\*(.+?)\*\*", r"<b>\1</b>", text)
    text = re.sub(r"(?<!\*)\*([^*]+?)\*(?!\*)", r"<i>\1</i>", text)
    text = re.sub(r"`([^`]+?)`", r'<font face="Courier" size="8.5">\1</font>', text)
    return text


def make_styles() -> dict[str, ParagraphStyle]:
    body = dict(fontName="Cambria", fontSize=10, leading=13.8, alignment=TA_LEFT)
    return {
        "title": ParagraphStyle(
            "title", fontName="Cambria-Bold", fontSize=16.5, leading=21,
            alignment=TA_CENTER, spaceAfter=10,
        ),
        "author": ParagraphStyle(
            "author", fontName="Cambria", fontSize=11, leading=14,
            alignment=TA_CENTER, spaceAfter=4,
        ),
        "note": ParagraphStyle(
            "note", fontName="Cambria-Italic", fontSize=9.5, leading=12.5,
            alignment=TA_CENTER, spaceAfter=6, textColor=colors.HexColor("#333333"),
        ),
        "h2": ParagraphStyle(
            "h2", fontName="Cambria-Bold", fontSize=12.5, leading=16,
            spaceBefore=16, spaceAfter=6, alignment=TA_LEFT,
        ),
        "h3": ParagraphStyle(
            "h3", fontName="Cambria-Bold", fontSize=10.8, leading=14,
            spaceBefore=11, spaceAfter=4, alignment=TA_LEFT,
        ),
        "body": ParagraphStyle("body", **body, spaceAfter=6),
        "italic_note": ParagraphStyle(
            "italic_note", fontName="Cambria-Italic", fontSize=9.3, leading=12.4,
            alignment=TA_LEFT, textColor=colors.HexColor("#222222"), spaceAfter=6,
        ),
        "bullet": ParagraphStyle(
            "bullet", **body, leftIndent=14, bulletIndent=4, spaceAfter=3,
        ),
        "caption": ParagraphStyle(
            "caption", fontName="Cambria-Bold", fontSize=9.3, leading=12.4,
            spaceBefore=8, spaceAfter=3, alignment=TA_LEFT,
        ),
        "table_caption": ParagraphStyle(
            "table_caption", fontName="Cambria-Bold", fontSize=9.3, leading=12,
            alignment=TA_LEFT,
        ),
        "table_cell": ParagraphStyle(
            "table_cell", fontName="Cambria", fontSize=8.8, leading=11.4,
            alignment=TA_LEFT,
        ),
        "table_cell_right": ParagraphStyle(
            "table_cell_right", fontName="Cambria", fontSize=8.8, leading=11.4,
            alignment=TA_RIGHT,
        ),
        "table_header": ParagraphStyle(
            "table_header", fontName="Cambria-Bold", fontSize=8.8, leading=11.4,
            alignment=TA_LEFT,
        ),
        "table_header_right": ParagraphStyle(
            "table_header_right", fontName="Cambria-Bold", fontSize=8.8, leading=11.4,
            alignment=TA_RIGHT,
        ),
    }


def parse_table(lines: list[str], start: int) -> tuple[list[list[str]], list[str], int]:
    header = [c.strip() for c in lines[start].strip().strip("|").split("|")]
    align = []
    for cell in [c.strip() for c in lines[start + 1].strip().strip("|").split("|")]:
        align.append("right" if cell.endswith(":") else "left")
    rows: list[list[str]] = []
    index = start + 2
    while index < len(lines) and lines[index].lstrip().startswith("|"):
        rows.append([c.strip() for c in lines[index].strip().strip("|").split("|")])
        index += 1
    return ([header] + rows), align, index


def build_table(rows: list[list[str]], align: list[str], styles: dict) -> Table:
    table_data = []
    for row_index, row in enumerate(rows):
        rendered = []
        for col_index, cell in enumerate(row):
            right = col_index < len(align) and align[col_index] == "right"
            if row_index == 0:
                style = styles["table_header_right" if right else "table_header"]
            else:
                style = styles["table_cell_right" if right else "table_cell"]
            rendered.append(Paragraph(inline_markup(cell), style))
        table_data.append(rendered)
    table = Table(table_data, hAlign="LEFT")
    table.setStyle(TableStyle([
        ("LINEBELOW", (0, 0), (-1, 0), 0.8, colors.black),
        ("LINEBELOW", (0, -1), (-1, -1), 0.8, colors.black),
        ("TOPPADDING", (0, 0), (-1, -1), 2.4),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 2.4),
        ("LEFTPADDING", (0, 0), (-1, -1), 5),
        ("RIGHTPADDING", (0, 0), (-1, -1), 5),
        ("VALIGN", (0, 0), (-1, -1), "TOP"),
    ]))
    return table


def parse_blocks(markdown: str):
    lines = markdown.splitlines()
    blocks: list[tuple[str, object]] = []
    index = 0
    while index < len(lines):
        line = lines[index]
        stripped = line.strip()
        if not stripped:
            index += 1
            continue
        if stripped == "---":
            blocks.append(("hr", None))
            index += 1
            continue
        if stripped.startswith("# "):
            blocks.append(("title", stripped[2:]))
            index += 1
            continue
        if stripped.startswith("### "):
            blocks.append(("h3", stripped[4:]))
            index += 1
            continue
        if stripped.startswith("## "):
            blocks.append(("h2", stripped[3:]))
            index += 1
            continue
        if stripped.startswith("|"):
            rows, align, index = parse_table(lines, index)
            blocks.append(("table", (rows, align)))
            continue
        if re.match(r"^[-*] ", stripped) or re.match(r"^\d+\. ", stripped):
            ordered = bool(re.match(r"^\d+\. ", stripped))
            marker = re.match(r"^(\d+)\. ", stripped)
            bullet = f"{marker.group(1)}. " if ordered else "•"
            text = re.sub(r"^([-*] |\d+\. )", "", stripped)
            index += 1
            while index < len(lines):
                cont = lines[index]
                if cont.startswith("   ") and cont.strip():
                    text += " " + cont.strip()
                    index += 1
                else:
                    break
            blocks.append(("bullet", (bullet, text)))
            continue
        text = stripped
        index += 1
        while index < len(lines):
            nxt = lines[index]
            nxts = nxt.strip()
            if (not nxts or nxts.startswith("#") or nxts.startswith("|")
                    or nxts == "---" or re.match(r"^[-*] ", nxts)
                    or re.match(r"^\d+\. ", nxts) or nxt.startswith("   ")):
                break
            text += " " + nxts
            index += 1
        kind = "italic_note" if text.startswith("*") and text.endswith("*") and not text.startswith("**") else "body"
        blocks.append((kind, text))
    return blocks


def footer(canvas, doc) -> None:
    canvas.saveState()
    canvas.setFont("Cambria", 8)
    canvas.setFillColor(colors.HexColor("#555555"))
    canvas.drawString(doc.leftMargin, 14 * mm, "Vastloom — streaming MoE inference on a 32 GiB desktop")
    canvas.drawRightString(A4[0] - doc.rightMargin, 14 * mm, f"{canvas.getPageNumber()}")
    canvas.restoreState()


def render(input_path: Path, output_path: Path) -> None:
    register_fonts()
    styles = make_styles()
    markdown = input_path.read_text(encoding="utf-8")
    blocks = parse_blocks(markdown)

    story = []
    for kind, payload in blocks:
        if kind == "title":
            story.append(Paragraph(inline_markup(payload), styles["title"]))
        elif kind == "h2":
            story.append(Paragraph(inline_markup(payload), styles["h2"]))
        elif kind == "h3":
            story.append(Paragraph(inline_markup(payload), styles["h3"]))
        elif kind == "hr":
            story.append(Spacer(1, 3))
            story.append(HRFlowable(width="100%", thickness=0.7, color=colors.HexColor("#888888")))
            story.append(Spacer(1, 6))
        elif kind == "table":
            rows, align = payload
            story.append(Spacer(1, 4))
            story.append(build_table(rows, align, styles))
            story.append(Spacer(1, 8))
        elif kind == "bullet":
            bullet, text = payload
            story.append(
                Paragraph(inline_markup(text), styles["bullet"], bulletText=bullet)
            )
        elif kind == "italic_note":
            story.append(Paragraph(inline_markup(payload.strip("*")), styles["italic_note"]))
        else:
            text = payload
            if text.startswith("**Table"):
                story.append(Paragraph(inline_markup(text), styles["caption"]))
            elif "Vastloom / TheStreamCode" in text:
                story.append(Paragraph(inline_markup(text), styles["author"]))
            elif text.startswith("*") and text.endswith("*") and not text.startswith("**"):
                story.append(Paragraph(inline_markup(text.strip("*")), styles["italic_note"]))
            else:
                story.append(Paragraph(inline_markup(text), styles["body"]))

    doc = SimpleDocTemplate(
        str(output_path),
        pagesize=A4,
        leftMargin=22 * mm,
        rightMargin=22 * mm,
        topMargin=20 * mm,
        bottomMargin=20 * mm,
        title="Streaming Mixture-of-Experts Inference on a 32 GiB Desktop",
        author="Michael Gasperini",
    )
    doc.build(story, onFirstPage=footer, onLaterPages=footer)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("input", type=Path, nargs="?", default=Path(DEFAULT_INPUT),
                        help="markdown source (default: %(default)s)")
    parser.add_argument("output", type=Path, nargs="?", default=Path(DEFAULT_OUTPUT),
                        help="PDF destination (default: %(default)s)")
    arguments = parser.parse_args()
    render(arguments.input, arguments.output)
    print(f"rendered {arguments.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
