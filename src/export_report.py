"""Export the Markdown final report to a submission-ready PDF."""

from __future__ import annotations

import argparse
import html
import re
from pathlib import Path

from reportlab.lib import colors
from reportlab.lib.enums import TA_CENTER
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
from reportlab.lib.units import cm
from reportlab.platypus import (
    Image,
    KeepTogether,
    PageBreak,
    Paragraph,
    SimpleDocTemplate,
    Spacer,
    Table,
    TableStyle,
)

from config import PROJECT_ROOT


REPORT_PATH = PROJECT_ROOT / "reports" / "final_report.md"
PDF_PATH = PROJECT_ROOT / "reports" / "Pearls_AQI_Predictor_Final_Report.pdf"


def inline_markup(text: str) -> str:
    text = html.escape(text)
    text = re.sub(r"`([^`]+)`", r"<font name='Courier'>\1</font>", text)
    text = re.sub(r"\*\*([^*]+)\*\*", r"<b>\1</b>", text)
    text = re.sub(r"\[([^\]]+)\]\(([^)]+)\)", r"<link href='\2' color='#2563eb'>\1</link>", text)
    return text


def build_styles() -> dict[str, ParagraphStyle]:
    base = getSampleStyleSheet()
    return {
        "title": ParagraphStyle(
            "ReportTitle",
            parent=base["Title"],
            fontName="Helvetica-Bold",
            fontSize=24,
            leading=30,
            alignment=TA_CENTER,
            textColor=colors.HexColor("#172033"),
            spaceAfter=18,
        ),
        "subtitle": ParagraphStyle(
            "ReportSubtitle",
            parent=base["Normal"],
            fontSize=13,
            leading=18,
            alignment=TA_CENTER,
            textColor=colors.HexColor("#657184"),
        ),
        "h2": ParagraphStyle(
            "Heading2Custom",
            parent=base["Heading2"],
            fontName="Helvetica-Bold",
            fontSize=15,
            leading=19,
            textColor=colors.HexColor("#0f766e"),
            spaceBefore=12,
            spaceAfter=7,
        ),
        "h3": ParagraphStyle(
            "Heading3Custom",
            parent=base["Heading3"],
            fontName="Helvetica-Bold",
            fontSize=12,
            leading=16,
            textColor=colors.HexColor("#172033"),
            spaceBefore=9,
            spaceAfter=5,
        ),
        "body": ParagraphStyle(
            "BodyCustom",
            parent=base["BodyText"],
            fontName="Helvetica",
            fontSize=9.5,
            leading=14,
            textColor=colors.HexColor("#273449"),
            spaceAfter=7,
        ),
        "bullet": ParagraphStyle(
            "BulletCustom",
            parent=base["BodyText"],
            fontName="Helvetica",
            fontSize=9.5,
            leading=14,
            leftIndent=14,
            firstLineIndent=-8,
            bulletIndent=4,
            textColor=colors.HexColor("#273449"),
            spaceAfter=3,
        ),
        "code": ParagraphStyle(
            "CodeCustom",
            parent=base["Code"],
            fontName="Courier",
            fontSize=7.8,
            leading=10,
            leftIndent=8,
            rightIndent=8,
            backColor=colors.HexColor("#f1f5f9"),
            borderColor=colors.HexColor("#d9e2ef"),
            borderWidth=0.5,
            borderPadding=7,
            spaceAfter=8,
        ),
        "caption": ParagraphStyle(
            "CaptionCustom",
            parent=base["Normal"],
            fontSize=8,
            leading=11,
            alignment=TA_CENTER,
            textColor=colors.HexColor("#657184"),
            spaceAfter=8,
        ),
    }


def parse_table(lines: list[str], styles: dict[str, ParagraphStyle]) -> Table:
    rows = []
    for index, line in enumerate(lines):
        if index == 1:
            continue
        cells = [cell.strip() for cell in line.strip().strip("|").split("|")]
        rows.append([Paragraph(inline_markup(cell), styles["body"]) for cell in cells])

    column_count = max(len(row) for row in rows)
    usable_width = A4[0] - 3.4 * cm
    table = Table(rows, colWidths=[usable_width / column_count] * column_count, repeatRows=1)
    table.setStyle(
        TableStyle(
            [
                ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#0f766e")),
                ("TEXTCOLOR", (0, 0), (-1, 0), colors.white),
                ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
                ("GRID", (0, 0), (-1, -1), 0.4, colors.HexColor("#cbd5e1")),
                ("BACKGROUND", (0, 1), (-1, -1), colors.white),
                ("VALIGN", (0, 0), (-1, -1), "TOP"),
                ("LEFTPADDING", (0, 0), (-1, -1), 5),
                ("RIGHTPADDING", (0, 0), (-1, -1), 5),
                ("TOPPADDING", (0, 0), (-1, -1), 4),
                ("BOTTOMPADDING", (0, 0), (-1, -1), 4),
            ]
        )
    )
    return table


def image_flowable(report_dir: Path, alt_text: str, image_path: str, styles: dict[str, ParagraphStyle]):
    path = report_dir / image_path
    if not path.exists():
        return Paragraph(f"Missing image: {inline_markup(image_path)}", styles["body"])

    image = Image(str(path))
    max_width = A4[0] - 4 * cm
    max_height = 14 * cm
    scale = min(max_width / image.imageWidth, max_height / image.imageHeight, 1)
    image.drawWidth = image.imageWidth * scale
    image.drawHeight = image.imageHeight * scale
    return KeepTogether([image, Paragraph(inline_markup(alt_text), styles["caption"])])


def markdown_to_flowables(markdown_text: str, report_dir: Path, styles: dict[str, ParagraphStyle]):
    lines = markdown_text.splitlines()
    flowables = []
    index = 0
    in_code = False
    code_lines: list[str] = []

    while index < len(lines):
        line = lines[index]
        stripped = line.strip()

        if stripped.startswith("```"):
            if in_code:
                flowables.append(Paragraph("<br/>".join(html.escape(item) for item in code_lines), styles["code"]))
                code_lines = []
                in_code = False
            else:
                in_code = True
            index += 1
            continue

        if in_code:
            code_lines.append(line)
            index += 1
            continue

        if not stripped:
            index += 1
            continue

        image_match = re.fullmatch(r"!\[([^\]]*)\]\(([^)]+)\)", stripped)
        if image_match:
            flowables.append(image_flowable(report_dir, image_match.group(1), image_match.group(2), styles))
            index += 1
            continue

        if stripped.startswith("|") and index + 1 < len(lines) and re.match(r"^\|\s*[-:]+", lines[index + 1]):
            table_lines = [line, lines[index + 1]]
            index += 2
            while index < len(lines) and lines[index].strip().startswith("|"):
                table_lines.append(lines[index])
                index += 1
            flowables.append(parse_table(table_lines, styles))
            flowables.append(Spacer(1, 8))
            continue

        if stripped.startswith("# "):
            index += 1
            continue
        if stripped.startswith("## "):
            flowables.append(Paragraph(inline_markup(stripped[3:]), styles["h2"]))
            index += 1
            continue
        if stripped.startswith("### "):
            flowables.append(Paragraph(inline_markup(stripped[4:]), styles["h3"]))
            index += 1
            continue
        if stripped.startswith("- "):
            flowables.append(Paragraph(inline_markup(stripped[2:]), styles["bullet"], bulletText="•"))
            index += 1
            continue

        paragraph_lines = [stripped]
        index += 1
        while index < len(lines):
            candidate = lines[index].strip()
            if not candidate or candidate.startswith(("#", "-", "|", "```", "![")):
                break
            paragraph_lines.append(candidate)
            index += 1
        flowables.append(Paragraph(inline_markup(" ".join(paragraph_lines)), styles["body"]))

    return flowables


def page_number(canvas, document) -> None:
    canvas.saveState()
    canvas.setFont("Helvetica", 8)
    canvas.setFillColor(colors.HexColor("#657184"))
    canvas.drawCentredString(A4[0] / 2, 0.8 * cm, f"Page {document.page}")
    canvas.restoreState()


def export_report(markdown_path: Path, output_path: Path) -> None:
    styles = build_styles()
    document = SimpleDocTemplate(
        str(output_path),
        pagesize=A4,
        rightMargin=1.7 * cm,
        leftMargin=1.7 * cm,
        topMargin=1.6 * cm,
        bottomMargin=1.5 * cm,
        title="Pearls AQI Predictor Final Report",
        author="Pearls AQI Predictor Project",
    )

    story = [
        Spacer(1, 4.2 * cm),
        Paragraph("Pearls AQI Predictor", styles["title"]),
        Paragraph("Internship Project Final Report", styles["subtitle"]),
        Spacer(1, 0.7 * cm),
        Paragraph("Karachi Air Quality Forecasting for 24, 48, and 72 Hours", styles["subtitle"]),
        Spacer(1, 1.4 * cm),
        Paragraph("Final submission: June 7, 2026", styles["subtitle"]),
        PageBreak(),
    ]
    markdown_text = markdown_path.read_text(encoding="utf-8")
    story.extend(markdown_to_flowables(markdown_text, markdown_path.parent, styles))
    document.build(story, onFirstPage=page_number, onLaterPages=page_number)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Export the final report to PDF.")
    parser.add_argument("--input", type=Path, default=REPORT_PATH)
    parser.add_argument("--output", type=Path, default=PDF_PATH)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    export_report(args.input, args.output)
    print(f"Saved PDF report to {args.output}")


if __name__ == "__main__":
    main()
