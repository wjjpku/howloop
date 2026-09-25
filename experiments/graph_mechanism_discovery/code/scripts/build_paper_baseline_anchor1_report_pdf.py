#!/usr/bin/env python3
"""Render the Chinese anchor-1 baseline report to a polished PDF."""

from __future__ import annotations

import argparse
import html
import re
from pathlib import Path

from PIL import Image as PILImage
from reportlab.lib import colors
from reportlab.lib.enums import TA_CENTER, TA_LEFT
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
from reportlab.lib.units import mm
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont
from reportlab.platypus import (
    HRFlowable,
    Image,
    KeepTogether,
    ListFlowable,
    ListItem,
    PageBreak,
    Paragraph,
    SimpleDocTemplate,
    Spacer,
    Table,
    TableStyle,
)


FONT_PATHS = [
    Path("/System/Library/AssetsV2/com_apple_MobileAsset_Font8/53fe5be564086fefc7523ccd0a31200acf92e0e5.asset/AssetData/STHEITI.ttf"),
    Path("/System/Library/Fonts/STHeiti Light.ttc"),
]


def register_fonts() -> str:
    font_path = next((path for path in FONT_PATHS if path.exists()), None)
    if font_path is None:
        raise FileNotFoundError("No Chinese report font was found")
    name = "BaselineReportCJK"
    pdfmetrics.registerFont(TTFont(name, str(font_path)))
    pdfmetrics.registerFontFamily(
        name, normal=name, bold=name, italic=name, boldItalic=name
    )
    return name


def rich_text(value: str, font_name: str) -> str:
    value = html.escape(value.strip())
    value = re.sub(
        r"\[([^\]]+)\]\((https?://[^)]+)\)",
        r'<link href="\2" color="#2563EB">\1</link>',
        value,
    )
    value = re.sub(r"\*\*([^*]+)\*\*", r"<b>\1</b>", value)
    value = re.sub(
        r"`([^`]+)`",
        rf'<font name="{font_name}" color="#334155">\1</font>',
        value,
    )
    value = value.replace("  ", " ")
    return value


def table_rows(lines: list[str], font_name: str) -> list[list[str]]:
    rows: list[list[str]] = []
    for line in lines:
        cells = [cell.strip() for cell in line.strip().strip("|").split("|")]
        if cells and all(re.fullmatch(r":?-{3,}:?", cell) for cell in cells):
            continue
        rows.append([rich_text(cell, font_name) for cell in cells])
    return rows


def char_weight(text: str) -> float:
    plain = re.sub(r"<[^>]+>", "", text)
    return sum(1.7 if ord(character) > 127 else 1.0 for character in plain)


def column_widths(rows: list[list[str]], total_width: float) -> list[float]:
    count = max(len(row) for row in rows)
    weights = []
    for column in range(count):
        values = [row[column] for row in rows if column < len(row)]
        weight = max(char_weight(value) for value in values)
        weights.append(max(4.0, min(weight, 22.0)))
    total = sum(weights)
    widths = [total_width * weight / total for weight in weights]
    minimum = 16 * mm
    if count <= 3:
        widths = [max(minimum, width) for width in widths]
        scale = total_width / sum(widths)
        widths = [width * scale for width in widths]
    return widths


def build_styles(font_name: str) -> dict[str, ParagraphStyle]:
    base = getSampleStyleSheet()
    return {
        "title": ParagraphStyle(
            "TitleCN",
            parent=base["Title"],
            fontName=font_name,
            fontSize=22,
            leading=30,
            textColor=colors.HexColor("#0F172A"),
            alignment=TA_CENTER,
            spaceAfter=9 * mm,
        ),
        "h2": ParagraphStyle(
            "H2CN",
            parent=base["Heading2"],
            fontName=font_name,
            fontSize=15,
            leading=21,
            textColor=colors.HexColor("#123A63"),
            spaceBefore=5 * mm,
            spaceAfter=2.5 * mm,
            keepWithNext=True,
        ),
        "h3": ParagraphStyle(
            "H3CN",
            parent=base["Heading3"],
            fontName=font_name,
            fontSize=11.5,
            leading=16,
            textColor=colors.HexColor("#1D4E89"),
            spaceBefore=3.5 * mm,
            spaceAfter=1.5 * mm,
            keepWithNext=True,
        ),
        "body": ParagraphStyle(
            "BodyCN",
            parent=base["BodyText"],
            fontName=font_name,
            fontSize=9.0,
            leading=13.4,
            textColor=colors.HexColor("#1F2937"),
            alignment=TA_LEFT,
            spaceAfter=2.0 * mm,
            wordWrap="CJK",
        ),
        "meta": ParagraphStyle(
            "MetaCN",
            parent=base["BodyText"],
            fontName=font_name,
            fontSize=9.4,
            leading=15,
            textColor=colors.HexColor("#475569"),
            alignment=TA_CENTER,
            spaceAfter=1.2 * mm,
            wordWrap="CJK",
        ),
        "caption": ParagraphStyle(
            "CaptionCN",
            parent=base["BodyText"],
            fontName=font_name,
            fontSize=8.3,
            leading=12,
            textColor=colors.HexColor("#475569"),
            alignment=TA_CENTER,
            spaceBefore=1.2 * mm,
            spaceAfter=4 * mm,
            wordWrap="CJK",
        ),
        "table": ParagraphStyle(
            "TableCN",
            parent=base["BodyText"],
            fontName=font_name,
            fontSize=7.6,
            leading=10,
            textColor=colors.HexColor("#1F2937"),
            wordWrap="CJK",
        ),
        "table_header": ParagraphStyle(
            "TableHeaderCN",
            parent=base["BodyText"],
            fontName=font_name,
            fontSize=7.8,
            leading=10,
            textColor=colors.white,
            wordWrap="CJK",
        ),
        "math": ParagraphStyle(
            "MathCN",
            parent=base["BodyText"],
            fontName=font_name,
            fontSize=11,
            leading=17,
            textColor=colors.HexColor("#0F172A"),
            alignment=TA_CENTER,
            spaceBefore=2 * mm,
            spaceAfter=3 * mm,
        ),
    }


def make_table(
    rows: list[list[str]],
    styles: dict[str, ParagraphStyle],
    available_width: float,
) -> Table:
    normalized = []
    for row_index, row in enumerate(rows):
        style = styles["table_header"] if row_index == 0 else styles["table"]
        normalized.append([Paragraph(cell, style) for cell in row])
    table = Table(
        normalized,
        colWidths=column_widths(rows, available_width),
        repeatRows=1,
        hAlign="LEFT",
    )
    table.setStyle(
        TableStyle(
            [
                ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#1D4E89")),
                ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
                ("LEFTPADDING", (0, 0), (-1, -1), 4),
                ("RIGHTPADDING", (0, 0), (-1, -1), 4),
                ("TOPPADDING", (0, 0), (-1, -1), 3.2),
                ("BOTTOMPADDING", (0, 0), (-1, -1), 3.2),
                ("GRID", (0, 0), (-1, -1), 0.35, colors.HexColor("#CBD5E1")),
                ("ROWBACKGROUNDS", (0, 1), (-1, -1), [colors.white, colors.HexColor("#F8FAFC")]),
            ]
        )
    )
    return table


def scaled_image(path: Path, available_width: float) -> Image:
    with PILImage.open(path) as opened:
        width, height = opened.size
    target_width = available_width
    target_height = target_width * height / width
    max_height = 155 * mm
    if target_height > max_height:
        target_height = max_height
        target_width = target_height * width / height
    return Image(str(path), width=target_width, height=target_height)


def paragraph_from_lines(
    lines: list[str], styles: dict[str, ParagraphStyle], font_name: str
) -> Paragraph:
    text = " ".join(line.strip() for line in lines)
    return Paragraph(rich_text(text, font_name), styles["body"])


def parse_markdown(
    markdown_path: Path,
    styles: dict[str, ParagraphStyle],
    font_name: str,
    available_width: float,
) -> list:
    lines = markdown_path.read_text(encoding="utf-8").splitlines()
    story: list = []
    index = 0
    first_title = True
    meta_mode = False
    section_page_breaks = {"3."}
    while index < len(lines):
        raw = lines[index]
        stripped = raw.strip()
        if not stripped:
            index += 1
            continue

        if stripped.startswith("# "):
            story.append(Paragraph(rich_text(stripped[2:], font_name), styles["title"]))
            story.append(HRFlowable(width="100%", thickness=1.2, color=colors.HexColor("#2B6CB0")))
            story.append(Spacer(1, 4 * mm))
            first_title = False
            meta_mode = True
            index += 1
            continue

        if stripped.startswith("## "):
            heading = stripped[3:]
            prefix = heading.split(maxsplit=1)[0]
            if prefix in section_page_breaks and story:
                story.append(PageBreak())
            story.append(Paragraph(rich_text(heading, font_name), styles["h2"]))
            meta_mode = False
            index += 1
            continue

        if stripped.startswith("### "):
            story.append(Paragraph(rich_text(stripped[4:], font_name), styles["h3"]))
            index += 1
            continue

        image_match = re.fullmatch(r"!\[([^]]+)\]\(([^)]+)\)", stripped)
        if image_match:
            image_path = (markdown_path.parent / image_match.group(2)).resolve()
            story.append(Spacer(1, 1.5 * mm))
            story.append(scaled_image(image_path, available_width))
            story.append(
                Paragraph(rich_text(image_match.group(1), font_name), styles["caption"])
            )
            index += 1
            continue

        if stripped.startswith("|"):
            collected = []
            while index < len(lines) and lines[index].strip().startswith("|"):
                collected.append(lines[index])
                index += 1
            rows = table_rows(collected, font_name)
            story.append(make_table(rows, styles, available_width))
            story.append(Spacer(1, 3 * mm))
            continue

        if stripped == "\\[":
            formula = []
            index += 1
            while index < len(lines) and lines[index].strip() != "\\]":
                formula.append(lines[index].strip())
                index += 1
            index += 1
            rendered = " ".join(formula)
            rendered = rendered.replace("\\odot", "×")
            rendered = rendered.replace("\\", "")
            story.append(Paragraph(html.escape(rendered), styles["math"]))
            continue

        bullet_match = re.match(r"^[-*] (.+)$", stripped)
        numbered_match = re.match(r"^(\d+)\. (.+)$", stripped)
        if bullet_match or numbered_match:
            ordered = bool(numbered_match)
            items = []
            while index < len(lines):
                current = lines[index].strip()
                match = (
                    re.match(r"^(\d+)\. (.+)$", current)
                    if ordered
                    else re.match(r"^[-*] (.+)$", current)
                )
                if not match:
                    break
                text_value = match.group(2) if ordered else match.group(1)
                index += 1
                continuation = []
                while index < len(lines):
                    candidate = lines[index]
                    if not candidate.strip():
                        break
                    if candidate.startswith("  ") and not candidate.strip().startswith("|"):
                        continuation.append(candidate.strip())
                        index += 1
                    else:
                        break
                if continuation:
                    text_value += " " + " ".join(continuation)
                items.append(
                    ListItem(
                        Paragraph(rich_text(text_value, font_name), styles["body"]),
                        leftIndent=4 * mm,
                    )
                )
            story.append(
                ListFlowable(
                    items,
                    bulletType="1" if ordered else "bullet",
                    start="1" if ordered else "circle",
                    leftIndent=6 * mm,
                    bulletFontName=font_name,
                    bulletFontSize=8,
                    spaceAfter=2 * mm,
                )
            )
            continue

        paragraph_lines = [stripped]
        index += 1
        while index < len(lines):
            candidate = lines[index].strip()
            if not candidate:
                break
            if candidate.startswith(("#", "|", "- ", "* ", "![", "\\[")):
                break
            if re.match(r"^\d+\. ", candidate):
                break
            paragraph_lines.append(candidate)
            index += 1
        style = styles["meta"] if meta_mode and not first_title else styles["body"]
        story.append(
            Paragraph(
                rich_text(" ".join(paragraph_lines), font_name),
                style,
            )
        )
    return story


def page_decorator(canvas, document, font_name: str) -> None:
    canvas.saveState()
    page_width, page_height = A4
    if document.page > 1:
        canvas.setStrokeColor(colors.HexColor("#CBD5E1"))
        canvas.setLineWidth(0.4)
        canvas.line(18 * mm, page_height - 14 * mm, page_width - 18 * mm, page_height - 14 * mm)
        canvas.setFont(font_name, 7.5)
        canvas.setFillColor(colors.HexColor("#64748B"))
        canvas.drawString(18 * mm, page_height - 11 * mm, "论文 Baseline 上的循环端粒干预测试")
        canvas.drawRightString(page_width - 18 * mm, 11 * mm, f"{document.page}")
    canvas.restoreState()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--markdown", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    font_name = register_fonts()
    styles = build_styles(font_name)
    page_width, _ = A4
    left_margin = right_margin = 18 * mm
    available_width = page_width - left_margin - right_margin
    story = parse_markdown(
        args.markdown, styles, font_name, available_width
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    document = SimpleDocTemplate(
        str(args.output),
        pagesize=A4,
        leftMargin=left_margin,
        rightMargin=right_margin,
        topMargin=19 * mm,
        bottomMargin=17 * mm,
        title="论文 Baseline 上的循环端粒干预测试报告",
        author="Jiaju research experiment",
        subject="Identity-initialized diagonal plus low-rank J on looped Transformer baselines",
    )
    decorator = lambda canvas, doc: page_decorator(canvas, doc, font_name)
    document.build(story, onFirstPage=decorator, onLaterPages=decorator)
    print(args.output)


if __name__ == "__main__":
    main()
