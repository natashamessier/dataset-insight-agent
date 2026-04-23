"""
PDF export using reportlab's Platypus layout engine.

xhtml2pdf and weasyprint both require native cairo/pango system libraries
that are not available in this environment. reportlab is pure Python and
produces equivalently styled output.

Public API:
    report_to_pdf(report_text, mode, plots) -> bytes
"""

import base64
import io
import re


def report_to_pdf(report_text: str, mode: str, plots: list) -> bytes:
    """
    Render a markdown report with embedded plot images to PDF bytes.

    Visualizations are rendered after the report body so the PDF places all
    charts after the written report content.
    """
    from reportlab.lib import colors
    from reportlab.lib.enums import TA_LEFT
    from reportlab.lib.pagesizes import A4
    from reportlab.lib.styles import ParagraphStyle
    from reportlab.lib.units import cm
    from reportlab.platypus import (
        HRFlowable,
        Image,
        Paragraph,
        SimpleDocTemplate,
        Spacer,
    )

    # ── Colour tokens (match app design system) ───────────────────────────────
    INK    = colors.HexColor("#2A2825")
    MUTED  = colors.HexColor("#9E9890")
    BORDER = colors.HexColor("#E4DDD3")

    # ── Paragraph styles ──────────────────────────────────────────────────────
    def _style(name, **kw):
        defaults = dict(fontName="Helvetica", fontSize=11, textColor=INK,
                        leading=18, spaceBefore=4, spaceAfter=8, alignment=TA_LEFT)
        defaults.update(kw)
        return ParagraphStyle(name, **defaults)

    S = {
        "meta":    _style("meta",   fontName="Helvetica", fontSize=9,
                          textColor=MUTED, spaceBefore=0, spaceAfter=20, leading=13),
        "h1":      _style("h1",     fontName="Times-Bold", fontSize=20,
                          spaceBefore=24, spaceAfter=8, leading=26),
        "h2":      _style("h2",     fontName="Times-Bold", fontSize=14,
                          spaceBefore=18, spaceAfter=6, leading=20),
        "h3":      _style("h3",     fontName="Times-Bold", fontSize=12,
                          spaceBefore=14, spaceAfter=4, leading=16),
        "body":    _style("body"),
        "bullet":  _style("bullet", leftIndent=16, spaceBefore=2, spaceAfter=2),
        "caption": _style("caption", fontName="Helvetica-Oblique", fontSize=8.5,
                          textColor=MUTED, spaceBefore=4, spaceAfter=12, leading=12),
    }

    # ── Inline markdown → reportlab-safe HTML ─────────────────────────────────
    def _inline(text: str) -> str:
        text = re.sub(r'\*\*(.*?)\*\*', r'<b>\1</b>', text)
        text = re.sub(r'\*(.*?)\*',     r'<i>\1</i>', text)
        text = re.sub(r'`(.*?)`',
                      r'<font face="Courier" size="9">\1</font>', text)
        # Escape bare ampersands that aren't already entities
        text = re.sub(r'&(?!#?\w+;)', '&amp;', text)
        return text

    # ── Plot flowables ────────────────────────────────────────────────────────
    def _plot_flowables(plot_list):
        items = []
        for plot in plot_list:
            img_bytes = base64.b64decode(plot["data"])
            img = Image(io.BytesIO(img_bytes), width=14 * cm, height=9 * cm, kind="proportional")
            items.append(Spacer(1, 8))
            items.append(img)
            items.append(Paragraph(_inline(plot.get("title", "")), S["caption"]))
        return items

    # ── Parse markdown into story flowables ───────────────────────────────────
    story: list = []
    story.append(Paragraph(f"Dataset Insight Agent &nbsp;&middot;&nbsp; {_inline(mode)}", S["meta"]))
    story.append(HRFlowable(width="100%", thickness=0.5, color=BORDER, spaceAfter=14))

    lines = report_text.split("\n")
    pending_bullets: list[str] = []

    def flush_bullets():
        for b in pending_bullets:
            story.append(Paragraph(f"&bull; {b}", S["bullet"]))
        if pending_bullets:
            story.append(Spacer(1, 6))
        pending_bullets.clear()

    i = 0
    while i < len(lines):
        line = lines[i]

        # ── Headings ──────────────────────────────────────────────────────────
        m = re.match(r'^(#{1,3})\s+(.*)', line)
        if m:
            flush_bullets()
            level = len(m.group(1))
            text  = _inline(m.group(2).strip())
            key   = f"h{min(level, 3)}"
            story.append(Paragraph(text, S[key]))
            if level == 1:
                story.append(HRFlowable(width="100%", thickness=0.5, color=BORDER, spaceAfter=10))
            i += 1
            continue

        # ── Bullet / numbered list ────────────────────────────────────────────
        m = re.match(r'^[-*+]\s+(.*)', line) or re.match(r'^\d+\.\s+(.*)', line)
        if m:
            pending_bullets.append(_inline(m.group(1).strip()))
            i += 1
            continue

        # ── Empty line ────────────────────────────────────────────────────────
        if not line.strip():
            flush_bullets()
            i += 1
            continue

        # ── Regular paragraph ─────────────────────────────────────────────────
        flush_bullets()
        story.append(Paragraph(_inline(line.strip()), S["body"]))
        i += 1

    flush_bullets()

    if plots:
        story.append(Spacer(1, 10))
        story.append(Paragraph("Visualizations", S["h2"]))
        story.extend(_plot_flowables(plots))

    # ── Build PDF ─────────────────────────────────────────────────────────────
    buf = io.BytesIO()
    doc = SimpleDocTemplate(
        buf,
        pagesize=A4,
        leftMargin=2.5 * cm,
        rightMargin=2.5 * cm,
        topMargin=2.5 * cm,
        bottomMargin=2.5 * cm,
    )
    doc.build(story)
    return buf.getvalue()
