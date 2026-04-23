import base64
import re
import io
import re

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd
import streamlit as st

from agent.agent import run_agent
from agent.tools import _apply_chart_style
from observability.logger import load_logs, compute_metrics
from utils.export import report_to_pdf

# ─────────────────────────────────────────────────────────────────────────────
# Page config — must be first Streamlit call
# ─────────────────────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Insight Agent · Data Analysis",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ─────────────────────────────────────────────────────────────────────────────
# Design System CSS
# ─────────────────────────────────────────────────────────────────────────────
DESIGN_CSS = """
<style>
@import url('https://fonts.googleapis.com/css2?family=DM+Serif+Display&family=DM+Sans:opsz,wght@9..40,300;9..40,400;9..40,500;9..40,600&display=swap');
@import url('https://fonts.googleapis.com/css2?family=Material+Symbols+Rounded:FILL@0..1&display=swap');

/* ── Base ── */
.stApp {
    background-color: #F7F5F1 !important;
    font-family: 'DM Sans', sans-serif !important;
}
.block-container {
    padding-top: 2rem !important;
    padding-bottom: 4rem !important;
    max-width: 1080px !important;
}
#MainMenu, footer, header { visibility: hidden; }

/* ── Sidebar ── */
[data-testid="stSidebar"] {
    background-color: #FFFFFF !important;
    border-right: 1px solid #E4DDD3 !important;
}
[data-testid="stSidebar"] > div { padding-top: 1.5rem; }

/* ── Typography ── */
h1, h2, h3 {
    font-family: 'DM Serif Display', serif !important;
    color: #2A2825 !important;
    letter-spacing: -0.01em !important;
}
.stMarkdown p, .stMarkdown li {
    font-family: 'DM Sans', sans-serif;
    color: #2A2825;
    line-height: 1.75;
    font-size: 0.95rem;
}

/* ── Tabs ── */
[data-testid="stTabs"] [data-testid="stTab"] {
    font-family: 'DM Sans', sans-serif !important;
    font-size: 0.88rem !important;
    font-weight: 600 !important;
    color: #9E9890 !important;
}
[data-testid="stTabs"] [data-testid="stTab"][aria-selected="true"] {
    color: #2A2825 !important;
    border-bottom-color: #7A9E8E !important;
}

/* ── Buttons ── */
.stButton > button {
    background-color: #7A9E8E !important;
    color: #FFFFFF !important;
    border: none !important;
    border-radius: 10px !important;
    font-family: 'DM Sans', sans-serif !important;
    font-weight: 600 !important;
    font-size: 0.9rem !important;
    padding: 0.65rem 1.5rem !important;
    letter-spacing: 0.015em !important;
    transition: all 0.18s ease !important;
}
.stButton > button:hover {
    background-color: #5B8070 !important;
    box-shadow: 0 4px 16px rgba(122,158,142,0.32) !important;
    transform: translateY(-1px) !important;
}

/* ── Radio ── */
[data-testid="stRadio"] label {
    font-family: 'DM Sans', sans-serif !important;
    font-size: 0.8rem !important;
    color: #6B6560 !important;
    font-weight: 500 !important;
}
[data-testid="stRadio"] label p,
[data-testid="stRadio"] label span,
[data-testid="stRadio"] label div {
    color: #6B6560 !important;
    font-size: 0.8rem !important;
}
[data-testid="stRadio"] [role="radio"][aria-checked="true"] {
    border-color: #7A9E8E !important;
    background-color: #7A9E8E !important;
}
[data-testid="stRadio"] [role="radio"][aria-checked="true"]::before {
    background-color: #FFFFFF !important;
}

/* ── File uploader ── */
[data-testid="stFileUploader"] {
    background: #FFFFFF !important;
    border: 1px solid #E4DDD3 !important;
    border-radius: 12px !important;
    box-shadow: 0 2px 10px rgba(42,40,37,0.06) !important;
    padding: 0.6rem 0.85rem 0.75rem !important;
}
[data-testid="stFileUploader"] > label,
[data-testid="stFileUploader"] [data-testid="stWidgetLabel"] {
    padding-left: 0.15rem !important;
    padding-right: 0.15rem !important;
    margin-bottom: 0.45rem !important;
}
[data-testid="stFileUploader"] [data-testid="stWidgetLabel"] > div {
    padding-right: 0.4rem !important;
}
[data-testid="stFileUploaderDropzone"] {
    border: 2px dashed #C8C0B4 !important;
    border-radius: 10px !important;
    background: #FDFCFA !important;
    padding: 0.5rem !important;
    transition: border-color 0.2s !important;
}
[data-testid="stFileUploaderDropzone"] > div {
    padding: 0.9rem 1rem !important;
}
[data-testid="stFileUploaderDropzoneInstructions"] {
    padding: 0 0.85rem !important;
}
[data-testid="stFileUploaderDropzoneInstructions"] > div {
    padding: 0.1rem 0.35rem !important;
}
[data-testid="stFileUploaderDropzone"] small,
[data-testid="stFileUploaderDropzone"] span,
[data-testid="stFileUploaderDropzone"] p {
    padding-left: 0.35rem !important;
    padding-right: 0.35rem !important;
}
[data-testid="stFileUploaderDropzone"]:hover {
    border-color: #7A9E8E !important;
}
[data-testid="stFileUploader"] button,
[data-testid="stFileUploaderUploadButton"] button {
    background-color: #7A9E8E !important;
    border-color: #7A9E8E !important;
    color: #FFFFFF !important;
}
[data-testid="stFileUploader"] [data-testid="stFileUploaderFile"] {
    padding: 0.55rem 0.75rem !important;
    border-radius: 8px !important;
}
[data-testid="stFileUploader"] [data-testid="stFileUploaderFile"] > div {
    padding-inline: 0.25rem !important;
}
[data-testid="stFileUploader"] [aria-label*="Remove file"],
[data-testid="stFileUploader"] [title*="Remove file"] {
    margin-right: 0.25rem !important;
}

/* ── Expander ── */
[data-testid="stExpanderDetails"] {
    padding-top: 0.65rem !important;
}

/* ── Metrics ── */
[data-testid="metric-container"],
[data-testid="stMetric"] {
    background: #FFFFFF !important;
    border: 1px solid #E4DDD3 !important;
    border-radius: 12px !important;
    padding: 1.1rem 1.25rem !important;
    box-shadow: 0 2px 10px rgba(42,40,37,0.06) !important;
}
[data-testid="metric-container"] label,
[data-testid="stMetric"] label {
    font-family: 'DM Sans', sans-serif !important;
    font-size: 0.73rem !important;
    font-weight: 600 !important;
    color: #9E9890 !important;
    text-transform: uppercase !important;
    letter-spacing: 0.06em !important;
}
[data-testid="metric-container"] [data-testid="stMetricValue"],
[data-testid="stMetric"] [data-testid="stMetricValue"] {
    font-family: 'DM Serif Display', serif !important;
    font-size: 1.8rem !important;
    color: #2A2825 !important;
}

/* ── Expanders ── */
[data-testid="stExpander"] {
    background: #FFFFFF !important;
    border: 1px solid #E4DDD3 !important;
    border-radius: 12px !important;
    box-shadow: 0 2px 10px rgba(42,40,37,0.06) !important;
    overflow: hidden !important;
    margin-bottom: 0.75rem;
}
[data-testid="stExpander"] summary {
    font-family: 'DM Sans', sans-serif !important;
    font-weight: 600 !important;
    font-size: 0.88rem !important;
    color: #2A2825 !important;
    padding: 0.9rem 1.25rem !important;
}
[data-testid="stExpander"] summary:hover {
    background: #F7F5F1 !important;
}
[data-testid="stExpander"] summary svg {
    fill: #2A2825 !important;
    color: #2A2825 !important;
}
[data-testid="stDataFrame"] {
    border-radius: 8px !important;
    overflow: hidden !important;
    border: 1px solid #E4DDD3 !important;
}
.stDataFrame th {
    background: #F7F5F1 !important;
    font-family: 'DM Sans', sans-serif !important;
    font-size: 0.78rem !important;
    font-weight: 600 !important;
    color: #9E9890 !important;
    text-transform: uppercase !important;
    letter-spacing: 0.04em !important;
}

/* ── Status ── */
[data-testid="stStatus"] {
    border-radius: 10px !important;
    border: 1px solid #E4DDD3 !important;
    font-family: 'DM Sans', sans-serif !important;
    font-size: 0.88rem !important;
}

/* ── Alert / info ── */
[data-testid="stAlert"] {
    border-radius: 10px !important;
    font-family: 'DM Sans', sans-serif !important;
    font-size: 0.875rem !important;
}

/* ── Divider ── */
hr { border-color: #E4DDD3 !important; }

/* ── Images ── */
[data-testid="stImage"] img {
    border-radius: 8px !important;
}

/* ── Captions ── */
.stCaption, [data-testid="stCaptionContainer"] {
    font-family: 'DM Sans', sans-serif !important;
    color: #9E9890 !important;
    font-size: 0.8rem !important;
}

/* ── Scrollbar ── */
::-webkit-scrollbar { width: 5px; height: 5px; }
::-webkit-scrollbar-track { background: transparent; }
::-webkit-scrollbar-thumb { background: #D9D2C8; border-radius: 999px; }

/* ── Material Symbols ── */
.material-symbols-rounded {
    font-family: 'Material Symbols Rounded';
    font-weight: normal;
    font-style: normal;
    line-height: 1;
    letter-spacing: normal;
    text-transform: none;
    display: inline-block;
    white-space: nowrap;
    word-wrap: normal;
    direction: ltr;
    -webkit-font-smoothing: antialiased;
    font-variation-settings: 'FILL' 0, 'wght' 500, 'GRAD' 0, 'opsz' 24;
    vertical-align: middle;
}
</style>
"""
st.markdown(DESIGN_CSS, unsafe_allow_html=True)


# ─────────────────────────────────────────────────────────────────────────────
# HTML component helpers
# ─────────────────────────────────────────────────────────────────────────────
def card(content_html: str, padding: str = "1.5rem") -> str:
    return f"""<div style="
        background:#FFFFFF;border:1px solid #E4DDD3;border-radius:12px;
        padding:{padding};box-shadow:0 2px 10px rgba(42,40,37,0.06);
        margin-bottom:1rem;">{content_html}</div>"""


def pill(text: str, color: str = "sage") -> str:
    palettes = {
        "sage":     ("#EDF3F0", "#5B8070"),
        "lavender": ("#EFEDF7", "#7B6FAA"),
        "coral":    ("#FBF0EC", "#C4603D"),
        "neutral":  ("#F2EFE9", "#6B6560"),
    }
    bg, fg = palettes.get(color, palettes["sage"])
    return (
        f'<span style="display:inline-block;background:{bg};color:{fg};'
        f'border-radius:999px;padding:3px 11px;font-size:0.74rem;'
        f'font-weight:600;font-family:\'DM Sans\',sans-serif;letter-spacing:0.02em;">'
        f'{text}</span>'
    )


def material_icon(name: str, size: str = "1.15rem", color: str = "#6B6560", fill: int = 0) -> str:
    return (
        f'<span class="material-symbols-rounded" '
        f'style="font-size:{size};color:{color};font-variation-settings:\'FILL\' {fill}, \'wght\' 500, \'GRAD\' 0, \'opsz\' 24;">'
        f'{name}</span>'
    )


def section_label(icon_name: str, title: str, subtitle: str = "") -> str:
    sub = f'<span style="font-size:0.82rem;color:#9E9890;font-family:\'DM Sans\',sans-serif;"> · {subtitle}</span>' if subtitle else ""
    return (
        '<div style="display:flex;align-items:baseline;gap:0.5rem;'
        'margin-bottom:1rem;margin-top:1.75rem;">'
        f'{material_icon(icon_name, size="1.15rem", color="#6B6560")}'
        f'<span style="font-family:\'DM Serif Display\',serif;font-size:1.25rem;color:#2A2825;">{title}</span>'
        f'{sub}'
        '</div>'
    )


def normalize_report_text(report_text: str) -> str:
    """Trim common model formatting wrappers that show up as stray code blocks."""
    text = (report_text or "").strip()

    # Some model outputs wrap the full report in a fenced block like ```div or ```html.
    text = re.sub(r"^\s*```[a-zA-Z0-9_-]*\s*\n", "", text)
    text = re.sub(r"\n```\s*$", "", text)

    # If a single stray "div" line survives at the start, drop it.
    lines = text.splitlines()
    while lines and lines[0].strip().lower() == "div":
        lines.pop(0)
    text = "\n".join(lines).strip()

    # Keep only the structured report body if the model adds a conversational preface.
    heading_markers = [
        "## Dataset Overview",
        "## Key Findings",
        "## Recommendations",
        "## Dataset Profile",
        "## Statistical Summary",
        "## Correlation Analysis",
        "## Distribution Analysis",
        "## Outlier Detection",
        "## Limitations & Recommended Next Steps",
        "## The Hook",
        "## What the Data Reveals",
        "## Why It Matters",
    ]
    first_heading = min(
        (text.find(marker) for marker in heading_markers if marker in text),
        default=-1,
    )
    if first_heading > 0:
        text = text[first_heading:].lstrip()

    return text


TOOL_META = {
    "clean_data":               ("mop", "coral",    "Data cleaning"),
    "inspect_dataset":          ("search", "sage",     "Dataset profiling"),
    "run_summary_stats":        ("query_stats", "lavender",  "Summary statistics"),
    "run_correlation_analysis": ("device_hub", "sage",      "Correlation analysis"),
    "run_distribution_analysis":("bar_chart", "lavender",  "Distribution analysis"),
    "detect_outliers":          ("warning", "coral",     "Outlier detection"),
    "generate_plot":            ("insert_chart", "neutral",   "Plot generation"),
}


# ─────────────────────────────────────────────────────────────────────────────
# Sidebar
# ─────────────────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("""
    <div style="padding:0 0.25rem 1.25rem;">
        <div style="display:flex;align-items:center;gap:0.5rem;margin-bottom:0.35rem;">
            """ + material_icon("auto_awesome", size="1.3rem", color="#6B6560") + """
            <span style="font-family:'DM Serif Display',serif;font-size:1.2rem;color:#2A2825;">Insight Agent</span>
        </div>
        <p style="font-size:0.8rem;color:#B0A89E;margin:0;font-family:'DM Sans',sans-serif;line-height:1.55;">
            Upload a CSV · the agent decides what matters · you get the report
        </p>
    </div>
    """, unsafe_allow_html=True)

    st.divider()
    st.markdown('<p style="font-family:\'DM Sans\',sans-serif;font-size:0.72rem;font-weight:600;color:#B0A89E;text-transform:uppercase;letter-spacing:0.08em;margin-bottom:0.6rem;">Report Mode</p>', unsafe_allow_html=True)

    mode = st.radio("mode", ["Executive Summary", "Technical Report", "Storytelling Mode"], label_visibility="collapsed")

    mode_cfg = {
        "Executive Summary": ("#EDF3F0", "#C5DDD5", "#5B8070", "target", "Decision-maker focused. Plain language, actionable, no jargon."),
        "Technical Report":  ("#EFEDF7", "#CEC9E8", "#7B6FAA", "biotech", "Full statistical breakdown. Numbers, methodology, completeness."),
        "Storytelling Mode": ("#FBF0EC", "#F0C4B4", "#C4603D", "menu_book", "Narrative arc. Hook → Insight → So What. Data journalism style."),
    }
    bg, border, fg, mode_icon, desc = mode_cfg[mode]
    st.markdown(f"""
    <div style="background:{bg};border:1px solid {border};border-radius:10px;padding:0.8rem 0.9rem;margin-top:0.4rem;">
        <p style="font-family:'DM Sans',sans-serif;font-size:0.81rem;color:{fg};margin:0;line-height:1.55;display:flex;align-items:flex-start;gap:0.45rem;">{material_icon(mode_icon, size="1rem", color=fg)}<span>{desc}</span></p>
    </div>""", unsafe_allow_html=True)

    st.divider()
    tools_html = '<p style="font-family:\'DM Sans\',sans-serif;font-size:0.72rem;font-weight:600;color:#B0A89E;text-transform:uppercase;letter-spacing:0.08em;margin-bottom:0.7rem;">Agent Tools</p><div style="display:flex;flex-direction:column;gap:0.45rem;">'
    for _, (ico, _, label) in TOOL_META.items():
        tools_html += f'<div style="display:flex;align-items:center;gap:0.5rem;">{material_icon(ico, size="0.95rem", color="#6B6560")}<span style="font-family:\'DM Sans\',sans-serif;font-size:0.8rem;color:#6B6560;">{label}</span></div>'
    tools_html += "</div>"
    st.markdown(tools_html, unsafe_allow_html=True)


# ─────────────────────────────────────────────────────────────────────────────
# Hero
# ─────────────────────────────────────────────────────────────────────────────
st.markdown("""
<div style="margin-bottom:1.75rem;">
    <h1 style="font-family:'DM Serif Display',serif;font-size:2rem;color:#2A2825;margin:0 0 0.25rem;">
        Dataset Insight Agent
    </h1>
    <p style="font-family:'DM Sans',sans-serif;font-size:0.95rem;color:#9E9890;margin:0;">
        Upload a CSV and the agent decides what to analyze, what to visualize, and what to say!
    </p>
</div>
""", unsafe_allow_html=True)


# ─────────────────────────────────────────────────────────────────────────────
# Top-level tabs
# ─────────────────────────────────────────────────────────────────────────────
tab_main, tab_obs = st.tabs(["Analysis", "Observability"])


# ═════════════════════════════════════════════════════════════════════════════
# ANALYSIS TAB
# ═════════════════════════════════════════════════════════════════════════════
with tab_main:

    # ── Upload ────────────────────────────────────────────────────────────────
    uploaded_file = st.file_uploader(
        "Drop a CSV file here, or click to browse",
        type=["csv"],
        help="The agent will profile your data and decide which analyses to run.",
    )

    if uploaded_file is None:
        st.session_state.pop("agent_result", None)
        st.session_state.pop("agent_mode", None)
        st.session_state.pop("agent_filename", None)
        st.markdown(card("""
        <div style="text-align:center;padding:1.5rem 1rem;">
            <div style="margin-bottom:0.6rem;">""" + material_icon("upload_file", size="2.25rem", color="#9E9890") + """</div>
            <p style="font-family:'DM Serif Display',serif;font-size:1.05rem;color:#2A2825;margin:0 0 0.35rem;">No dataset yet</p>
            <p style="font-family:'DM Sans',sans-serif;font-size:0.83rem;color:#9E9890;margin:0;line-height:1.6;">
                The agent will inspect your columns, decide which analyses matter,<br>generate visualizations, and write a report, all autonomously.
            </p>
        </div>""", "1rem"), unsafe_allow_html=True)

    else:
        # ── Load CSV ──────────────────────────────────────────────────────────
        try:
            df = pd.read_csv(uploaded_file)
        except UnicodeDecodeError:
            uploaded_file.seek(0)
            try:
                df = pd.read_csv(uploaded_file, encoding="latin-1")
            except Exception as e:
                st.error(f"Could not read CSV: {e}")
                df = None
        except Exception as e:
            st.error(f"Could not read CSV: {e}")
            df = None

        if df is not None:
            numeric_cols = df.select_dtypes(include="number").columns.tolist()
            cat_cols     = df.select_dtypes(include="object").columns.tolist()
            missing_pct  = round(df.isna().mean().mean() * 100, 1)

            # ── Dataset overview ──────────────────────────────────────────────
            st.markdown(section_label("dataset", "Dataset Overview", uploaded_file.name), unsafe_allow_html=True)

            c1, c2, c3, c4 = st.columns(4)
            c1.metric("Rows", f"{df.shape[0]:,}")
            c2.metric("Columns", df.shape[1])
            c3.metric("Numeric", len(numeric_cols))
            c4.metric("Categorical", len(cat_cols))

            pills = '<div style="display:flex;flex-wrap:wrap;gap:0.35rem;align-items:center;margin-top:0.25rem;">'
            pills += '<span style="font-family:\'DM Sans\',sans-serif;font-size:0.77rem;color:#B0A89E;font-weight:500;margin-right:0.2rem;">Columns</span>'
            for col in numeric_cols[:8]:
                pills += pill(col, "sage")
            for col in cat_cols[:6]:
                pills += pill(col, "lavender")
            extra = len(df.columns) - 14
            if extra > 0:
                pills += pill(f"+ {extra} more", "neutral")
            pills += "</div>"
            if missing_pct > 5:
                pills += f'<div style="margin-top:0.5rem;">{pill(f"Missing data: {missing_pct}%", "coral")}</div>'

            st.markdown(card(pills, "1rem 1.25rem"), unsafe_allow_html=True)

            with st.expander("Preview  ·  first 10 rows"):
                st.dataframe(df.head(10), use_container_width=True, hide_index=True)

            # ── Generate button ───────────────────────────────────────────────
            st.markdown("<div style='height:0.5rem;'></div>", unsafe_allow_html=True)
            btn_col, _ = st.columns([1, 2])
            with btn_col:
                run = st.button(f"Generate {mode}", type="primary", use_container_width=True)

            if run:
                # ── Run agent ─────────────────────────────────────────────────
                with st.status("Agent is analyzing your dataset…", expanded=True) as status:
                    status_lines = []

                    def push_agent_update(message: str) -> None:
                        if not status_lines or status_lines[-1] != message:
                            status_lines.append(message)
                            st.write(message)

                    push_agent_update("Profiling columns and deciding which analyses to run…")
                    result = run_agent(df, mode, uploaded_file.name, progress_callback=push_agent_update)
                    if result["success"]:
                        status.update(label=f"Analysis complete · {result['latency']}s", state="complete", expanded=False)
                    else:
                        status.update(label="Analysis incomplete — try again", state="error")

                # Store so download-button reruns don't lose the result.
                st.session_state["agent_result"]   = result
                st.session_state["agent_mode"]     = mode
                st.session_state["agent_filename"] = uploaded_file.name

            # ── Render results (reads from session_state, survives any rerun) ──
            cached = st.session_state.get("agent_result")
            if cached and st.session_state.get("agent_filename") == uploaded_file.name:
                result    = cached
                r_mode    = st.session_state.get("agent_mode", mode)
                tool_calls    = result["tool_calls"]
                success_count = sum(1 for tc in tool_calls if tc["success"])

                # ── Agent decision trace ───────────────────────────────────────
                with st.expander(f"Agent decisions  ·  {len(tool_calls)} tools called  ·  {success_count} succeeded"):
                    st.markdown('<p style="font-family:\'DM Sans\',sans-serif;font-size:0.83rem;color:#9E9890;margin-bottom:1rem;">The LLM chose these tools based on what it found in your data. Not a fixed pipeline.</p>', unsafe_allow_html=True)

                    rows = '<div style="display:flex;flex-direction:column;gap:0.5rem;padding-bottom:1rem;">'
                    for i, tc in enumerate(tool_calls, 1):
                        ico, clr, label = TOOL_META.get(tc["tool"], ("build", "neutral", tc["tool"]))
                        ok_color  = "#7A9E8E" if tc["success"] else "#D4836A"
                        ok_symbol = "Success" if tc["success"] else "Failed"
                        extras = ""
                        if tc.get("inputs"):
                            parts = [f'{k}: <code style="font-size:0.75rem;background:#F2EFE9;padding:1px 5px;border-radius:4px;">{v}</code>' for k, v in tc["inputs"].items()]
                            extras = '<span style="color:#B0A89E;font-size:0.79rem;margin-left:0.5rem;">' + "  ·  ".join(parts) + "</span>"

                        rows += f"""<div style="display:flex;align-items:center;gap:0.7rem;
                                                background:#F7F5F1;border-radius:8px;padding:0.55rem 0.85rem;">
                            <span style="font-size:0.7rem;font-weight:700;color:#C8C0B4;min-width:18px;">{i:02d}</span>
                            <span>{material_icon(ico, size="0.95rem", color="#6B6560")}</span>
                            <span style="font-family:'DM Sans',sans-serif;font-size:0.85rem;font-weight:600;color:#2A2825;flex:1;">
                                {tc['tool']}{extras}
                            </span>
                            <span style="color:{ok_color};font-weight:700;font-size:0.85rem;">{ok_symbol}</span>
                        </div>"""
                    rows += "</div>"
                    st.markdown(rows, unsafe_allow_html=True)
                    st.markdown("<div style='height:0.75rem;'></div>", unsafe_allow_html=True)

                # ── Visualizations ────────────────────────────────────────────
                if result["plots"]:
                    st.markdown(section_label("insert_chart", "Visualizations", "generated by the agent"), unsafe_allow_html=True)
                    plots = result["plots"]
                    cols  = st.columns(min(len(plots), 2))
                    for i, plot in enumerate(plots):
                        with cols[i % 2]:
                            st.markdown(
                                f"""<div style="min-height:2.8rem;display:flex;align-items:flex-end;margin:0 0 0.6rem;">
                                <p style="font-family:'DM Sans',sans-serif;font-size:0.75rem;font-weight:600;color:#9E9890;
                                           text-transform:uppercase;letter-spacing:0.05em;margin:0;">{plot['title']}</p>
                                </div>""",
                                unsafe_allow_html=True,
                            )
                            st.image(base64.b64decode(plot["data"]), use_column_width=True)
                            st.markdown("<div style='height:1rem;'></div>", unsafe_allow_html=True)

                # ── Report ────────────────────────────────────────────────────
                mode_pill_color = {"Executive Summary": "sage", "Technical Report": "lavender", "Storytelling Mode": "coral"}
                st.markdown(section_label("description", "Report"), unsafe_allow_html=True)

                report_text = result.get("report", "")
                if not report_text:
                    print(f"[app] report is empty. Full result: {result}")
                    st.warning(f"Report came back empty. Raw result: {result}")
                else:
                    import markdown as md_lib
                    clean_report_text = normalize_report_text(report_text)
                    report_html = md_lib.markdown(
                        clean_report_text,
                        extensions=["nl2br", "tables", "fenced_code"],
                    )
                    st.markdown(
                        '<div style="background:#FFFFFF;border:1px solid #E4DDD3;border-radius:12px;'
                        'padding:2rem 2.25rem;box-shadow:0 2px 10px rgba(42,40,37,0.06);margin-bottom:1rem;">'
                        '<div style="margin-bottom:1.25rem;">'
                        + pill(r_mode, mode_pill_color.get(r_mode, "sage"))
                        + '</div>'
                        '<div style="font-family:\'DM Sans\',sans-serif;color:#2A2825;'
                        'line-height:1.75;font-size:0.95rem;">'
                        + report_html
                        + '</div></div>',
                        unsafe_allow_html=True,
                    )

                # ── Export button ──────────────────────────────────────────────
                if report_text:
                    pdf_bytes = report_to_pdf(normalize_report_text(report_text), r_mode, result.get("plots", []))
                    export_col, _ = st.columns([1, 2])
                    with export_col:
                        st.download_button(
                            "⬇  Export Report",
                            data=pdf_bytes,
                            file_name="insight_report.pdf",
                            mime="application/pdf",
                            use_container_width=True,
                        )


# ═════════════════════════════════════════════════════════════════════════════
# OBSERVABILITY TAB
# ═════════════════════════════════════════════════════════════════════════════
with tab_obs:
    logs    = load_logs()
    metrics = compute_metrics(logs)

    if not logs:
        st.markdown(card("""
        <div style="text-align:center;padding:2rem 1rem;">
            <div style="margin-bottom:0.6rem;">""" + material_icon("monitoring", size="2.25rem", color="#9E9890") + """</div>
            <p style="font-family:'DM Serif Display',serif;font-size:1.05rem;color:#2A2825;margin:0 0 0.35rem;">No runs logged yet</p>
            <p style="font-family:'DM Sans',sans-serif;font-size:0.83rem;color:#9E9890;margin:0;line-height:1.6;">
                Generate a report on the Analysis tab to populate this dashboard.
            </p>
        </div>""", "1rem"), unsafe_allow_html=True)

    else:
        # ── Section 1: Metrics ────────────────────────────────────────────────
        st.markdown(section_label("query_stats", "Metrics", "tracked on every run"), unsafe_allow_html=True)

        m1, m2, m3, m4 = st.columns(4)
        m1.metric(
            "Tool Success Rate",
            f"{metrics['tool_success_rate']}%" if metrics["tool_success_rate"] is not None else "—",
        )
        m2.metric(
            "Avg End-to-End Latency",
            f"{metrics['avg_latency_sec']}s" if metrics["avg_latency_sec"] is not None else "—",
        )
        m3.metric("Total Runs",       metrics["total_runs"])
        m4.metric("Run Success Rate", f"{metrics['run_success_rate']}%")

        st.markdown("""
        <div style="display:grid;grid-template-columns:repeat(4,1fr);gap:1rem;margin-top:0.5rem;margin-bottom:0.5rem;">
            <p style="font-family:'DM Sans',sans-serif;font-size:0.76rem;color:#9E9890;line-height:1.55;margin:0;">
                Tool success rate shows how often the agent's tool calls work. If it drops, the inputs may be off or a tool may be failing.
            </p>
            <p style="font-family:'DM Sans',sans-serif;font-size:0.76rem;color:#9E9890;line-height:1.55;margin:0;">
                End-to-end latency is the total time from upload to finished report. Higher numbers usually mean more steps or slower model responses.
            </p>
            <p style="font-family:'DM Sans',sans-serif;font-size:0.76rem;color:#9E9890;line-height:1.55;margin:0;">
                Total runs tells you how much history these metrics are based on. With only a few runs, the percentages can change a lot.
            </p>
            <p style="font-family:'DM Sans',sans-serif;font-size:0.76rem;color:#9E9890;line-height:1.55;margin:0;">
                Run success rate shows how often the agent finishes cleanly. If it drops, people are likely getting incomplete or broken reports.
            </p>
        </div>
        """, unsafe_allow_html=True)

        st.markdown("<div style='height:0.5rem;'></div>", unsafe_allow_html=True)

        # ── Section 2: Tool Usage ─────────────────────────────────────────────
        st.markdown(section_label("build", "Tool Usage", "which analyses the agent decides to run"), unsafe_allow_html=True)

        if metrics.get("tool_usage"):
            tool_freq = metrics["tool_usage"]
            tool_breakdown = metrics.get("tool_success_breakdown", {})

            # Bar chart — matplotlib so we can apply the app's design system.
            sorted_tools = sorted(tool_freq.items(), key=lambda x: -x[1])
            labels = [t for t, _ in sorted_tools]
            values = [v for _, v in sorted_tools]

            fig, ax = plt.subplots(figsize=(10, 5.4))
            ax.bar(range(len(labels)), values, color="#7A9E8E", edgecolor="white", linewidth=0.4)
            ax.set_xticks(range(len(labels)))
            ax.set_xticklabels(labels, rotation=38, ha="right", fontsize=10)
            ax.set_xlabel("Tool", labelpad=8)
            ax.set_ylabel("Total Calls", labelpad=8)
            ax.set_title("Tool Usage Frequency", color="#2A2825", fontsize=12, fontweight="500", pad=14)
            _apply_chart_style(ax, fig)
            plt.tight_layout()

            buf = io.BytesIO()
            plt.savefig(buf, format="png", dpi=220, bbox_inches="tight", facecolor="#FFFFFF")
            plt.close(fig)
            buf.seek(0)
            chart_b64 = base64.b64encode(buf.read()).decode()

            st.markdown(
                f"""<div style="background:#FFFFFF;border:1px solid #E4DDD3;border-radius:12px;
                padding:1rem 1rem 0.75rem;box-shadow:0 2px 10px rgba(42,40,37,0.06);margin-bottom:1rem;">
                    <img src="data:image/png;base64,{chart_b64}" alt="Tool Usage Frequency"
                         style="width:100%;height:auto;display:block;border-radius:8px;" />
                </div>""",
                unsafe_allow_html=True,
            )
            st.markdown(
                '<p style="font-family:\'DM Sans\',sans-serif;font-size:0.8rem;color:#6B6560;line-height:1.55;margin:0 0 1.75rem;">'
                'Each bar is how many times the agent called that tool across all runs. High frequency tools are what the agent most often decides are relevant.'
                '</p>',
                unsafe_allow_html=True,
            )

            # Detailed table
            table_rows = []
            for tool, calls in sorted(tool_freq.items(), key=lambda x: -x[1]):
                bd = tool_breakdown.get(tool, {})
                successes = bd.get("successes", 0)
                rate = round(successes / calls * 100, 1) if calls else 0
                table_rows.append({"Tool": tool, "Total Calls": calls, "Success Count": successes, "Success Rate %": rate})

            st.dataframe(
                pd.DataFrame(table_rows),
                use_container_width=True,
                hide_index=True,
            )

        st.markdown("<div style='height:0.5rem;'></div>", unsafe_allow_html=True)

        # ── Section 3: Run Log / Trace Viewer ────────────────────────────────
        st.markdown(section_label("folder_open", "Run Log", f"last {min(len(logs), 20)} runs · oldest → newest"), unsafe_allow_html=True)

        recent_runs = list(reversed(logs))[:20]
        for run in recent_runs:
            ts       = run.get("timestamp", "")[:19].replace("T", " ")
            fname    = run.get("filename", "unknown")
            run_mode = run.get("mode", "—")
            rows_n   = run.get("rows", "?")
            cols_n   = run.get("cols", "?")
            latency  = run.get("latency", "?")
            success  = run.get("success", False)
            error    = run.get("error", None)
            tcs      = run.get("tool_calls", [])

            status_icon  = "Success" if success else "Failed"
            status_color = "#7A9E8E" if success else "#D4836A"
            label = (
                f'{status_icon}  {ts}  ·  {fname}  ·  {run_mode}  ·  '
                f'{rows_n}×{cols_n}  ·  {len(tcs)} tools  ·  {latency}s'
            )

            with st.expander(label):
                if error:
                    st.markdown(
                        f'<div style="background:#FBF0EC;border:1px solid #F0C4B4;border-radius:8px;'
                        f'padding:0.6rem 1rem;margin-bottom:0.75rem;font-family:\'DM Sans\',sans-serif;'
                        f'font-size:0.82rem;color:#C4603D;">'
                        f'<strong>Error:</strong> {error}</div>',
                        unsafe_allow_html=True,
                    )

                if tcs:
                    tc_rows = '<div style="display:flex;flex-direction:column;gap:0.4rem;">'
                    for i, tc in enumerate(tcs, 1):
                        ico, _, _ = TOOL_META.get(tc["tool"], ("build", "neutral", tc["tool"]))
                        ok_c = "#7A9E8E" if tc.get("success") else "#D4836A"
                        ok_s = "Success" if tc.get("success") else "Failed"
                        inp_parts = ""
                        if tc.get("inputs"):
                            parts = [
                                f'{k}: <code style="font-size:0.74rem;background:#F2EFE9;padding:1px 4px;border-radius:3px;">{v}</code>'
                                for k, v in tc["inputs"].items()
                            ]
                            inp_parts = (
                                '<span style="color:#B0A89E;font-size:0.78rem;margin-left:0.4rem;">'
                                + "  ·  ".join(parts) + "</span>"
                            )
                        tc_rows += (
                            f'<div style="display:flex;align-items:center;gap:0.6rem;'
                            f'background:#F7F5F1;border-radius:7px;padding:0.45rem 0.75rem;">'
                            f'<span style="font-size:0.68rem;font-weight:700;color:#C8C0B4;min-width:16px;">{i:02d}</span>'
                            f'<span>{material_icon(ico, size="0.9rem", color="#6B6560")}</span>'
                            f'<span style="font-family:\'DM Sans\',sans-serif;font-size:0.83rem;font-weight:600;'
                            f'color:#2A2825;flex:1;">{tc["tool"]}{inp_parts}</span>'
                            f'<span style="color:{ok_c};font-weight:700;font-size:0.83rem;">{ok_s}</span>'
                            f'</div>'
                        )
                    tc_rows += "</div>"
                    st.markdown(tc_rows, unsafe_allow_html=True)
                else:
                    st.caption("No tool calls recorded for this run.")

                st.markdown(
                    f'<div style="margin-top:0.6rem;margin-bottom:0.45rem;font-family:\'DM Sans\',sans-serif;font-size:0.78rem;color:#9E9890;">'
                    f'Total latency: <strong style="color:#2A2825;">{latency}s</strong> · '
                    f'Status: <strong style="color:{status_color};">'
                    f'{"Succeeded" if success else "Failed"}</strong></div>',
                    unsafe_allow_html=True,
                )

        st.markdown("<div style='height:0.5rem;'></div>", unsafe_allow_html=True)

        # ── Section 4: Failure Analysis ───────────────────────────────────────
        failed_runs = [r for r in logs if not r.get("success")]
        tool_failures = [
            (r, tc)
            for r in logs
            for tc in r.get("tool_calls", [])
            if not tc.get("success")
        ]

        if failed_runs or tool_failures:
            st.markdown(section_label("warning", "Failure Analysis", f"{len(failed_runs)} failed runs · {len(tool_failures)} failed tool calls"), unsafe_allow_html=True)

            if failed_runs:
                st.markdown(
                    '<p style="font-family:\'DM Sans\',sans-serif;font-size:0.83rem;color:#9E9890;margin-bottom:0.75rem;">'
                    'Runs where the agent did not complete successfully. Inspect the tool trace to see where it broke.</p>',
                    unsafe_allow_html=True,
                )

                for run in failed_runs:
                    ts       = run.get("timestamp", "")[:19].replace("T", " ")
                    fname    = run.get("filename", "unknown")
                    run_mode = run.get("mode", "—")
                    error    = run.get("error", "unknown error")
                    tcs      = run.get("tool_calls", [])
                    latency  = run.get("latency", "?")
                    rows_n   = run.get("rows", "?")
                    cols_n   = run.get("cols", "?")

                    st.markdown(
                        f'<div style="background:#FBF0EC;border:1px solid #F0C4B4;border-radius:12px;'
                        f'padding:1.1rem 1.4rem;margin-bottom:0.75rem;">'
                        f'<div style="display:flex;align-items:baseline;justify-content:space-between;margin-bottom:0.5rem;">'
                        f'<span style="font-family:\'DM Serif Display\',serif;font-size:1rem;color:#2A2825;">{fname}</span>'
                        f'<span style="font-family:\'DM Sans\',sans-serif;font-size:0.78rem;color:#B0A89E;">{ts}</span>'
                        f'</div>'
                        f'<div style="font-family:\'DM Sans\',sans-serif;font-size:0.81rem;color:#6B6560;margin-bottom:0.5rem;">'
                        f'{run_mode} · {rows_n}×{cols_n} · {latency}s · {len(tcs)} tools called</div>'
                        f'<div style="font-family:\'DM Sans\',sans-serif;font-size:0.82rem;color:#C4603D;">'
                        f'<strong>Error:</strong> {error}</div>',
                        unsafe_allow_html=True,
                    )

                    if tcs:
                        tc_rows = '<div style="display:flex;flex-direction:column;gap:0.35rem;margin-top:0.75rem;">'
                        for i, tc in enumerate(tcs, 1):
                            ico, _, _ = TOOL_META.get(tc["tool"], ("build", "neutral", tc["tool"]))
                            ok_c = "#7A9E8E" if tc.get("success") else "#C4603D"
                            ok_s = "Success" if tc.get("success") else "Failed"
                            bg   = "#FFFFFF" if tc.get("success") else "#FDF3F0"
                            tc_rows += (
                                f'<div style="display:flex;align-items:center;gap:0.6rem;'
                                f'background:{bg};border-radius:7px;padding:0.4rem 0.7rem;">'
                                f'<span style="font-size:0.68rem;font-weight:700;color:#C8C0B4;min-width:16px;">{i:02d}</span>'
                                f'<span>{material_icon(ico, size="0.9rem", color="#6B6560")}</span>'
                                f'<span style="font-family:\'DM Sans\',sans-serif;font-size:0.83rem;'
                                f'font-weight:600;color:#2A2825;flex:1;">{tc["tool"]}</span>'
                                f'<span style="color:{ok_c};font-weight:700;font-size:0.83rem;">{ok_s}</span>'
                                f'</div>'
                            )
                        tc_rows += "</div>"
                        st.markdown(tc_rows + "</div>", unsafe_allow_html=True)
                    else:
                        st.markdown("</div>", unsafe_allow_html=True)
