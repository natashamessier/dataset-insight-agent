"""
Agentic analysis loop.

The LLM drives every decision — which tools to call, in what order, and
when to stop — via the Anthropic tool-use API. Nothing is hard-coded.
"""

import json
import os
import pathlib
import time

import anthropic
import pandas as pd

from .tools import (
    clean_data,
    inspect_dataset,
    run_summary_stats,
    run_correlation_analysis,
    run_distribution_analysis,
    detect_outliers,
    generate_plot,
)
from .prompts import get_system_prompt
from observability.logger import log_run

# ── Tool schemas sent to the Anthropic API ────────────────────────────────────

TOOL_DEFINITIONS = [
    {
        "name": "clean_data",
        "description": (
            "Clean the dataset before analysis. Call this if inspect_dataset revealed "
            "significant missing values (>5%), object columns that look like they contain "
            "numbers (e.g. '$1,200', '3.5%'), or likely duplicate rows. "
            "Do NOT call this if the data already looks clean — it is not always necessary. "
            "Returns a summary of what was changed; the cleaned data will be used automatically "
            "for all subsequent tool calls."
        ),
        "input_schema": {"type": "object", "properties": {}, "required": []},
    },
    {
        "name": "inspect_dataset",
        "description": (
            "Profile the dataset: shape, column names, dtypes, missing value counts, "
            "unique value counts, and a small sample. ALWAYS call this first."
        ),
        "input_schema": {"type": "object", "properties": {}, "required": []},
    },
    {
        "name": "run_summary_stats",
        "description": (
            "Compute descriptive statistics (mean, std, min, max, quartiles) for numeric columns. "
            "Optionally pass a list of column names to limit scope."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "columns": {
                    "type": "array",
                    "items": {"type": "string"},
                    "description": "Specific columns to analyze. Omit for all numeric columns.",
                }
            },
            "required": [],
        },
    },
    {
        "name": "run_correlation_analysis",
        "description": (
            "Compute a correlation matrix and return the top correlated pairs. "
            "Only useful when the dataset has 2 or more numeric columns."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "columns": {
                    "type": "array",
                    "items": {"type": "string"},
                    "description": "Specific numeric columns to include. Omit for all.",
                }
            },
            "required": [],
        },
    },
    {
        "name": "run_distribution_analysis",
        "description": (
            "Get value counts and frequency distributions for categorical columns. "
            "Only useful when the dataset has categorical/string columns."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "columns": {
                    "type": "array",
                    "items": {"type": "string"},
                    "description": "Specific categorical columns to analyze. Omit for all.",
                }
            },
            "required": [],
        },
    },
    {
        "name": "detect_outliers",
        "description": (
            "Detect outliers using the IQR method for numeric columns. "
            "Returns outlier count, percentage, and bounds per column."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "columns": {
                    "type": "array",
                    "items": {"type": "string"},
                    "description": "Specific numeric columns to check. Omit for all.",
                }
            },
            "required": [],
        },
    },
    {
        "name": "generate_plot",
        "description": (
            "Generate a data visualization. The image will be shown to the user directly — "
            "you will receive only a short confirmation, not the image bytes. "
            "Choose the plot type that best matches the data and finding:\n"
            "- histogram: distribution of a single numeric column\n"
            "- scatter: relationship between two numeric columns\n"
            "- bar: value counts for a categorical column\n"
            "- correlation_heatmap: full correlation matrix (3+ numeric columns)\n"
            "- boxplot: spread and outliers for a numeric column"
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "plot_type": {
                    "type": "string",
                    "enum": ["histogram", "scatter", "bar", "correlation_heatmap", "boxplot"],
                },
                "x": {"type": "string", "description": "Primary column (x-axis or main variable)."},
                "y": {"type": "string", "description": "Secondary column (y-axis, scatter only)."},
                "title": {"type": "string", "description": "Descriptive plot title."},
            },
            "required": ["plot_type"],
        },
    },
]


# ── Tool dispatcher ────────────────────────────────────────────────────────────

def _run_tool(name: str, inputs: dict, df: pd.DataFrame) -> tuple:
    """
    Execute a tool and return (result, success: bool, updated_df_or_None).

    updated_df_or_None is non-None only for clean_data; the agent loop
    replaces its working df when it receives a non-None value here.
    """
    try:
        if name == "clean_data":
            summary, cleaned_df = clean_data(df)
            return summary, True, cleaned_df
        elif name == "inspect_dataset":
            return inspect_dataset(df), True, None
        elif name == "run_summary_stats":
            return run_summary_stats(df, inputs.get("columns")), True, None
        elif name == "run_correlation_analysis":
            return run_correlation_analysis(df, inputs.get("columns")), True, None
        elif name == "run_distribution_analysis":
            return run_distribution_analysis(df, inputs.get("columns")), True, None
        elif name == "detect_outliers":
            return detect_outliers(df, inputs.get("columns")), True, None
        elif name == "generate_plot":
            result = generate_plot(
                df,
                inputs.get("plot_type"),
                inputs.get("x"),
                inputs.get("y"),
                inputs.get("title", ""),
            )
            return result, result is not None, None
        else:
            return f"Unknown tool: {name}", False, None
    except Exception as e:
        return f"Tool execution error: {str(e)}", False, None


# ── API key — resolved lazily at call time ─────────────────────────────────────

def _get_api_key() -> str:
    # Streamlit Community Cloud: secrets injected via st.secrets.
    # Local dev: .env file or shell environment variable.
    try:
        import streamlit as st
        key = st.secrets.get("ANTHROPIC_API_KEY")
        if key:
            return key
    except Exception:
        pass

    try:
        from dotenv import load_dotenv
        # Resolve .env relative to this file so it's found regardless of CWD.
        # override=True ensures .env wins even if st.secrets wrote an empty
        # string into os.environ first.
        _env_path = pathlib.Path(__file__).parent.parent / ".env"
        load_dotenv(_env_path, override=True)
    except ImportError:
        pass

    key = os.getenv("ANTHROPIC_API_KEY")
    if not key:
        raise ValueError(
            "ANTHROPIC_API_KEY not found. "
            "Add it to .streamlit/secrets.toml (Streamlit Cloud) or a .env file (local)."
        )
    return key


# ── Main agent loop ────────────────────────────────────────────────────────────

_MAX_ITERATIONS = 20  # hard cap to prevent runaway loops


def run_agent(df: pd.DataFrame, mode: str, filename: str, progress_callback=None) -> dict:
    """
    Run the analysis agent on a dataframe.

    The model decides which tools to call and in what order. It emits
    stop_reason="end_turn" once it has finished all tool calls and written
    the final report.

    Returns:
        {
            "report":     str,
            "plots":      [{"title": str, "data": base64_str}, ...],
            "tool_calls": [{"tool": str, "inputs": dict, "success": bool}, ...],
            "latency":    float,
            "success":    bool,
        }
    """
    client = anthropic.Anthropic(api_key=_get_api_key())
    start_time = time.time()

    def emit_progress(message: str) -> None:
        if progress_callback:
            progress_callback(message)

    messages = [
        {
            "role": "user",
            "content": (
                f"I've uploaded a dataset called '{filename}'. "
                f"Please analyze it and generate a {mode} report."
            ),
        }
    ]

    system_prompt = get_system_prompt(mode)
    tool_calls_log = []
    plots = []
    final_report = ""
    success = False

    emit_progress("Reading the request and planning the analysis…")

    for _iteration in range(_MAX_ITERATIONS):
        response = client.messages.create(
            model="claude-haiku-4-5-20251001",
            max_tokens=4096,
            system=system_prompt,
            tools=TOOL_DEFINITIONS,
            messages=messages,
        )

        # Always append the assistant turn before branching on stop_reason so
        # message history stays valid for the next API call.
        messages.append({"role": "assistant", "content": response.content})

        # ── Agent finished ─────────────────────────────────────────────────
        if response.stop_reason == "end_turn":
            emit_progress("Writing the final report…")
            for block in response.content:
                if hasattr(block, "text"):
                    final_report += block.text
            print(f"[agent] end_turn reached, report length: {len(final_report)}")
            success = True
            break

        # ── Tool use ───────────────────────────────────────────────────────
        if response.stop_reason in ("tool_use", "max_tokens"):
            tool_results = []
            for block in response.content:
                if block.type != "tool_use":
                    continue

                tool_name   = block.name
                tool_inputs = block.input

                progress_message = {
                    "inspect_dataset": "Inspecting columns, data types, and coverage…",
                    "clean_data": "Cleaning missing values and standardizing messy fields…",
                    "run_summary_stats": "Computing summary statistics for the key numeric columns…",
                    "run_correlation_analysis": "Checking relationships between numeric variables…",
                    "run_distribution_analysis": "Looking for patterns in category distributions…",
                    "detect_outliers": "Scanning for unusual values and edge cases…",
                    "generate_plot": "Generating visualizations for the strongest findings…",
                }.get(tool_name, f"Running {tool_name}…")
                emit_progress(progress_message)

                result, tool_success, updated_df = _run_tool(tool_name, tool_inputs, df)

                # clean_data returns a replacement df; swap it in so every
                # tool called after this point operates on the cleaned data.
                if updated_df is not None:
                    df = updated_df

                tool_calls_log.append(
                    {"tool": tool_name, "inputs": tool_inputs, "success": tool_success}
                )

                # Plots are stored in the plots list; only a short ack goes
                # back to the model so image bytes never bloat the context.
                if tool_name == "generate_plot" and tool_success and result:
                    plots.append(
                        {
                            "title": (
                                tool_inputs.get("title")
                                or tool_inputs.get("plot_type", "Plot").replace("_", " ").title()
                            ),
                            "data": result,
                        }
                    )
                    result_content = "Plot generated successfully and will be shown to the user."
                elif isinstance(result, (dict, list)):
                    # default=str handles NaN / numpy scalar types that aren't
                    # JSON-serialisable out of the box (e.g. from correlation matrix).
                    result_content = json.dumps(result, default=str)
                else:
                    result_content = str(result)

                tool_results.append(
                    {
                        "type":        "tool_result",
                        "tool_use_id": block.id,
                        "content":     result_content,
                    }
                )

            messages.append({"role": "user", "content": tool_results})
            continue

        # Any other stop_reason — exit cleanly rather than looping forever.
        break

    latency = round(time.time() - start_time, 2)

    log_run(
        {
            "filename":  filename,
            "mode":      mode,
            "rows":      int(df.shape[0]),
            "cols":      int(df.shape[1]),
            "tool_calls": tool_calls_log,
            "num_plots":  len(plots),
            "latency":    latency,
            "success":    success,
            "error":      None if success else "max_iterations_exceeded",
        }
    )

    if not final_report:
        final_report = "Analysis did not complete. Please try again."

    return {
        "report":     final_report,
        "plots":      plots,
        "tool_calls": tool_calls_log,
        "latency":    latency,
        "success":    success,
    }
