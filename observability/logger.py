"""
Observability layer.

Each agent run is appended as a JSON line to logs/runs.jsonl.
Captures: timestamp, filename, mode, dataset shape, tool calls (name + inputs + success),
          number of plots, latency, overall success, and any error reason.

compute_metrics() derives the two primary tracked metrics:
  1. Tool call success rate  — quality/reliability signal
  2. Average end-to-end latency — operational signal
Plus bonus counts for context in the dashboard.
"""

import json
import os
from datetime import datetime, timezone

LOG_FILE = "logs/runs.jsonl"


def log_run(data: dict) -> None:
    """Append a run record to the JSONL log file."""
    os.makedirs("logs", exist_ok=True)
    entry = {"timestamp": datetime.now(timezone.utc).isoformat(), **data}
    with open(LOG_FILE, "a") as f:
        f.write(json.dumps(entry) + "\n")


def load_logs() -> list:
    """Load all run records from the log file. Returns [] if the file doesn't exist yet."""
    if not os.path.exists(LOG_FILE):
        return []
    records = []
    with open(LOG_FILE, "r") as f:
        for line in f:
            line = line.strip()
            if line:
                try:
                    records.append(json.loads(line))
                except json.JSONDecodeError:
                    pass
    return records


def compute_metrics(logs: list) -> dict:
    """
    Compute tracked metrics from log records.

    Metric 1 — Tool call success rate:
        Measures reliability of the agent's tool execution layer.
        A low rate indicates bugs in tool implementations or bad LLM inputs.

    Metric 2 — Average end-to-end latency (seconds):
        Measures system responsiveness. High latency = poor UX, likely too many
        tool calls or slow model responses.
    """
    if not logs:
        return {}

    # Run-level
    successful_runs = [l for l in logs if l.get("success")]
    latencies = [l["latency"] for l in logs if "latency" in l]

    # Tool-level
    all_tool_calls = [tc for l in logs for tc in l.get("tool_calls", [])]
    successful_tool_calls = [tc for tc in all_tool_calls if tc.get("success")]

    # Tool usage frequency + per-tool success breakdown
    tool_freq = {}
    tool_success_breakdown = {}
    for tc in all_tool_calls:
        t = tc["tool"]
        tool_freq[t] = tool_freq.get(t, 0) + 1
        if t not in tool_success_breakdown:
            tool_success_breakdown[t] = {"calls": 0, "successes": 0}
        tool_success_breakdown[t]["calls"] += 1
        if tc.get("success"):
            tool_success_breakdown[t]["successes"] += 1

    # Mode breakdown
    mode_freq = {}
    for l in logs:
        m = l.get("mode", "unknown")
        mode_freq[m] = mode_freq.get(m, 0) + 1

    return {
        # Core metrics
        "tool_success_rate": (
            round(len(successful_tool_calls) / len(all_tool_calls) * 100, 1)
            if all_tool_calls
            else None
        ),
        "avg_latency_sec": (
            round(sum(latencies) / len(latencies), 2) if latencies else None
        ),
        # Supporting counts
        "total_runs":              len(logs),
        "run_success_rate":        round(len(successful_runs) / len(logs) * 100, 1),
        "total_tool_calls":        len(all_tool_calls),
        "avg_tools_per_run":       round(len(all_tool_calls) / len(logs), 1),
        "tool_usage":              tool_freq,
        "tool_success_breakdown":  tool_success_breakdown,
        "mode_breakdown":          mode_freq,
    }
