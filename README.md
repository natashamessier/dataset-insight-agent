# Dataset Insight Agent

Dataset Insight Agent is a Streamlit web application that accepts a CSV upload and uses an Anthropic Claude LLM agent to autonomously analyze the data and generate a written report. What makes it agentic — rather than a scripted pipeline — is that the model reads the dataset profile and decides which tools to invoke, in what order, and whether preprocessing is needed before any analysis runs. The same upload can produce very different tool sequences depending on what the model finds: a clean numeric dataset might go straight to correlation and plots; a messy one with currency strings and missing values will trigger a cleaning step first. Reports are produced in three modes selectable by the user: **Executive Summary** (plain-language, decision-maker focused), **Technical Report** (full statistical breakdown with methodology), and **Storytelling Mode** (narrative arc, data journalism style). The tech stack is Anthropic Claude via the raw Python SDK, Streamlit, pandas, matplotlib, and seaborn.

---

## What Makes This Agentic

The LLM is not following a fixed script. It receives the dataset description and a set of tool definitions, then decides which tools to call and in what order based on what it observes. Here is the exact decision sequence:

**Step 1 — `inspect_dataset` (always called first)**
Profiles the dataset: shape, column names, dtypes, missing-value counts and percentages, unique value counts, and a small sample. The model uses this output to make every subsequent decision.

**Step 2 — `clean_data` (conditional)**
Called only if `inspect_dataset` reveals a data quality problem: more than 5% missing values across the dataset, object columns whose sample values look numeric (e.g. `"$1,200"`, `"3.5%"`, `"1,000"`), or datasets likely to contain duplicate rows. On a clean dataset, this tool is skipped entirely.

**Step 3 — Analysis tools (each conditional)**

| Tool | When the agent calls it |
|------|------------------------|
| `run_summary_stats` | Whenever numeric columns are present — always runs if there is anything to measure |
| `run_correlation_analysis` | Only when 2 or more numeric columns exist; skipped on single-numeric or all-categorical datasets |
| `run_distribution_analysis` | Only when categorical (string) columns are present |
| `detect_outliers` | When numeric columns are present and the data may contain anomalies — the model judges this from the summary stats range |

**Step 4 — `generate_plot` (called 2–4 times, choices are agent-driven)**
The model selects plot type based on what it found. Scatter plots for correlated pairs, heatmaps for wide numeric datasets, bar charts for categorical distributions, histograms or boxplots for individual numeric columns. It does not generate a fixed set of charts — it picks the ones that represent the actual findings.

**Step 5 — Report generation**
After all tool calls return, the model writes the full report from the tool outputs. It does not fabricate — if a tool was not called, the corresponding section is omitted.

The result is that two different CSV files will almost certainly produce a different sequence of tool calls. This is visible in the app's "Agent Decisions" trace, which shows every tool called, its inputs, and whether it succeeded.

---

## Architecture

The application is split into three layers: the Streamlit UI (`app.py`), the agent package that owns the LLM loop and all analysis logic, and the observability package that handles run logging and metric computation.

`app.py` handles file upload, passes the dataframe and selected mode to `run_agent`, then renders the returned report, plots, and tool trace. It never calls analysis tools directly — that is entirely the agent's job.

`agent/agent.py` drives the multi-turn Anthropic tool-use loop. It sends the tool definitions to the model, processes `tool_use` response blocks, dispatches each call to the correct function in `tools.py`, replaces the working dataframe if `clean_data` was called, and accumulates results until the model emits `end_turn`.

`agent/tools.py` contains all seven tool implementations as pure functions that take a dataframe and return a dict (or, for `generate_plot`, a base64 PNG string).

`agent/prompts.py` defines the base system prompt and the three mode-specific report format instructions.

`observability/logger.py` appends a structured JSON record to `logs/runs.jsonl` after every run and exposes `compute_metrics()` for the in-app dashboard.

```
dataset-insight-agent/
├── app.py                        # Streamlit entry point
├── agent/
│   ├── __init__.py
│   ├── agent.py                  # LLM loop, tool dispatcher, API key resolution
│   ├── tools.py                  # All 7 tool implementations
│   └── prompts.py                # System prompts and mode instructions
├── observability/
│   ├── __init__.py
│   └── logger.py                 # JSONL logger and metric computation
├── utils/
│   ├── __init__.py
│   └── export.py                 # PDF export via reportlab (markdown → styled PDF bytes)
├── .streamlit/
│   └── secrets.toml              # Local only — gitignored
├── logs/                         # Runtime only — gitignored
├── requirements.txt
├── .env.example
├── .gitignore
└── README.md
```

---

## Setup — Local Development

```bash
# 1. Clone
git clone https://github.com/natashamessier/dataset-insight-agent
cd dataset-insight-agent

# 2. Create and activate virtual environment
python -m venv venv
source venv/bin/activate       # Windows: venv\Scripts\activate

# 3. Install dependencies
pip install -r requirements.txt

# 4. Set up environment variables
cp .env.example .env
# Open .env and add your ANTHROPIC_API_KEY

# 5. Run
streamlit run app.py
```

App will be available at `http://localhost:8501`.

---

## Environment Variables

| Variable | Description | Required |
|----------|-------------|----------|
| `ANTHROPIC_API_KEY` | Anthropic API key from [console.anthropic.com](https://console.anthropic.com/) | Yes |

Never commit `.env` or `.streamlit/secrets.toml`. Both are listed in `.gitignore`.

---

## Deployment — Streamlit Community Cloud

1. Push the repo to GitHub. Before pushing, confirm `.env` and `logs/` are in `.gitignore` — run `git status` and verify neither appears as a tracked file.
2. Go to [share.streamlit.io](https://share.streamlit.io) and sign in with GitHub.
3. Click **New app**, select your repository and branch, and set the entry file to `app.py`.
4. Under **App Settings → Secrets**, paste exactly:
   ```toml
   ANTHROPIC_API_KEY = "your_key_here"
   ```
5. Click **Deploy**.

The app reads `st.secrets["ANTHROPIC_API_KEY"]` first and falls back to the environment variable, so the same codebase works locally and on Streamlit Cloud without any changes.

**Note on logs:** `logs/runs.jsonl` is written at runtime to the ephemeral Streamlit Cloud filesystem. It persists within a session but resets on redeployment. For persistent observability across deploys, replace the file logger in `observability/logger.py` with a database write (SQLite, Supabase, or similar).

---

## Observability & Metrics

Every agent run appends a record to `logs/runs.jsonl`:

```json
{
  "timestamp": "2025-01-15T14:32:01Z",
  "filename": "sales_data.csv",
  "mode": "Executive Summary",
  "rows": 1200,
  "cols": 8,
  "tool_calls": [
    {"tool": "inspect_dataset",        "inputs": {},                                          "success": true},
    {"tool": "clean_data",             "inputs": {},                                          "success": true},
    {"tool": "run_summary_stats",      "inputs": {},                                          "success": true},
    {"tool": "run_correlation_analysis","inputs": {},                                         "success": true},
    {"tool": "generate_plot",          "inputs": {"plot_type": "scatter", "x": "price", "y": "units"}, "success": true}
  ],
  "num_plots": 2,
  "latency": 18.4,
  "success": true,
  "error": null
}
```

Two metrics are computed from this log and displayed live in the app's observability dashboard:

**Metric 1 — Tool call success rate**
The percentage of individual tool calls that completed without error, across all logged runs. A low rate indicates either bugs in a tool implementation or the model generating inputs that don't match the actual column names in the dataset.

**Metric 2 — Average end-to-end latency**
Mean time in seconds from the first API call to the final report, measured per run. Reflects the combined cost of model inference and tool execution. A spike here usually means too many tool calls or an unusually large dataset being passed to the analysis tools.

Both metrics are computed in `observability/logger.py` via `compute_metrics()` and rendered in the expandable **System Metrics & Observability** panel at the bottom of the app.

---

## Acknowledgements

- [Anthropic Claude API](https://console.anthropic.com/) — LLM and tool-use framework. [claude.ai](https://claude.ai) assisted with code generation and scaffolding during development.
- [Streamlit](https://streamlit.io) — web application framework
- [pandas](https://pandas.pydata.org/) and [NumPy](https://numpy.org/) — data manipulation
- [Matplotlib](https://matplotlib.org/) and [seaborn](https://seaborn.pydata.org/) — visualization
- [DM Sans and DM Serif Display](https://fonts.google.com/) — typography via Google Fonts
