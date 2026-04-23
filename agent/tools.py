import pandas as pd
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
import io
import base64
from typing import Optional


def clean_data(df: pd.DataFrame, strategy: str = "auto") -> tuple:
    """
    Clean the dataset and return (summary_dict, cleaned_df).

    The summary goes back to the model; the cleaned df replaces the working
    dataframe in the agent loop so all subsequent tools see clean data.

    Steps (always run in this order):
      1. Drop columns where >60% of values are null — not analytically useful.
      2. Strip leading/trailing whitespace from all string columns.
      3. Coerce object columns that look numeric (handles $, %, commas).
      4. Fill remaining nulls — median for numeric, mode/"Unknown" for categorical.
      5. Drop exact duplicate rows.
    """
    df = df.copy()
    summary = {
        "columns_dropped":    [],
        "columns_coerced":    [],
        "nulls_filled":       {},
        "duplicates_removed": 0,
        "warnings":           [],
    }

    # ── 1. Drop high-null columns ──────────────────────────────────────────
    null_fracs = df.isna().mean()
    to_drop = null_fracs[null_fracs > 0.6].index.tolist()
    if to_drop:
        df = df.drop(columns=to_drop)
        summary["columns_dropped"] = to_drop

    # ── 2. Strip whitespace ────────────────────────────────────────────────
    for col in df.select_dtypes(include="object").columns:
        df[col] = df[col].str.strip()

    # ── 3. Type coercion — object columns that look numeric ────────────────
    # Strip currency symbols, percent signs, and thousands separators, then
    # attempt conversion. Accept the coercion only if >50% of non-null values
    # parse successfully; use errors='coerce' so partial failures become NaN
    # rather than crashing.
    for col in list(df.select_dtypes(include="object").columns):
        non_null = df[col].dropna()
        if non_null.empty:
            continue
        stripped = (
            non_null.astype(str)
            .str.replace(r"[$%\s]", "", regex=True)
            .str.replace(",", "", regex=False)
        )
        probe = pd.to_numeric(stripped, errors="coerce")
        if probe.notna().mean() > 0.5:
            df[col] = pd.to_numeric(
                df[col].astype(str)
                .str.replace(r"[$%\s]", "", regex=True)
                .str.replace(",", "", regex=False),
                errors="coerce",
            )
            summary["columns_coerced"].append(col)

    # ── 4. Fill nulls ──────────────────────────────────────────────────────
    for col in df.columns:
        null_count = int(df[col].isna().sum())
        if null_count == 0:
            continue
        if pd.api.types.is_numeric_dtype(df[col]):
            fill_val = df[col].median()
            df[col] = df[col].fillna(fill_val)
        else:
            mode_vals = df[col].mode()
            fill_val = mode_vals.iloc[0] if not mode_vals.empty else "Unknown"
            df[col] = df[col].fillna(fill_val)
        summary["nulls_filled"][col] = null_count

    # ── 5. Duplicate rows ──────────────────────────────────────────────────
    n_before = len(df)
    df = df.drop_duplicates()
    summary["duplicates_removed"] = n_before - len(df)

    # Warn if cleaning left no numeric columns — downstream tools will be limited.
    if df.select_dtypes(include="number").empty:
        summary["warnings"].append(
            "No numeric columns remain after cleaning — "
            "summary stats, correlation, and outlier detection will not be available."
        )

    return summary, df


def inspect_dataset(df: pd.DataFrame) -> dict:
    """Profile the dataset — always call this first."""
    return {
        "shape": {"rows": int(df.shape[0]), "cols": int(df.shape[1])},
        "columns": {
            col: {
                "dtype": str(df[col].dtype),
                "missing": int(df[col].isna().sum()),
                "missing_pct": round(float(df[col].isna().mean()) * 100, 2),
                "nunique": int(df[col].nunique()),
            }
            for col in df.columns
        },
        "numeric_columns": df.select_dtypes(include="number").columns.tolist(),
        "categorical_columns": df.select_dtypes(include="object").columns.tolist(),
        "sample": df.head(3).fillna("").to_dict(orient="records"),
    }


def run_summary_stats(df: pd.DataFrame, columns: list = None) -> dict:
    """Descriptive statistics for numeric columns."""
    # Filter to numeric-only from the provided list; non-numeric columns are skipped.
    numeric_df = df[columns].select_dtypes(include="number") if columns else df.select_dtypes(include="number")
    if numeric_df.empty:
        return {"error": "No numeric columns found"}
    return numeric_df.describe().round(3).to_dict()


def run_correlation_analysis(df: pd.DataFrame, columns: list = None) -> dict:
    """Correlation matrix + top correlated pairs."""
    numeric_df = df[columns].select_dtypes(include="number") if columns else df.select_dtypes(include="number")
    if numeric_df.shape[1] < 2:
        return {"error": "Need at least 2 numeric columns for correlation"}
    corr = numeric_df.corr().round(3)
    pairs = []
    cols = corr.columns.tolist()
    for i in range(len(cols)):
        for j in range(i + 1, len(cols)):
            pairs.append({"col1": cols[i], "col2": cols[j], "r": round(float(corr.iloc[i, j]), 3)})
    pairs.sort(key=lambda x: abs(x["r"]), reverse=True)
    return {
        "matrix": corr.to_dict(),
        "top_correlations": pairs[:5],
    }


def run_distribution_analysis(df: pd.DataFrame, columns: list = None) -> dict:
    """Value counts for categorical columns."""
    cat_cols = columns or df.select_dtypes(include="object").columns.tolist()
    if not cat_cols:
        return {"error": "No categorical columns found"}
    return {col: df[col].value_counts().head(10).to_dict() for col in cat_cols[:5]}


def detect_outliers(df: pd.DataFrame, columns: list = None) -> dict:
    """IQR-based outlier detection for numeric columns."""
    if df.empty:
        return {}
    all_numeric = df.select_dtypes(include="number").columns.tolist()
    # Only analyse columns that are actually numeric; silently skip invalid names.
    numeric_cols = [c for c in (columns or all_numeric) if c in all_numeric]
    result = {}
    for col in numeric_cols:
        q1 = df[col].quantile(0.25)
        q3 = df[col].quantile(0.75)
        iqr = q3 - q1
        outliers = df[(df[col] < q1 - 1.5 * iqr) | (df[col] > q3 + 1.5 * iqr)]
        result[col] = {
            "outlier_count": int(len(outliers)),
            "outlier_pct": round(len(outliers) / len(df) * 100, 2) if len(df) else 0.0,
            "bounds": {
                "lower": round(float(q1 - 1.5 * iqr), 3),
                "upper": round(float(q3 + 1.5 * iqr), 3),
            },
        }
    return result


def _apply_chart_style(ax, fig):
    """Apply the app's design system to matplotlib charts."""
    # Background
    fig.patch.set_facecolor("#FFFFFF")
    ax.set_facecolor("#FDFCFA")

    # Spines — remove top/right, soften bottom/left
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["left"].set_color("#E4DDD3")
    ax.spines["bottom"].set_color("#E4DDD3")

    # Grid
    ax.yaxis.grid(True, color="#F0EDE7", linewidth=0.8, zorder=0)
    ax.set_axisbelow(True)

    # Tick labels
    ax.tick_params(colors="#9E9890", labelsize=9)

    # Axis labels
    ax.xaxis.label.set_color("#6B6560")
    ax.xaxis.label.set_size(10)
    ax.yaxis.label.set_color("#6B6560")
    ax.yaxis.label.set_size(10)


# Design system palette
SAGE    = "#7A9E8E"
SAGE_LT = "#A8C4B8"
LAV     = "#9B91C1"
CORAL   = "#D4836A"
NEUTRAL = ["#7A9E8E", "#9B91C1", "#D4836A", "#C8B89A", "#6B9DB0", "#B89EC8"]


def generate_plot(
    df: pd.DataFrame,
    plot_type: str,
    x: str = None,
    y: str = None,
    title: str = "",
) -> Optional[str]:
    """Generate a plot and return as base64-encoded PNG string."""
    fig, ax = plt.subplots(figsize=(8, 5))
    try:
        if plot_type == "histogram" and x and x in df.columns:
            df[x].dropna().hist(ax=ax, bins=30, color=SAGE, edgecolor="white", linewidth=0.4)
            ax.set_xlabel(x)
            ax.set_ylabel("Count")

        elif plot_type == "scatter" and x and y and x in df.columns and y in df.columns:
            ax.scatter(df[x], df[y], alpha=0.78, color="#5B8070", s=22, edgecolors="white", linewidths=0.35)
            ax.set_xlabel(x)
            ax.set_ylabel(y)

        elif plot_type == "bar" and x and x in df.columns:
            counts = df[x].value_counts().head(10)
            bars   = ax.bar(range(len(counts)), counts.values, color=SAGE, edgecolor="white", linewidth=0.4)
            ax.set_xticks(range(len(counts)))
            ax.set_xticklabels(counts.index, rotation=38, ha="right", fontsize=8.5)
            ax.set_xlabel(x)
            ax.set_ylabel("Count")

        elif plot_type == "correlation_heatmap":
            numeric_df = df.select_dtypes(include="number")
            if numeric_df.shape[1] < 2:
                plt.close(fig)
                return None
            cmap = sns.blend_palette(["#D4836A", "#F7F5F1", "#7A9E8E"], as_cmap=True)
            sns.heatmap(
                numeric_df.corr(),
                annot=True,
                fmt=".2f",
                ax=ax,
                cmap=cmap,
                center=0,
                linewidths=0.5,
                linecolor="#F0EDE7",
                annot_kws={"size": 8.5, "color": "#2A2825"},
                cbar_kws={"shrink": 0.8},
            )
            ax.tick_params(labelsize=8.5)
            _apply_chart_style(ax, fig)
            # Grid lines would render over heatmap cells — disable after style is applied.
            ax.yaxis.grid(False)
            ax.xaxis.grid(False)

        elif plot_type == "boxplot" and x and x in df.columns:
            bp = ax.boxplot(
                df[x].dropna(),
                patch_artist=True,
                boxprops=dict(facecolor="#EDF3F0", color=SAGE),
                medianprops=dict(color=SAGE_LT, linewidth=2),
                whiskerprops=dict(color="#9E9890"),
                capprops=dict(color="#9E9890"),
                flierprops=dict(marker="o", color=CORAL, alpha=0.5, markersize=4),
            )
            ax.set_ylabel(x)

        else:
            plt.close(fig)
            return None

        # correlation_heatmap already called _apply_chart_style inline above.
        if plot_type != "correlation_heatmap":
            _apply_chart_style(ax, fig)
        title_text = title or plot_type.replace("_", " ").title()
        ax.set_title(title_text, color="#2A2825", fontsize=11, fontweight="500", pad=12)

        plt.tight_layout(pad=1.1)
        buf = io.BytesIO()
        plt.savefig(buf, format="png", dpi=150, facecolor="#FFFFFF")
        plt.close(fig)
        buf.seek(0)
        return base64.b64encode(buf.read()).decode("utf-8")

    except Exception as e:
        plt.close(fig)
        print(f"[generate_plot] error: {e}")
        return None
