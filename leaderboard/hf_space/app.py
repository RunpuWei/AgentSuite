"""AgentSuite leaderboard: Hugging Face Space (Gradio).

Read-only renderer of the frozen leaderboard CSVs produced by
``compute_split_scores.py`` (one Overall table + one per benchmark). The CSVs are
synced into ``data/`` from the AgentSuite repo's ``results/splits/`` on every
leaderboard update; the Space itself never computes scores. Models are ranked by
the **Original** split (the canonical task set); the **Filtering** rank (COBA-audited
issues removed) is shown alongside as a side rank.

Following nvidia/ProfBench: a small Gradio Blocks app, one parent Tabs group, a
search box per table, percentages formatted to 2 dp.
"""

import glob
import os

import gradio as gr
import pandas as pd

DATA_DIR = os.path.join(os.path.dirname(__file__), "data")

# Benchmark display order + pretty names for the per-benchmark tabs.
BENCH_ORDER = ["tau-bench", "tau2-bench", "ACEBench", "complex-func-bench",
               "NexusBench", "BFCL", "BFCL_V4", "DrafterBench"]
BENCH_TITLE = {
    "tau-bench": "τ-bench", "tau2-bench": "τ²-bench", "ACEBench": "ACEBench",
    "complex-func-bench": "ComplexFuncBench", "NexusBench": "NexusBench",
    "BFCL": "BFCL v3", "BFCL_V4": "BFCL v4", "DrafterBench": "DrafterBench",
}
# Benchmarks with no Original/Filtering split (issues fixed upstream → single score).
FIXED_ONLY = {"DrafterBench"}

INTRO = """# AgentSuite Leaderboard
Unified tool-use & agent evaluation across **8 benchmarks**, reported on two splits:
**Original** (canonical task set) and **Filtering** (COBA-audited issues removed).
Models are ranked by **Original**; the **Filtering** rank is shown alongside as an add-on.
(DrafterBench is a **fixed-only** benchmark: a single score, no split; see its tab.)

*A living leaderboard: models and benchmark updates land continuously; [submit a model](https://github.com/Agent-Suite/AgentSuite/blob/main/leaderboard/README.md) to be included.*

[📄 Paper](https://openreview.net/forum?id=2Exmr1eIKZ) · [➕ Submit a model](https://github.com/Agent-Suite/AgentSuite/blob/main/leaderboard/README.md) · [💻 GitHub](https://github.com/Agent-Suite/AgentSuite)
"""


def _pct(df, cols):
    for c in cols:
        if c in df.columns:
            df[c] = (df[c].astype(float) * 100).round(2)
    return df


def _col_widths(df):
    """Percentage widths (fixed proportions, so they don't auto-size to header text):
    narrow, equal rank columns; uniform score columns; the freed width to Model."""
    weights = []
    for c in df.columns:
        cl = str(c).lower()
        if cl == "model":
            weights.append(9.0)
        elif cl in ("rank", "filt. rank"):
            weights.append(1.0)
        else:
            weights.append(2.0)
    total = sum(weights)
    return [f"{w / total * 100:.2f}%" for w in weights]


def load_overall():
    df = pd.read_csv(os.path.join(DATA_DIR, "overall_leaderboard.csv"))
    pct_cols = [c for c in df.columns if c.endswith("_original")
                or c in ("original_avg", "filtering_avg")]
    df = _pct(df, pct_cols)
    rename = {"rank": "Rank", "filtering_rank": "Filt. Rank", "model": "Model",
              "original_avg": "Original (avg)", "filtering_avg": "Filtering (avg)"}
    rename.update({f"{b}_original": BENCH_TITLE.get(b, b) for b in BENCH_ORDER})
    return df.rename(columns=rename)


def load_bench(bench):
    path = os.path.join(DATA_DIR, f"{bench}_split_scores.csv")
    df = pd.read_csv(path)
    # Fixed-only benchmarks have no Original/Filtering split → one "Score" column.
    if bench in FIXED_ONLY:
        df = df[["rank", "model", "original_micro"]].copy()
        df = _pct(df, ["original_micro"])
        return df.rename(columns={"rank": "Rank", "model": "Model",
                                  "original_micro": "Score"})
    # Task counts (n_all / n_filtered) are identical for every model, so they are
    # stated once in a note above the table rather than repeated as columns.
    keep = ["rank", "filtering_rank", "model", "original_micro", "filtering_micro"]
    df = df[[c for c in keep if c in df.columns]].copy()
    df = _pct(df, ["original_micro", "filtering_micro"])
    return df.rename(columns={
        "rank": "Rank", "filtering_rank": "Filt. Rank", "model": "Model",
        "original_micro": "Original", "filtering_micro": "Filtering"})


def bench_note(bench):
    """One-line task-count summary shown above a per-benchmark table."""
    df = pd.read_csv(os.path.join(DATA_DIR, f"{bench}_split_scores.csv"))
    n_all = int(df["n_all"].iloc[0]) if "n_all" in df.columns and not df.empty else 0
    if bench in FIXED_ONLY:
        return ("**DrafterBench: fixed variant.** Its evaluation issues were corrected "
                "upstream, so we run only the fixed benchmark: there is **no "
                "Original/Filtering split**. **Score** = the DrafterBench average metric "
                f"over all **{n_all:,}** tasks (same set for every model).")
    if not n_all:
        return "**Original** = average over all tasks; **Filtering** = same over the audited-clean subset."
    n_filt = int(df["n_filtered"].iloc[0])
    dropped = n_all - n_filt
    return (f"**Original** = average over all **{n_all:,}** tasks · "
            f"**Filtering** = over the **{n_filt:,}** tasks remaining after removing "
            f"**{dropped:,}** COBA-audited issues (same task set for every model).")


def search(df, query):
    if not query:
        return df
    pats = [p.strip() for p in query.split(",") if p.strip()]
    if not pats:
        return df
    mask = df["Model"].astype(str).str.contains("|".join(pats), case=False, regex=True)
    return df[mask]


THEME = gr.themes.Soft(
    font=["system-ui", "-apple-system", "Segoe UI", "Roboto", "Helvetica Neue",
          "Arial", "sans-serif"],
    font_mono=["ui-monospace", "SFMono-Regular", "Menlo", "Consolas", "monospace"],
)

# Gradio's Dataframe renders with table-layout: auto, so column_widths act only as
# hints and content overrides them (long headers hold their columns wide while the
# wrappable Model names collapse to the narrowest column). Forcing table-layout:
# fixed makes our widths authoritative; the per-table rules below then set: narrow,
# EQUAL rank columns and a wide Model column. Column positions are stable per table
# type — Overall: Rank, Filt. Rank, Model, then score columns; split benchmarks:
# Rank, Filt. Rank, Model, Original, Filtering; DrafterBench (fixed): Rank, Model,
# Score; partial-coverage: Model first.
CSS = """
.tbl-overall table, .tbl-split table, .tbl-fixed table, .tbl-partial table {
    table-layout: fixed !important;
    width: 100% !important;
}
.tbl-overall td, .tbl-overall th, .tbl-split td, .tbl-split th,
.tbl-fixed td, .tbl-fixed th, .tbl-partial td, .tbl-partial th {
    overflow-wrap: anywhere;
}
/* Split benchmarks: Rank, Filt. Rank, Model, Original, Filtering */
.tbl-split th:nth-child(1), .tbl-split td:nth-child(1),
.tbl-split th:nth-child(2), .tbl-split td:nth-child(2) { width: 7% !important; }
.tbl-split th:nth-child(3), .tbl-split td:nth-child(3) { width: 60% !important; }
.tbl-split th:nth-child(n+4), .tbl-split td:nth-child(n+4) { width: 13% !important; }
/* Overall: Rank, Filt. Rank, Model, then 10 score columns */
.tbl-overall th:nth-child(1), .tbl-overall td:nth-child(1),
.tbl-overall th:nth-child(2), .tbl-overall td:nth-child(2) { width: 3.5% !important; }
.tbl-overall th:nth-child(3), .tbl-overall td:nth-child(3) { width: 27% !important; }
.tbl-overall th:nth-child(n+4), .tbl-overall td:nth-child(n+4) { width: 6.6% !important; }
/* DrafterBench (fixed): Rank, Model, Score */
.tbl-fixed th:nth-child(1), .tbl-fixed td:nth-child(1) { width: 8% !important; }
.tbl-fixed th:nth-child(2), .tbl-fixed td:nth-child(2) { width: 70% !important; }
.tbl-fixed th:nth-child(3), .tbl-fixed td:nth-child(3) { width: 22% !important; }
/* Partial-coverage: wide Model (first) column */
.tbl-partial th:nth-child(1), .tbl-partial td:nth-child(1) { width: 22% !important; }
"""

with gr.Blocks(title="AgentSuite Leaderboard", theme=THEME, css=CSS) as demo:
    gr.Markdown(INTRO)
    with gr.Tabs():
        with gr.Tab("Overall"):
            gr.Markdown("Equal-weight average across the 8 benchmarks, over the "
                        "model set common to **all** of them. Rows are ordered by the "
                        "**Original**-split rank; **Filt. Rank** shows where each model "
                        "would place under Filtering. The per-benchmark columns show "
                        "**Original** scores (DrafterBench contributes its single fixed "
                        "score to both averages). Partial-coverage models are listed "
                        "separately and excluded from the overall rank.")
            ov = load_overall()
            ov_box = gr.Textbox(label="Filter models (comma-separated, regex ok)",
                                placeholder="e.g. gpt, qwen, gemini")
            ov_tbl = gr.Dataframe(ov, interactive=False, wrap=True,
                                  column_widths=_col_widths(ov),
                                  elem_classes="tbl-overall")
            ov_box.change(lambda q: search(load_overall(), q), ov_box, ov_tbl)

            partial_path = os.path.join(DATA_DIR, "overall_partial_models.csv")
            if os.path.exists(partial_path):
                with gr.Accordion("Partial-coverage models (not ranked overall)", open=False):
                    pdf = pd.read_csv(partial_path)
                    pdf = _pct(pdf, [c for c in pdf.columns if c.endswith("_original")])
                    gr.Dataframe(pdf, interactive=False, wrap=True,
                                 column_widths=_col_widths(pdf),
                                 elem_classes="tbl-partial")

        for bench in BENCH_ORDER:
            if not os.path.exists(os.path.join(DATA_DIR, f"{bench}_split_scores.csv")):
                continue
            with gr.Tab(BENCH_TITLE.get(bench, bench)):
                gr.Markdown(bench_note(bench))
                df0 = load_bench(bench)
                box = gr.Textbox(label="Filter models (comma-separated, regex ok)")
                tbl = gr.Dataframe(df0, interactive=False, wrap=True,
                                   column_widths=_col_widths(df0),
                                   elem_classes=("tbl-fixed" if bench in FIXED_ONLY
                                                 else "tbl-split"))
                box.change((lambda b: (lambda q: search(load_bench(b), q)))(bench), box, tbl)

if __name__ == "__main__":
    demo.launch()
