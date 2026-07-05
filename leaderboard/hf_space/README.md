---
title: AgentSuite Leaderboard
emoji: 🛠️
colorFrom: indigo
colorTo: blue
sdk: gradio
sdk_version: 5.49.1
app_file: app.py
pinned: false
license: mit
---

# AgentSuite Leaderboard

Unified tool-use & agent evaluation across 8 benchmarks (τ-bench, τ²-bench,
ACEBench, ComplexFuncBench, NexusBench, BFCL v3, BFCL v4, DrafterBench), reported
on two splits:

- **Original**: the canonical task set, scored as published.
- **Filtering**: the same trajectories re-scored after removing tasks flagged by
  the **COBA (Component-Based Benchmark Auditing)** pipeline (malformed tools,
  incorrect ground truth, vague prompts, flawed environment responses).

Models are ranked by **Original**; the **Filtering** rank is shown alongside so you
can see how cleaning shifts positions. The overall tab is an equal-weight
average across benchmarks over the model set common to all of them.

Scores are **recomputed server-side** from submitted trajectories with
[`compute_split_scores.py`](https://github.com/Agent-Suite/AgentSuite/blob/main/compute_split_scores.py);
numbers in a submission are never trusted.

## This Space is read-only

It renders frozen CSVs in `data/`, synced from the AgentSuite repo's
`results/splits/` on every leaderboard update. To add your model, submit a
trajectory via the repo; see the
[submission guide](https://github.com/Agent-Suite/AgentSuite/blob/main/leaderboard/README.md).

[📄 Paper](https://openreview.net/forum?id=2Exmr1eIKZ) · [💻 GitHub](https://github.com/Agent-Suite/AgentSuite)
