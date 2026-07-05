# AgentSuite Leaderboard — submissions

The [AgentSuite Leaderboard](https://huggingface.co/spaces/AgentSuite/AgentSuite)
ranks models across 7 tool-use / agent benchmarks on two splits — **Original** (the
canonical task set) and **Filtering** (COBA-audited issues removed). This directory
holds everything for the leaderboard: the Gradio Space (`hf_space/`), the submission
validator (`validate_submission.py`), the canonical task-id lists (`expected_ids/`),
and submission manifests (`submissions/`). It is self-contained — adding your model
does not touch the rest of the repo.

## How scoring works (trust model)

You submit **trajectories**, not scores. CI downloads each trajectory, **recomputes**
its score with [`compute_split_scores.py`](../compute_split_scores.py), and updates
the board. Any `eval_result.score` in your file is recomputed, never trusted. The
manifest you commit carries only pointers — there is nowhere to put a number.

## Submit your model — 4 steps

**1. Generate trajectories.** Run the benchmark(s) with AgentSuite so the output
matches the expected JSONL schema:

```bash
python run_benchmarks.py --model <your-model> --benchmark ACEBench   # etc.
```

This writes one JSONL per benchmark where each line is one task:

```json
{"benchmark_name": "...", "task_name": "...", "messages": [...],
 "eval_result": {"score": 0.0}, "meta": {"id": "<task_id>", "is_correct": false}}
```

Each benchmark you submit must be a **complete run** — every in-scope task present.
The validator checks coverage against `expected_ids/<bench>.txt`. (You can submit a
subset of *benchmarks*; you cannot submit a subset of *tasks* within one.)

**2. Upload the trajectories to Hugging Face.** Put each JSONL in an HF dataset you
control (or open a PR to `AgentSuite/<bench>-trajectories`). We do **not** commit
multi-MB trajectories into this git repo — they live on HF, which also powers the
public trajectory viewer.

```bash
huggingface-cli upload your-org/my-model-agentsuite ./ACEBench.jsonl ACEBench.jsonl --repo-type dataset
```

**3. Add a manifest.** Copy [`submissions/EXAMPLE.yaml`](submissions/EXAMPLE.yaml)
to `submissions/<your-model>.yaml`, set `model`, and add one `hf://datasets/...`
pointer per benchmark.

**4. Open a pull request.** CI runs `validate_submission.py` on your manifest. On a
green check, a maintainer merges; scores are recomputed and the Space updates
automatically.

## Validate locally first

```bash
# a single local file
python leaderboard/validate_submission.py --jsonl ACEBench.jsonl --benchmark ACEBench

# the whole manifest (pulls trajectories from HF)
python leaderboard/validate_submission.py --manifest leaderboard/submissions/<your-model>.yaml
```

A passing run reports full id coverage and no schema errors.

## Notes

- **Benchmark keys** (exact): `tau-bench`, `tau2-bench`, `ACEBench`,
  `complex-func-bench`, `NexusBench`, `BFCL`, `BFCL_V4`.
- **Overall rank** is an equal-weight macro-average over benchmarks, restricted to
  models present in **all** of them. Submit fewer benchmarks and you're ranked
  per-benchmark and listed under "partial coverage" — still on the board, just not
  in the common-model overall rank.
- **Only the Original/Filtering splits** are live. Fixed-data runs are not accepted
  yet (no Fixed leaderboard results published).
- **How publishing works** (maintainers): on merge to `main`, CI runs
  `merge_submissions.py`, which reuses the committed `results/splits/` board and
  scores **only the new model** from its manifest trajectories — no bulk reference
  download. It replaces-by-model-name (resubmitting updates), re-ranks, recomputes
  the overall macro board, commits the CSVs, and syncs the Space.
- **Seeding / full rebuild** (maintainers, rarely — when the reference set itself
  changes): fetch the reference trajectories
  (`python pipeline/src/download_model_trajectory.py --include-optional`, optional
  and heavy) and run `compute_split_scores.py --traj-root pipeline/benchmark` to
  regenerate the whole board from scratch.
- **Regenerating id lists** (maintainers, when the reference set changes):
  `python leaderboard/validate_submission.py --build-expected --traj-root pipeline/benchmark`.
