# Evaluation splits & issue manifests

AgentSuite records every flagged task in a single per-benchmark manifest and
derives the cleaned evaluation variants from it. This keeps the *policy* (which
tasks are problematic and what to do about them) in one place, separate from the
per-benchmark *mechanics* of running and scoring.

## The manifest (source of truth)

Each benchmark has one manifest, `<bench>_issues.csv`, with the columns:

| Column | Meaning |
|---|---|
| `task_id` | Task identifier, in the benchmark's native id form (e.g. `airline_3`, `multi_turn_base_10`). |
| `detection` | How the issue was found (`pipeline_review`, `GT_manual`, `invalid_function_call`, …). |
| `issue_reason` | Human-readable description of the problem. |
| `resolution` | One of `FIX`, `REMOVE`, `KEEP` (and a transitional `TODO`). |
| `resolution_detail` | What the fix is, or why the task is kept/removed. |
| `source` | `pipeline` (COBA) or `GT` (manual ground-truth review). |

`resolution` semantics:

- **`FIX`** — the task has a correctable issue; a repair is applied in the fixed
  data and the task is rerun for the Fixed variant.
- **`REMOVE`** — the task is unfixable / invalid; it is dropped from the cleaned
  variants.
- **`KEEP`** — the task was flagged but reviewed as valid (e.g. easy but
  gradeable, or a `detection=manual_extrapolation` sibling edit); retained in all
  variants.
- **`TODO`** — flagged as problematic but the resolution is not yet recorded.
  Treated as a drop for the Filtering variant until reclassified.

(ACEBench's manifest carries an extra `gt_confirmed` column from manual GT
review; it does not affect split derivation.)

## The three splits

- **Original** — the canonical run over the full task set on original data.
- **Filtering** — re-scores the *same* Original trajectories over the surviving
  task set, dropping every flagged task that is not `KEEP` (i.e. `FIX` ∪
  `REMOVE` ∪ `TODO`). No reruns, original data. This is the launch variant.
- **Fixed** — applies targeted repairs (`FIX`, rerun on fixed data) and drops
  the unfixable tasks (`REMOVE`); `KEEP` and unflagged tasks reuse the Original
  results.

## Deriving the id-lists

`derive_split_ids.py` reads each manifest and writes, next to it:

- `<bench>_filtering_ids.csv` — the Filtering drop set (every non-`KEEP` task).
- `<bench>_remove_ids.csv` — the `REMOVE`-only set (used by the Fixed variant).

```bash
python derive_split_ids.py
```

The script is idempotent and only emits these two derived files. It never
touches the manifests or the existing `<bench>_filter_ids.csv` / `<bench>_fixes.csv`
files that drive the current generation path.

Current derived counts:

| Benchmark | Filtering drop | `REMOVE` | `KEEP` retained |
|---|---|---|---|
| `tau2bench` | 32 | 3 | 38 |
| `taubench` | 26 | 0 | 17 |
| `bfcl_v3` | 151 | 43 | 0 |
| `bfcl_v4` | 13 | 0 | 0 |
| `acebench` | 126 | 26 | 32 |
| `nexusbench` | 159 | 159 | 0 |
| `cfbench` | 176 | 146 | 0 |

## Per-benchmark notes

**tau2bench — fixed data is a union.** `domains_fixed/` is the union of the
upstream tau2-verified task corrections and our COBA fixes, to provide the most
comprehensive fixed set. It therefore alters more tasks (≈74: 29 airline,
43 retail, 2 telecom) than the manifest flags as `FIX` (29). The extra edits are
upstream user-instruction clarifications (e.g. "…even if the upgrade is not
possible") carried in for completeness. For the Fixed variant the rerun set
should be taken from the actual `domains` vs `domains_fixed` diff, not the
manifest `FIX` list alone.

**drafterbench — not applicable.** Issues we identified were reported and fixed
upstream; no filtering is applied here.

**nexusbench / multichallenge — Fixed variant not yet supported** by
`run_benchmarks.py` (`--fixed` errors for these). Filtering id-lists are still
derived from the manifest.
