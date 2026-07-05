#!/usr/bin/env python3
"""Derive per-benchmark split id-lists from the unified issue manifests.

The manifest (`<bench>_issues.csv`) is the single source of truth. Each row's
`resolution` column (FIX | REMOVE | KEEP | TODO) decides which leaderboard
split a flagged task participates in:

  * Filtering split  -> drop every flagged task that is NOT explicitly KEEP
                        (i.e. FIX, REMOVE, TODO, ...). Re-scores the ORIGINAL
                        trajectories over the surviving task set; no reruns.
  * Fixed split      -> drop only REMOVE (unfixable); FIX tasks are rerun on
                        fixed data, KEEP/unflagged reuse the original results.

This script ONLY emits derived id-lists; it never touches the manifests or the
existing `<bench>_filter_ids.csv` / `<bench>_fixes.csv` files (those drive the
already-tested generation path). Outputs, written next to each manifest:

  <bench>_filtering_ids.csv   single column `task_id`, the Filtering drop set
  <bench>_remove_ids.csv      single column `task_id`, the REMOVE-only set

Run from anywhere:  python3 derive_split_ids.py
"""

import csv
import re
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent

# Manifests, relative to the repo root. Output prefix + dir are derived from
# each path (basename minus `_issues.csv`, written in the same directory).
MANIFESTS = [
    "tau2-bench/tau2bench_issues.csv",
    "tau-bench/taubench_issues.csv",
    "gorilla/berkeley-function-call-leaderboard/bfcl_v3_issues.csv",
    "gorilla/berkeley-function-call-leaderboard/bfcl_v4_issues.csv",
    "ACEBench/acebench_issues.csv",
    "NexusBench/nexusbench_issues.csv",
    "ComplexFuncBench/cfbench_issues.csv",
]


def _natural_key(task_id):
    """Sort `prefix_<int>` ids by (prefix, int) so e.g. retail_2 < retail_10."""
    return [int(t) if t.isdigit() else t for t in re.split(r"(\d+)", task_id)]


def _write_ids(path, ids):
    with open(path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["task_id"])
        for tid in sorted(ids, key=_natural_key):
            w.writerow([tid])


def derive(manifest_path):
    prefix = manifest_path.name[: -len("_issues.csv")]
    filtering, remove, keep, todo = set(), set(), 0, 0
    with open(manifest_path) as f:
        for row in csv.DictReader(f):
            tid = (row.get("task_id") or "").strip()
            res = (row.get("resolution") or "").strip().upper()
            if not tid:
                continue
            if res == "KEEP":
                keep += 1
                continue
            # Anything flagged and not explicitly kept is dropped for Filtering.
            filtering.add(tid)
            if res == "REMOVE":
                remove.add(tid)
            elif res == "TODO":
                todo += 1

    out_dir = manifest_path.parent
    _write_ids(out_dir / f"{prefix}_filtering_ids.csv", filtering)
    _write_ids(out_dir / f"{prefix}_remove_ids.csv", remove)
    note = f"  (incl. {todo} TODO)" if todo else ""
    print(
        f"{prefix:10s}  filtering={len(filtering):3d}{note}   "
        f"remove={len(remove):3d}   keep_retained={keep:3d}"
    )


def main():
    print("Deriving split id-lists from manifests (source of truth):\n")
    for rel in MANIFESTS:
        derive(REPO_ROOT / rel)
    print("\nDone. Existing *_filter_ids.csv / *_fixes.csv left untouched.")


if __name__ == "__main__":
    main()
