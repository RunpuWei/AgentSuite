#!/usr/bin/env python3
"""Incrementally merge validated submissions into the published leaderboard.

The reference board (~30 models) is already scored and committed as
``results/splits/split_scores.json`` (a dump of ``{benchmark: [rows]}``) plus the
per-benchmark / overall CSVs. Re-downloading every reference trajectory just to
re-rank one new model would be wasteful, so this script does the cheap thing:

  1. load the committed board state from ``split_scores.json``;
  2. for each submission manifest, download + **validate** each per-benchmark
     trajectory and score *only that model* with ``score_model_file``;
  3. replace-by-model-name into the board (resubmitting a model updates it),
     re-rank every touched benchmark, and recompute the overall macro board.

The reference rows are reused as-is — they were computed by us, not uploaded, so
the trust model is intact (every *submission* is still recomputed here, never
trusted). Trajectory download stays fully optional for the rest of the repo: this
path only pulls the small per-submission files named in the manifests.

Usage::

    python leaderboard/merge_submissions.py \
        --manifest-dir leaderboard/submissions \
        --splits-dir results/splits
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
REPO_ROOT = HERE.parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(REPO_ROOT))

import validate_submission as V  # noqa: E402
from compute_split_scores import (  # noqa: E402
    _rank_key,
    aggregate_overall,
    load_filtering_ids,
    score_model_file,
    write_outputs,
)


def _safe_model_name(name):
    return "".join(c if (c.isalnum() or c in "-._") else "_" for c in str(name))


def rerank(rows):
    """Assign Original-primary rank and Filtering side rank (matches score_benchmark)."""
    rows = list(rows)
    rows.sort(key=lambda r: _rank_key(r["original_micro"]))
    for i, r in enumerate(rows, 1):
        r["rank"] = i
    for i, r in enumerate(sorted(rows, key=lambda r: _rank_key(r["filtering_micro"])), 1):
        r["filtering_rank"] = i
    return rows


def load_board(splits_dir):
    path = splits_dir / "split_scores.json"
    if not path.exists():
        raise SystemExit(
            f"no committed board at {path}; run compute_split_scores.py once to seed it"
        )
    return json.loads(path.read_text())


def merge_manifests(manifest_dir, board):
    """Score each submission and replace-by-model-name into ``board``. Returns touched benchmarks."""
    import yaml

    touched = set()
    placed, skipped = 0, 0
    manifests = sorted(p for p in manifest_dir.glob("*.y*ml") if p.name != "EXAMPLE.yaml")
    for mpath in manifests:
        spec = yaml.safe_load(mpath.read_text()) or {}
        model = _safe_model_name(spec.get("model") or mpath.stem)
        for bench, src in (spec.get("trajectories") or {}).items():
            if bench not in V.LEADERBOARD_BENCHMARKS:
                print(f"  ! {mpath.name}: unknown benchmark '{bench}', skipping")
                skipped += 1
                continue
            try:
                local = V._resolve_source(str(src))
                ok, errors, info = V.validate_jsonl(local, bench)
            except Exception as e:  # noqa: BLE001
                ok, errors = False, [f"load failed: {e}"]
            if not ok:
                print(f"  ! {model}/{bench}: INVALID, skipping ({errors[0] if errors else 'unknown'})")
                skipped += 1
                continue
            row = score_model_file(Path(local), load_filtering_ids(bench), bench)
            row["model"] = model  # the scorer derives this from the file stem; pin to the manifest
            rows = [r for r in board.get(bench, []) if r["model"] != model]  # replace on resubmit
            rows.append(row)
            board[bench] = rows
            touched.add(bench)
            placed += 1
            print(f"  + {model}/{bench}: ok ({info.get('present')}/{info.get('expected')} ids)")
    print(f"submissions: merged {placed}, skipped {skipped}")
    return touched


def main():
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument("--manifest-dir", type=Path, default=HERE / "submissions")
    ap.add_argument("--splits-dir", type=Path, default=REPO_ROOT / "results" / "splits")
    args = ap.parse_args()

    board = load_board(args.splits_dir)
    touched = merge_manifests(args.manifest_dir, board)
    if not touched:
        print("nothing to merge; board unchanged")
        return

    # Re-rank every benchmark (a new model shifts ranks board-wide) and rewrite CSVs.
    for bench in board:
        board[bench] = rerank(board[bench])
        write_outputs(bench, board[bench], args.splits_dir)

    (args.splits_dir / "split_scores.json").write_text(json.dumps(board, indent=2))
    if len(board) > 1:
        overall, partial, benchmarks = aggregate_overall(board, args.splits_dir)
        print(
            f"overall: {len(overall)} common models, {len(partial)} partial, "
            f"{len(benchmarks)} benchmarks"
        )
    print(f"board updated at {args.splits_dir}")


if __name__ == "__main__":
    main()
