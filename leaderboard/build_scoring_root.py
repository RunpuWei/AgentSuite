#!/usr/bin/env python3
"""Assemble a scoring root from reference trajectories + submission manifests.

``compute_split_scores.py`` scores a directory laid out as
``<root>/<bench>-evaluation/<model>.jsonl``. This script builds that root by:

1. copying the canonical reference trajectories (the published board) from a local
   ``--reference-root`` (e.g. ``pipeline/benchmark``), and
2. overlaying every submission in ``--manifest-dir``: each manifest's per-benchmark
   trajectory is downloaded from Hugging Face, **validated** (schema + completeness),
   and written as ``<bench>-evaluation/<model>.jsonl``.

Invalid submissions are skipped with a warning rather than poisoning the board. The
result is fed to ``compute_split_scores.py`` unchanged.
"""

from __future__ import annotations

import argparse
import shutil
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import validate_submission as V  # noqa: E402


def _safe_model_name(name):
    # the scorer derives the model label from the file stem; keep it filesystem-safe.
    return "".join(c if (c.isalnum() or c in "-._") else "_" for c in str(name))


def copy_reference(reference_root, out_root):
    n = 0
    for bench in V.LEADERBOARD_BENCHMARKS:
        src = reference_root / f"{bench}-evaluation"
        if not src.is_dir():
            continue
        dst = out_root / f"{bench}-evaluation"
        dst.mkdir(parents=True, exist_ok=True)
        for p in src.glob("*.jsonl"):
            shutil.copy2(p, dst / p.name)
            n += 1
    print(f"reference: copied {n} trajectory files")


def overlay_submissions(manifest_dir, out_root):
    import yaml
    placed, skipped = 0, 0
    manifests = sorted(p for p in manifest_dir.glob("*.y*ml")
                       if p.name != "EXAMPLE.yaml")
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
            except Exception as e:
                ok, errors = False, [f"load failed: {e}"]
            if not ok:
                print(f"  ! {model}/{bench}: INVALID, skipping ({errors[0] if errors else 'unknown'})")
                skipped += 1
                continue
            dst_dir = out_root / f"{bench}-evaluation"
            dst_dir.mkdir(parents=True, exist_ok=True)
            shutil.copy2(local, dst_dir / f"{model}.jsonl")
            placed += 1
            print(f"  + {model}/{bench}: ok ({info.get('present')}/{info.get('expected')} ids)")
    print(f"submissions: placed {placed}, skipped {skipped}")


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--reference-root", type=Path, default=None,
                    help="Local root with <bench>-evaluation/ reference trajectories.")
    ap.add_argument("--manifest-dir", type=Path, default=HERE / "submissions")
    ap.add_argument("--out-root", type=Path, required=True)
    args = ap.parse_args()

    args.out_root.mkdir(parents=True, exist_ok=True)
    if args.reference_root:
        copy_reference(args.reference_root, args.out_root)
    if args.manifest_dir and args.manifest_dir.is_dir():
        overlay_submissions(args.manifest_dir, args.out_root)
    print(f"scoring root ready at {args.out_root}")


if __name__ == "__main__":
    main()
