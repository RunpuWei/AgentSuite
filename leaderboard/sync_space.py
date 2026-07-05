#!/usr/bin/env python3
"""Push the Gradio app + frozen leaderboard CSVs to the Hugging Face Space.

Uploads ``leaderboard/hf_space/`` (app.py, requirements.txt, README.md) and the
current ``results/splits/*.csv`` into the Space's ``data/`` directory, so the live
board always reflects the latest server-side recomputation. The Space stays
read-only — it renders these CSVs and never computes scores itself.

Requires HF_TOKEN (write access to the Space) in the environment.
"""

from __future__ import annotations

import argparse
import os
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
HF_SPACE_DIR = Path(__file__).resolve().parent / "hf_space"
SPLITS_DIR = REPO_ROOT / "results" / "splits"

# CSVs the app reads (Overall + per-benchmark + partial); JSON intermediate excluded.
DATA_FILES = ["overall_leaderboard.csv", "overall_partial_models.csv"]


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--space", required=True, help="HF Space id, e.g. AgentSuite/AgentSuite")
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()

    token = os.environ.get("HF_TOKEN")
    if not token and not args.dry_run:
        raise SystemExit("HF_TOKEN not set")

    # Which split CSVs to ship: the two overall files + every per-benchmark table.
    data_files = list(DATA_FILES) + [p.name for p in SPLITS_DIR.glob("*_split_scores.csv")]
    data_files = [f for f in data_files if (SPLITS_DIR / f).exists()]

    print(f"Space: {args.space}")
    print(f"App files: {[p.name for p in HF_SPACE_DIR.iterdir() if p.is_file()]}")
    print(f"Data files -> data/: {data_files}")
    if args.dry_run:
        print("(dry run, nothing uploaded)")
        return

    from huggingface_hub import HfApi
    api = HfApi(token=token)
    api.create_repo(args.space, repo_type="space", space_sdk="gradio", exist_ok=True)

    # app.py / requirements.txt / README.md at the Space root
    for p in HF_SPACE_DIR.iterdir():
        if p.is_file():
            api.upload_file(path_or_fileobj=str(p), path_in_repo=p.name,
                            repo_id=args.space, repo_type="space")
    # CSVs under data/
    for f in data_files:
        api.upload_file(path_or_fileobj=str(SPLITS_DIR / f), path_in_repo=f"data/{f}",
                        repo_id=args.space, repo_type="space")
    print("sync complete")


if __name__ == "__main__":
    main()
