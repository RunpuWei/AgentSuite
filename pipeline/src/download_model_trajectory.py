#!/usr/bin/env python3
"""
Download evaluation JSONL (model trajectories) from Hugging Face into
``benchmark/<benchmark>-evaluation/``, matching what ``main.py`` / ``BenchmarkDataLoader``
expect (see README "Benchmark Data Setup").

Dataset repos are hosted under the ``AgentSuite`` organization on Hugging Face. The curated
public release lives in the ``<benchmark>-trajectories`` repos; for benchmarks that don't have
a curated trajectory repo yet we fall back to the full release ``<benchmark>-evaluation-full``.
Either way the local folder is named ``<benchmark>-evaluation`` so the pipeline loader (which
looks for ``benchmark/<benchmark>-evaluation/``) keeps working unchanged.

Default repos line up with ``Benchmark`` in ``src.utils.types`` (one HF dataset per
supported benchmark). Additional public datasets can be pulled with repeated ``--repo``.

Usage (from repository root)::

    python src/download_model_trajectory.py
    python src/download_model_trajectory.py --benchmark complex-func-bench --patterns "*.jsonl"
    python src/download_model_trajectory.py --patterns "*.jsonl"
    python src/download_model_trajectory.py --repo AgentSuite/complex-func-bench-trajectories --patterns "*.jsonl"

Public datasets usually work without login. For private datasets:

    huggingface-cli login
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path
from typing import List, Optional, Tuple

try:
    import huggingface_hub
    from huggingface_hub import snapshot_download
except Exception:  # pragma: no cover - import guard for CLI
    print(
        "ERROR: huggingface_hub is not installed. Run: pip install -U huggingface_hub",
        file=sys.stderr,
    )
    raise

# COBA pipeline repository root (parent of ``src/``)
REPO_ROOT = Path(__file__).resolve().parent.parent

# Hugging Face dataset ids per pipeline benchmark. Keys match ``Benchmark.value`` in
# ``src/utils/types.py`` (same strings as ``main.py --target-benchmark``).
#
# Curated public trajectories ship as ``AgentSuite/<benchmark>-trajectories``. DrafterBench's
# curated release is named ``AgentSuite/DrafterBench-fixed-trajectories`` (the fixed variant).
BENCHMARK_VALUE_TO_HF_REPO: dict[str, str] = {
    "tau-bench": "AgentSuite/tau-bench-trajectories",
    "tau2-bench": "AgentSuite/tau2-bench-trajectories",
    "ACEBench": "AgentSuite/ACEBench-trajectories",
    "complex-func-bench": "AgentSuite/complex-func-bench-trajectories",
    "NexusBench": "AgentSuite/NexusBench-trajectories",
    "DrafterBench": "AgentSuite/DrafterBench-fixed-trajectories",
    "BFCL": "AgentSuite/BFCL-trajectories",
    "BFCL_V4": "AgentSuite/BFCL_V4-trajectories",
}


def local_dir_name(benchmark_value: str) -> str:
    """Local folder the pipeline loader expects, regardless of the remote repo name.

    ``BenchmarkDataLoader.load_benchmark_data`` resolves data at
    ``benchmark/<benchmark>-evaluation/``, so we always materialize into that folder even
    when the source repo is ``<benchmark>-trajectories`` or ``<benchmark>-evaluation-full``.
    """
    return f"{benchmark_value}-evaluation"


# Default batch download when neither ``--benchmark`` nor ``--repo`` is set (benchmark values).
DEFAULT_PIPELINE_BENCHMARKS: List[str] = [
    "tau-bench",
    "tau2-bench",
    "ACEBench",
    "complex-func-bench",
    "DrafterBench",
    "BFCL",
]

# Extra benchmarks (use ``--include-optional`` with the default batch, or pass ``--benchmark``).
OPTIONAL_EXTRA_BENCHMARKS: List[str] = [
    "BFCL_V4",
    "NexusBench",
]


def download_evaluation_repo(
    repo_id: str,
    *,
    dest_parent: Path,
    allow_patterns: Optional[List[str]] = None,
    local_dir_name: Optional[str] = None,
) -> Path:
    """Download one HF dataset repo into ``dest_parent/<local_dir_name>/``.

    ``local_dir_name`` defaults to the repo suffix (e.g. ``BFCL-trajectories``); the pipeline
    benchmark paths pass ``<benchmark>-evaluation`` so the loader finds the data.
    Returns the path to the created directory.
    """
    local_dir = dest_parent / (local_dir_name or repo_id.split("/")[-1])
    print(f"\n=== Downloading {repo_id} → {local_dir} ===")
    if allow_patterns:
        print(f"    allow_patterns={allow_patterns}")

    max_retries = 3
    retry_delay = 5.0

    for attempt in range(max_retries):
        try:
            cache_dir = snapshot_download(
                repo_id=repo_id,
                repo_type="dataset",
                local_dir=str(local_dir),
                local_dir_use_symlinks=False,
                allow_patterns=allow_patterns,
                max_workers=4,
                resume_download=True,
                token=True,
            )
            print(f"    Done. Local dir: {cache_dir}")
            return local_dir

        except Exception as e:
            err = str(e).lower()
            if "429" in str(e) or "rate limit" in err:
                if attempt < max_retries - 1:
                    print(
                        f"    Rate limited. Waiting {retry_delay:.0f}s "
                        f"(retry {attempt + 2}/{max_retries})"
                    )
                    time.sleep(retry_delay)
                    retry_delay *= 2
                    continue
                print(f"    Failed after {max_retries} attempts (rate limiting)")
            else:
                print(f"    Error: {e}")
            raise


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description=(
            "Download evaluation JSONL from Hugging Face into the pipeline "
            "``benchmark/`` tree (consumable by BenchmarkDataLoader / main.py)."
        )
    )
    benchmark_choices = sorted(BENCHMARK_VALUE_TO_HF_REPO.keys())
    p.add_argument(
        "--benchmark",
        action="append",
        dest="benchmarks",
        metavar="NAME",
        choices=benchmark_choices,
        help=(
            "Pipeline benchmark name (same as main.py --target-benchmark), e.g. complex-func-bench. "
            "Repeat to download several. If set with --repo, both are downloaded."
        ),
    )
    p.add_argument(
        "--repo",
        action="append",
        dest="repos",
        metavar="REPO_ID",
        help=(
            "Full HF dataset repo_id (e.g. AgentSuite/tau-bench-trajectories). "
            "Repeat to add multiple. Downloaded into a folder named after the repo suffix, "
            "so for the pipeline loader prefer --benchmark (which uses <benchmark>-evaluation). "
            "If neither --benchmark nor --repo is given, defaults to the built-in list "
            f"({len(DEFAULT_PIPELINE_BENCHMARKS)} benchmarks; use --include-optional for more)."
        ),
    )
    p.add_argument(
        "--include-optional",
        action="store_true",
        help=(
            f"When downloading the default batch only: also fetch {len(OPTIONAL_EXTRA_BENCHMARKS)} "
            f"extra benchmarks ({', '.join(OPTIONAL_EXTRA_BENCHMARKS)}). "
            "Ignored if you pass --benchmark or --repo."
        ),
    )
    p.add_argument(
        "--patterns",
        nargs="+",
        metavar="GLOB",
        help='Limit files, e.g. --patterns "*.jsonl"',
    )
    p.add_argument(
        "--out",
        type=Path,
        default=None,
        help=(
            "Parent directory for ``<name>-evaluation/`` folders "
            f"(default: {REPO_ROOT / 'benchmark'})."
        ),
    )
    return p.parse_args()


def main() -> None:
    args = parse_args()
    dest_parent = (args.out if args.out is not None else REPO_ROOT / "benchmark").resolve()

    # Build (repo_id, local_dir_name) targets.
    targets: List[Tuple[str, str]] = []
    if args.benchmarks:
        targets.extend(
            (BENCHMARK_VALUE_TO_HF_REPO[name], local_dir_name(name)) for name in args.benchmarks
        )
    if args.repos:
        # Ad-hoc repos: keep the repo suffix as the folder name (caller's responsibility).
        targets.extend((repo_id, repo_id.split("/")[-1]) for repo_id in args.repos)
    if not targets:
        names = list(DEFAULT_PIPELINE_BENCHMARKS)
        if args.include_optional:
            names.extend(OPTIONAL_EXTRA_BENCHMARKS)
        targets = [(BENCHMARK_VALUE_TO_HF_REPO[name], local_dir_name(name)) for name in names]

    # Stable order, drop duplicate (repo, dir) pairs while preserving first occurrence.
    targets = list(dict.fromkeys(targets))

    patterns: Optional[List[str]] = args.patterns

    print(f"huggingface_hub {huggingface_hub.__version__}")
    print(f"Destination (benchmark root): {dest_parent}")
    dest_parent.mkdir(parents=True, exist_ok=True)

    errors: List[tuple[str, str]] = []
    for repo_id, ldir in targets:
        try:
            download_evaluation_repo(
                repo_id,
                dest_parent=dest_parent,
                allow_patterns=patterns,
                local_dir_name=ldir,
            )
        except Exception as e:
            errors.append((repo_id, str(e)))
            print(f"!! ERROR downloading {repo_id}: {e}", file=sys.stderr)

    if errors:
        print("\nSome downloads failed:")
        for repo_id, msg in errors:
            print(f"  - {repo_id}: {msg}")
        sys.exit(1)

    print("\nAll requested evaluation datasets downloaded successfully.")
    print(
        "Run the pipeline from the repo root, e.g.: "
        "python main.py --target-benchmark complex-func-bench --llm-max-samples 10"
    )


if __name__ == "__main__":
    main()
