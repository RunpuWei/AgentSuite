#!/usr/bin/env python3
"""Validate a leaderboard trajectory submission.

A submission is a small YAML *manifest* (``leaderboard/submissions/<model>.yaml``)
that points at one trajectory JSONL per benchmark, hosted on Hugging Face. We never
commit the multi-MB trajectories into git: the manifest carries pointers, CI pulls
the JSONL from HF, and scores are always recomputed server-side
(``compute_split_scores.py``) — uploaded numbers are never trusted.

This tool enforces two things before a submission is accepted:

1. **Schema** — every JSONL row has the fields the scorer needs
   (``messages``, ``task_name``, ``meta.id`` and a score signal).
2. **Completeness** — for each benchmark it submits, the trajectory covers the
   *full* in-scope task set (no cherry-picking). The canonical id set per benchmark
   is the committed ``leaderboard/expected_ids/<bench>.txt`` (built once from the
   reference trajectories via ``--build-expected``). Submitting only *some*
   benchmarks is fine — partial-coverage models simply don't enter the common
   overall rank (see ``aggregate_overall`` in ``compute_split_scores.py``).

Usage::

    # build the committed expected-id lists from local reference trajectories
    python leaderboard/validate_submission.py --build-expected \
        --traj-root pipeline/benchmark

    # validate a single local JSONL against a benchmark (used in verification/tests)
    python leaderboard/validate_submission.py \
        --jsonl path/to/model.jsonl --benchmark ACEBench

    # validate a full submission manifest (downloads JSONL from HF)
    python leaderboard/validate_submission.py --manifest leaderboard/submissions/grok-4.yaml
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
EXPECTED_DIR = Path(__file__).resolve().parent / "expected_ids"

# Reuse the scorer's per-benchmark conventions verbatim — id normalization, task
# scoping and the score predicate must match exactly, or validation and scoring
# could disagree.
sys.path.insert(0, str(REPO_ROOT))
from compute_split_scores import (  # noqa: E402
    FILTERING_IDS,
    TASK_SCOPE,
    _normalize_id,
    _score,
)

# Benchmarks that appear on the leaderboard (have a reference trajectory dir and a
# filtering id-list). DrafterBench is excluded — no filtering, not a split benchmark.
LEADERBOARD_BENCHMARKS = [b for b in FILTERING_IDS if b != "DrafterBench"]


# --------------------------------------------------------------------------- #
# Loading
# --------------------------------------------------------------------------- #
def _read_jsonl(path):
    rows = []
    with open(path) as f:
        for n, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError as e:
                raise ValueError(f"{path}:{n}: invalid JSON ({e})") from None
    return rows


def _in_scope(rows, benchmark):
    scope = TASK_SCOPE.get(benchmark)
    return [r for r in rows if scope(r.get("task_name"))] if scope else list(rows)


def _raw_ids(rows):
    return {(r.get("meta") or {}).get("id") for r in rows}


# --------------------------------------------------------------------------- #
# Expected-id lists (canonical full task set per benchmark)
# --------------------------------------------------------------------------- #
def build_expected(traj_root):
    """Write leaderboard/expected_ids/<bench>.txt from the reference trajectories.

    The expected set is the union of in-scope raw meta.id across every reference
    model for that benchmark — i.e. every task any complete run attempted.
    """
    EXPECTED_DIR.mkdir(parents=True, exist_ok=True)
    written = []
    for benchmark in LEADERBOARD_BENCHMARKS:
        d = traj_root / f"{benchmark}-evaluation"
        if not d.is_dir():
            print(f"  skip {benchmark}: no reference dir at {d}")
            continue
        ids = set()
        n_models = 0
        for p in sorted(d.glob("*.jsonl")):
            rows = _in_scope(_read_jsonl(p), benchmark)
            ids |= {i for i in _raw_ids(rows) if i}
            n_models += 1
        out = EXPECTED_DIR / f"{benchmark}.txt"
        out.write_text("\n".join(sorted(ids)) + "\n")
        written.append((benchmark, len(ids), n_models, out))
        print(f"  {benchmark:20s} {len(ids):5d} ids  (from {n_models} reference models) -> {out}")
    return written


def load_expected(benchmark):
    path = EXPECTED_DIR / f"{benchmark}.txt"
    if not path.exists():
        raise FileNotFoundError(
            f"no expected-id list for {benchmark} at {path}; "
            f"run `--build-expected` first")
    return {ln.strip() for ln in path.read_text().splitlines() if ln.strip()}


# --------------------------------------------------------------------------- #
# Validation
# --------------------------------------------------------------------------- #
def validate_jsonl(path, benchmark):
    """Return (ok, errors, info) for one trajectory file against one benchmark."""
    errors, info = [], {}
    if benchmark not in LEADERBOARD_BENCHMARKS:
        return False, [f"unknown benchmark '{benchmark}' "
                       f"(expected one of {LEADERBOARD_BENCHMARKS})"], info

    rows = _read_jsonl(path)
    if not rows:
        return False, [f"{path}: empty trajectory"], info

    # 1. schema — every row must carry what the scorer reads.
    for n, r in enumerate(rows, 1):
        if not isinstance(r.get("messages"), list) or not r["messages"]:
            errors.append(f"row {n}: missing/empty 'messages'")
        if not (r.get("task_name")):
            errors.append(f"row {n}: missing 'task_name'")
        meta = r.get("meta") or {}
        if not meta.get("id"):
            errors.append(f"row {n}: missing 'meta.id'")
        er = r.get("eval_result") or {}
        has_score = er.get("score") is not None
        has_correct = isinstance(meta.get("is_correct"), bool)
        if not (has_score or has_correct):
            errors.append(f"row {n}: no score signal (eval_result.score or meta.is_correct)")
        elif has_score:
            try:
                s = float(er["score"])
                if not (0.0 <= s <= 1.0):
                    errors.append(f"row {n}: eval_result.score {s} out of [0,1]")
            except (TypeError, ValueError):
                errors.append(f"row {n}: eval_result.score not numeric ({er.get('score')!r})")
        # exercise the real score path so a row that scores cannot later crash.
        try:
            _score(r, benchmark)
        except Exception as e:  # pragma: no cover - defensive
            errors.append(f"row {n}: scorer rejected record ({e})")
        if len(errors) >= 50:
            errors.append("… (further schema errors suppressed)")
            break

    # 2. completeness — full in-scope task coverage, no cherry-picking.
    scoped = _in_scope(rows, benchmark)
    present = {i for i in _raw_ids(scoped) if i}
    expected = load_expected(benchmark)
    missing = expected - present
    extra = present - expected
    info.update(rows=len(rows), in_scope=len(scoped),
                expected=len(expected), present=len(present),
                missing=len(missing), extra=len(extra))
    if missing:
        sample = ", ".join(sorted(missing)[:10])
        errors.append(f"incomplete: {len(missing)}/{len(expected)} expected task ids "
                      f"absent (e.g. {sample}{'…' if len(missing) > 10 else ''})")
    if extra:
        # non-fatal: unexpected ids (e.g. a benchmark revision) are reported, not blocked.
        info["extra_sample"] = sorted(extra)[:10]
    return (not errors), errors, info


def _resolve_source(src):
    """Resolve a manifest trajectory pointer to a local path.

    Accepts a local path or an ``hf://datasets/<repo>/<path…>`` URI; the latter is
    fetched from Hugging Face (public datasets need no auth; private need HF_TOKEN).
    """
    if not src.startswith("hf://"):
        p = Path(src)
        if not p.is_absolute():
            p = REPO_ROOT / p
        return p
    rest = src[len("hf://"):]
    if not rest.startswith("datasets/"):
        raise ValueError(f"hf:// pointer must be 'hf://datasets/<repo>/<path>': {src}")
    rest = rest[len("datasets/"):]
    parts = rest.split("/")
    if len(parts) < 3:
        raise ValueError(f"hf:// pointer needs <org>/<repo>/<path>: {src}")
    repo_id = "/".join(parts[:2])
    filename = "/".join(parts[2:])
    from huggingface_hub import hf_hub_download
    return Path(hf_hub_download(repo_id=repo_id, filename=filename, repo_type="dataset"))


def validate_manifest(path):
    import yaml  # lazy: only manifest mode needs it
    spec = yaml.safe_load(Path(path).read_text())
    model = spec.get("model")
    trajs = spec.get("trajectories") or {}
    if not model:
        return False, {"_manifest": ["missing 'model'"]}
    if not trajs:
        return False, {"_manifest": ["no 'trajectories' entries"]}
    results = {}
    overall_ok = True
    for benchmark, src in trajs.items():
        try:
            local = _resolve_source(str(src))
            ok, errors, info = validate_jsonl(local, benchmark)
        except Exception as e:
            ok, errors, info = False, [f"could not load '{src}': {e}"], {}
        results[benchmark] = {"ok": ok, "errors": errors, "info": info}
        overall_ok = overall_ok and ok
    return overall_ok, results


# --------------------------------------------------------------------------- #
# CLI
# --------------------------------------------------------------------------- #
def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--build-expected", action="store_true",
                    help="(Re)generate leaderboard/expected_ids/<bench>.txt from reference trajectories.")
    ap.add_argument("--traj-root", type=Path, default=REPO_ROOT / "pipeline" / "benchmark",
                    help="Root of <bench>-evaluation/ reference dirs (for --build-expected).")
    ap.add_argument("--jsonl", type=Path, default=None, help="A single local trajectory to validate.")
    ap.add_argument("--benchmark", default=None, help="Benchmark for --jsonl.")
    ap.add_argument("--manifest", type=Path, default=None, help="A submission manifest YAML to validate.")
    args = ap.parse_args()

    if args.build_expected:
        print("Building expected-id lists:")
        build_expected(args.traj_root)
        return 0

    if args.jsonl is not None:
        if not args.benchmark:
            ap.error("--jsonl requires --benchmark")
        ok, errors, info = validate_jsonl(args.jsonl, args.benchmark)
        print(f"[{args.benchmark}] {args.jsonl}")
        print(f"  rows={info.get('rows')} in_scope={info.get('in_scope')} "
              f"present={info.get('present')}/{info.get('expected')} "
              f"missing={info.get('missing')} extra={info.get('extra')}")
        for e in errors:
            print(f"  ✗ {e}")
        print("  RESULT:", "PASS" if ok else "FAIL")
        return 0 if ok else 1

    if args.manifest is not None:
        overall_ok, results = validate_manifest(args.manifest)
        print(f"Manifest: {args.manifest}")
        for benchmark, r in results.items():
            i = r["info"]
            status = "PASS" if r["ok"] else "FAIL"
            cov = (f"  ({i.get('present')}/{i.get('expected')} ids)"
                   if i.get("expected") else "")
            print(f"  [{status}] {benchmark}{cov}")
            for e in r["errors"]:
                print(f"      ✗ {e}")
        print("OVERALL:", "PASS" if overall_ok else "FAIL")
        return 0 if overall_ok else 1

    ap.error("nothing to do: pass --build-expected, --jsonl/--benchmark, or --manifest")


if __name__ == "__main__":
    raise SystemExit(main())
