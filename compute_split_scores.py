#!/usr/bin/env python3
"""Compute leaderboard split scores from HF trajectory datasets.

The public ``AgentSuite/<bench>-trajectories`` datasets ship one JSONL per model,
already scored per task. Each record carries::

    {"benchmark_name": ..., "task_name": ..., "messages": [...],
     "eval_result": {"score": <float 0..1>},
     "meta": {"id": "<task_id>", "is_correct": <bool>, ...}}

The headline metric is micro = mean(eval_result.score) over a model's task rows
(partial credit on multi-turn tasks, so the fractional score — not boolean
is_correct — is authoritative). From one trajectory set this tool produces two
columns:

    Original  = micro over ALL task rows
    Filtering = micro over rows whose meta.id is NOT in <bench>_filtering_ids.csv

Point it at the *original* trajectories for the Original/Filtering launch splits,
or at the *fixed* trajectories for the Fixed split (same computation either way).

Usage::

    # all <bench>-evaluation dirs under a root (default pipeline/benchmark)
    python compute_split_scores.py --traj-root pipeline/benchmark --out-dir results/splits

    # a single benchmark from an explicit dir
    python compute_split_scores.py --benchmark ACEBench \
        --traj-dir _scratch/corrected_2026-05-29/hf_trajectories_ACEBench-evaluation
"""

import argparse
import csv
import json
import math
import re
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent

# Benchmark value (as used in <value>-evaluation trajectory dir names, matching
# download_model_trajectory.py) -> filtering id-list, relative to the repo root.
# DrafterBench has no filtering (issues fixed upstream) -> None.
FILTERING_IDS = {
    "tau-bench": "tau-bench/taubench_filtering_ids.csv",
    "tau2-bench": "tau2-bench/tau2bench_filtering_ids.csv",
    "ACEBench": "ACEBench/acebench_filtering_ids.csv",
    "complex-func-bench": "ComplexFuncBench/cfbench_filtering_ids.csv",
    "NexusBench": "NexusBench/nexusbench_filtering_ids.csv",
    "BFCL": "gorilla/berkeley-function-call-leaderboard/bfcl_v3_filtering_ids.csv",
    "BFCL_V4": "gorilla/berkeley-function-call-leaderboard/bfcl_v4_filtering_ids.csv",
    "DrafterBench": None,
}

# Some trajectory sets carry tasks beyond a benchmark's scored domain. The BFCL
# trajectories we use are version-specific: v3 is scored on its multi_turn domain
# only (~800 tasks), v4 on its web_search + memory domain only (~665; the large
# format_sensitivity category is excluded). task_name predicate -> keep the record.
TASK_SCOPE = {
    "BFCL": lambda n: bool(n) and n.startswith("multi_turn"),
    "BFCL_V4": lambda n: bool(n) and ("memory" in n or "web_search" in n),
}


def _normalize_id(benchmark, tid):
    """Map a trajectory meta.id to the manifest's id form.

    BFCL_V4 evaluates each logical task under several memory backends
    (kv/rec_sum/vector) and web variants (base/no_snippet); the manifest holds the
    backend-free logical id, so strip the variant token before matching.
    """
    if benchmark == "BFCL_V4" and tid:
        tid = re.sub(r"^memory_(kv|rec_sum|vector)_", "memory_", tid)
        tid = re.sub(r"^web_search_(base|no_snippet)_", "web_search_", tid)
    return tid


def load_filtering_ids(benchmark):
    rel = FILTERING_IDS.get(benchmark)
    if not rel:
        return set()
    path = REPO_ROOT / rel
    if not path.exists():
        raise FileNotFoundError(f"filtering ids not found for {benchmark}: {path}")
    with open(path) as f:
        return {r["task_id"].strip() for r in csv.DictReader(f) if r.get("task_id")}


def _score(rec, benchmark):
    """Per-task score in [0,1]; eval_result.score is authoritative, is_correct is fallback.

    ACEBench multi_* tasks (agent_multi_turn/step, normal_multi_turn_*) carry partial
    credit, but the original ACEBench implementation grades them as binary task success
    — a multi_* task counts only if fully correct. Binarize those to align.
    """
    er = rec.get("eval_result") or {}
    if "score" in er and er["score"] is not None:
        s = float(er["score"])
    else:
        s = 1.0 if (rec.get("meta") or {}).get("is_correct") else 0.0
    if benchmark == "ACEBench" and "multi" in (rec.get("task_name") or ""):
        return 1.0 if s >= 1.0 else 0.0
    return s


def _rank_key(v):
    """Sort key for descending rank with NaN (no data) always last."""
    nan = v != v
    return (nan, -v if not nan else 0.0)


def _micro(records, benchmark):
    if not records:
        return float("nan")
    return sum(_score(r, benchmark) for r in records) / len(records)


def _macro(records, benchmark):
    """Mean over task_name groups of each group's micro score."""
    groups = {}
    for r in records:
        groups.setdefault(r.get("task_name", ""), []).append(r)
    if not groups:
        return float("nan")
    return sum(_micro(v, benchmark) for v in groups.values()) / len(groups)


def score_model_file(path, filtering_ids, benchmark):
    records = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if line:
                records.append(json.loads(line))
    scope = TASK_SCOPE.get(benchmark)
    if scope:
        records = [r for r in records if scope(r.get("task_name"))]

    # Dedup by RAW meta.id (keep first occurrence). Some trajectory files are
    # accidental concatenations of two runs (e.g. NexusBench gpt-4.1-mini =
    # 2217 rows / 1118 ids), which would otherwise bias the mean toward the
    # run that happens to duplicate. Dedup on the raw id, NOT the normalized id
    # — BFCL_V4 normalization maps distinct variant ids onto one logical id,
    # so deduping on the normalized id would wrongly collapse real tasks.
    seen = set()
    deduped = []
    for r in records:
        raw_id = (r.get("meta") or {}).get("id")
        if raw_id is not None:
            if raw_id in seen:
                continue
            seen.add(raw_id)
        deduped.append(r)
    records = deduped

    def norm(r):
        return _normalize_id(benchmark, (r.get("meta") or {}).get("id"))

    kept = [r for r in records if norm(r) not in filtering_ids]
    present_ids = {norm(r) for r in records}
    return {
        "model": path.stem,
        "n_all": len(records),
        "n_filtered": len(kept),
        "n_dropped": len(records) - len(kept),
        "filter_ids_matched": len(filtering_ids & present_ids),
        "filter_ids_total": len(filtering_ids),
        "original_micro": _micro(records, benchmark),
        "filtering_micro": _micro(kept, benchmark),
        "original_macro": _macro(records, benchmark),
        "filtering_macro": _macro(kept, benchmark),
    }


def score_benchmark(benchmark, traj_dir):
    filtering_ids = load_filtering_ids(benchmark)
    rows = [score_model_file(p, filtering_ids, benchmark) for p in sorted(traj_dir.glob("*.jsonl"))]
    # Primary rank = Original (the launch leaderboard ranks by Original; Filtering is
    # the side rank showing how cleaning shifts positions).
    rows.sort(key=lambda r: _rank_key(r["original_micro"]))
    for i, r in enumerate(rows, 1):
        r["rank"] = i
    # Secondary rank = Filtering.
    for i, r in enumerate(sorted(rows, key=lambda r: _rank_key(r["filtering_micro"])), 1):
        r["filtering_rank"] = i
    return rows


def write_outputs(benchmark, rows, out_dir):
    out_dir.mkdir(parents=True, exist_ok=True)
    cols = ["rank", "filtering_rank", "model", "original_micro", "filtering_micro",
            "original_macro", "filtering_macro",
            "n_all", "n_filtered", "n_dropped",
            "filter_ids_matched", "filter_ids_total"]
    csv_path = out_dir / f"{benchmark}_split_scores.csv"
    with open(csv_path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=cols)
        w.writeheader()
        for r in rows:
            w.writerow({c: r[c] for c in cols})
    return csv_path


def aggregate_overall(combined, out_dir):
    """Cross-benchmark macro leaderboard.

    Each benchmark is scored independently (the per-model micro), then the headline
    is the *equal-weight mean across benchmarks* — NOT a global task pool, so the
    large benchmarks (1000+ tasks) don't drown out the small ones (~165). The
    overall ranking is restricted to the model set common to ALL scored benchmarks
    (the intersection) so every overall number averages the same benchmarks; models
    missing from one or more benchmarks (e.g. the 4 absent from BFCL_v3) are reported
    separately as "partial" and excluded from the overall rank. Primary rank is by
    Original-avg; Filtering-avg is the side rank.
    """
    benchmarks = list(combined.keys())
    by_model = {}  # model -> benchmark -> (original_micro, filtering_micro)
    for b, rows in combined.items():
        for r in rows:
            by_model.setdefault(r["model"], {})[b] = (r["original_micro"], r["filtering_micro"])

    common = sorted(m for m, d in by_model.items() if all(b in d for b in benchmarks))
    partial = sorted(m for m, d in by_model.items() if not all(b in d for b in benchmarks))

    def _avg(model, idx):
        vals = [by_model[model][b][idx] for b in benchmarks]
        return sum(vals) / len(vals)

    overall = []
    for m in common:
        row = {"model": m,
               "original_avg": _avg(m, 0),
               "filtering_avg": _avg(m, 1)}
        for b in benchmarks:
            row[f"{b}_original"] = by_model[m][b][0]
        overall.append(row)
    overall.sort(key=lambda r: _rank_key(r["original_avg"]))
    for i, r in enumerate(overall, 1):
        r["rank"] = i
    for i, r in enumerate(sorted(overall, key=lambda r: _rank_key(r["filtering_avg"])), 1):
        r["filtering_rank"] = i

    out_dir.mkdir(parents=True, exist_ok=True)
    cols = (["rank", "filtering_rank", "model", "original_avg", "filtering_avg"]
            + [f"{b}_original" for b in benchmarks])
    with open(out_dir / "overall_leaderboard.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=cols)
        w.writeheader()
        for r in overall:
            w.writerow({c: r.get(c, "") for c in cols})

    # Partial-coverage models: per-benchmark Original where available, no overall rank.
    if partial:
        pcols = ["model", "n_benchmarks", "missing"] + [f"{b}_original" for b in benchmarks]
        with open(out_dir / "overall_partial_models.csv", "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=pcols)
            w.writeheader()
            for m in partial:
                present = by_model[m]
                w.writerow({"model": m,
                            "n_benchmarks": len(present),
                            "missing": ";".join(b for b in benchmarks if b not in present),
                            **{f"{b}_original": present.get(b, ("",))[0] for b in benchmarks}})
    return overall, partial, benchmarks


def main():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--traj-root", type=Path, default=REPO_ROOT / "pipeline" / "benchmark",
                   help="Root holding <benchmark>-evaluation/ dirs (default: pipeline/benchmark).")
    p.add_argument("--traj-dir", type=Path, default=None,
                   help="Explicit single trajectory dir (requires --benchmark).")
    p.add_argument("--benchmark", default=None,
                   help="Benchmark value (e.g. ACEBench); required with --traj-dir.")
    p.add_argument("--out-dir", type=Path, default=REPO_ROOT / "results" / "splits")
    args = p.parse_args()

    if args.traj_dir is not None:
        if not args.benchmark:
            p.error("--traj-dir requires --benchmark")
        targets = [(args.benchmark, args.traj_dir)]
    else:
        targets = []
        for benchmark in FILTERING_IDS:
            d = args.traj_root / f"{benchmark}-evaluation"
            if d.is_dir():
                targets.append((benchmark, d))
        if not targets:
            p.error(f"no <benchmark>-evaluation/ dirs found under {args.traj_root}")

    combined = {}
    for benchmark, traj_dir in targets:
        rows = score_benchmark(benchmark, traj_dir)
        csv_path = write_outputs(benchmark, rows, args.out_dir)
        combined[benchmark] = rows
        print(f"\n=== {benchmark}  ({len(rows)} models, {traj_dir}) -> {csv_path} ===")
        if rows:
            r0 = rows[0]
            print(f"  filter ids matched: {r0['filter_ids_matched']}/{r0['filter_ids_total']}  "
                  f"tasks {r0['n_all']} -> {r0['n_filtered']} (dropped {r0['n_dropped']})")
            print(f"  {'model':40s} {'Original':>9s} {'Filtering':>9s} {'Δ':>7s}")
            for r in rows[:10]:
                d = r["filtering_micro"] - r["original_micro"]
                print(f"  {r['model']:40s} {r['original_micro']*100:9.2f} "
                      f"{r['filtering_micro']*100:9.2f} {d*100:+7.2f}")

    with open(args.out_dir / "split_scores.json", "w") as f:
        json.dump(combined, f, indent=2)
    print(f"\nWrote {args.out_dir / 'split_scores.json'}")

    # Cross-benchmark macro leaderboard (overall), ranked by Original-avg.
    if len(combined) > 1:
        overall, partial, benchmarks = aggregate_overall(combined, args.out_dir)
        print(f"\n=== OVERALL  (macro-avg over {len(benchmarks)} benchmarks, "
              f"{len(overall)} common models; {len(partial)} partial excluded) ===")
        print(f"  {'rank':>4s} {'model':40s} {'Orig-avg':>9s} {'Filt-avg':>9s} {'(Filt rk)':>9s}")
        for r in overall[:15]:
            print(f"  {r['rank']:4d} {r['model']:40s} {r['original_avg']*100:9.2f} "
                  f"{r['filtering_avg']*100:9.2f} {r['filtering_rank']:9d}")
        if partial:
            print(f"  partial (no overall rank): {', '.join(partial)}")
        print(f"\nWrote {args.out_dir / 'overall_leaderboard.csv'}")


if __name__ == "__main__":
    main()
