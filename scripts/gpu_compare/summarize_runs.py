#!/usr/bin/env python3
"""Summarize GPU comparison stdout logs, nvidia-smi samples, and summary.json files."""

from __future__ import annotations

import argparse
import csv
import json
import math
import re
from collections import defaultdict
from pathlib import Path


TRAIN_RE = re.compile(
    r"Train: Iter=(?P<iter>\d+).*?train_loss=(?P<loss>[-+0-9.eE]+).*?"
    r"iter_dt=(?P<iter_dt>[-+0-9.eE]+)s.*?elapsed=(?P<elapsed>[-+0-9.eE]+)s.*?"
    r"avg_iter_dt=(?P<avg_iter_dt>[-+0-9.eE]+)s.*?lr=(?P<lr>[-+0-9.eE]+)"
)
EVAL_RE = re.compile(
    r">Eval: Iter=(?P<iter>\d+).*?val_loss=(?P<val_loss>[-+0-9.eE]+).*?"
    r"val_pp=(?P<val_pp>[-+0-9.eE]+)"
)
FAIL_PATTERNS = ("Traceback", "RuntimeError", "CUDA out of memory", "OutOfMemoryError", "nan", "NaN")


def mean(values: list[float]) -> float:
    return sum(values) / len(values) if values else math.nan


def median(values: list[float]) -> float:
    if not values:
        return math.nan
    values = sorted(values)
    mid = len(values) // 2
    if len(values) % 2:
        return values[mid]
    return (values[mid - 1] + values[mid]) / 2


def p95(values: list[float]) -> float:
    if not values:
        return math.nan
    values = sorted(values)
    idx = min(len(values) - 1, math.ceil(0.95 * len(values)) - 1)
    return values[idx]


def parse_run_name(name: str) -> dict[str, str]:
    meta: dict[str, str] = {}
    if "_capacity_" in name:
        meta["phase"] = "capacity"
    elif "_track3-probe_" in name:
        meta["phase"] = "track3-probe"
    elif "_benchmark-probe_" in name:
        meta["phase"] = "benchmark-probe"
    elif "_ddp-smoke_" in name:
        meta["phase"] = "ddp-smoke"
    for part in name.split("_"):
        if "-" not in part:
            continue
        key, value = part.split("-", 1)
        meta[key] = value
    return meta


def parse_log(path: Path) -> tuple[list[dict], list[dict], bool, int | None]:
    trains: list[dict] = []
    evals: list[dict] = []
    failed = False
    returncode: int | None = None
    for line in path.read_text(encoding="utf-8", errors="ignore").splitlines():
        train_match = TRAIN_RE.search(line)
        if train_match:
            trains.append(
                {
                    "iter": int(train_match.group("iter")),
                    "train_loss": float(train_match.group("loss")),
                    "iter_dt": float(train_match.group("iter_dt")),
                    "elapsed": float(train_match.group("elapsed")),
                    "avg_iter_dt": float(train_match.group("avg_iter_dt")),
                    "lr": float(train_match.group("lr")),
                }
            )
        eval_match = EVAL_RE.search(line)
        if eval_match:
            evals.append(
                {
                    "iter": int(eval_match.group("iter")),
                    "val_loss": float(eval_match.group("val_loss")),
                    "val_pp": float(eval_match.group("val_pp")),
                }
            )
        if any(pattern in line for pattern in FAIL_PATTERNS):
            failed = True
        if line.startswith("# returncode:"):
            try:
                returncode = int(line.split(":", 1)[1].strip())
            except ValueError:
                failed = True
    if returncode not in (None, 0):
        failed = True
    return trains, evals, failed, returncode


def parse_smi(path: Path) -> dict:
    if not path.exists():
        return {
            "gpu_peak_total_mib": math.nan,
            "gpu_peak_per_gpu_mib": math.nan,
            "gpu_steady_median_total_mib": math.nan,
            "gpu_avg_utilization_pct": math.nan,
            "gpu_samples": 0,
        }
    total_by_timestamp: defaultdict[str, float] = defaultdict(float)
    per_gpu_memory: list[float] = []
    utils: list[float] = []
    with path.open("r", encoding="utf-8", errors="ignore", newline="") as handle:
        reader = csv.DictReader(handle)
        for row in reader:
            try:
                timestamp = row["timestamp"]
                memory = float(row["memory_used_mib"])
                util = float(row["utilization_gpu_pct"])
            except (KeyError, TypeError, ValueError):
                continue
            total_by_timestamp[timestamp] += memory
            per_gpu_memory.append(memory)
            utils.append(util)
    totals = list(total_by_timestamp.values())
    nonzero_totals = [value for value in totals if value > 0]
    steady_source = nonzero_totals if nonzero_totals else totals
    return {
        "gpu_peak_total_mib": max(totals) if totals else math.nan,
        "gpu_peak_per_gpu_mib": max(per_gpu_memory) if per_gpu_memory else math.nan,
        "gpu_steady_median_total_mib": median(steady_source),
        "gpu_avg_utilization_pct": mean(utils),
        "gpu_samples": len(per_gpu_memory),
    }


def load_summary(results_root: Path, run_name: str) -> dict:
    summary_path = results_root / run_name / "summary.json"
    if not summary_path.exists():
        return {}
    try:
        return json.loads(summary_path.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return {}


def summarize_one(log_path: Path, results_root: Path) -> dict:
    run_name = log_path.stem
    meta = parse_run_name(run_name)
    trains, evals, failed, returncode = parse_log(log_path)
    smi = parse_smi(log_path.with_name(f"{run_name}_smi.csv"))
    summary = load_summary(results_root, run_name)
    iter_dts = [row["iter_dt"] for row in trains]
    avg_iter_dt = summary.get("avg_iter_dt") or mean(iter_dts)
    try:
        effbs = float(meta.get("effbs", "nan"))
        seq = float(meta.get("seq", "nan"))
        tokens_per_step = effbs * seq
        tokens_per_second = tokens_per_step / avg_iter_dt if avg_iter_dt else math.nan
    except ValueError:
        tokens_per_step = math.nan
        tokens_per_second = math.nan
    last_eval = evals[-1] if evals else {}
    return {
        "run_name": run_name,
        "gpu": meta.get("gpu", ""),
        "phase": meta.get("phase", ""),
        "model": meta.get("model", ""),
        "seq": meta.get("seq", ""),
        "effbs": meta.get("effbs", ""),
        "microbs": meta.get("microbs", ""),
        "acc": meta.get("acc", ""),
        "world": meta.get("world", ""),
        "opt": meta.get("opt", ""),
        "seed": meta.get("seed", ""),
        "train_points": len(trains),
        "final_train_loss": trains[-1]["train_loss"] if trains else math.nan,
        "final_val_loss": last_eval.get("val_loss", math.nan),
        "avg_iter_dt": avg_iter_dt,
        "median_iter_dt": median(iter_dts),
        "p95_iter_dt": p95(iter_dts),
        "tokens_per_step": tokens_per_step,
        "tokens_per_second": tokens_per_second,
        "wall_clock_seconds": summary.get("wall_clock_seconds", math.nan),
        **smi,
        "failed": failed or not trains,
        "returncode": returncode if returncode is not None else "",
        "log_path": str(log_path),
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--log-dir", default="/root/work/llmopt-results/5090-comparison-20260701/logs")
    parser.add_argument("--results-root", default="/root/autodl-tmp/llmopt/exps/5090-comparison-20260701")
    parser.add_argument("--output", default="/root/work/llmopt-results/5090-comparison-20260701/logs/results.csv")
    args = parser.parse_args()

    log_dir = Path(args.log_dir)
    results_root = Path(args.results_root)
    rows = [
        summarize_one(path, results_root)
        for path in sorted(log_dir.glob("*.log"))
        if not path.name.endswith("_smi.log")
    ]
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "run_name",
        "gpu",
        "phase",
        "model",
        "seq",
        "effbs",
        "microbs",
        "acc",
        "world",
        "opt",
        "seed",
        "train_points",
        "final_train_loss",
        "final_val_loss",
        "avg_iter_dt",
        "median_iter_dt",
        "p95_iter_dt",
        "tokens_per_step",
        "tokens_per_second",
        "wall_clock_seconds",
        "gpu_peak_total_mib",
        "gpu_peak_per_gpu_mib",
        "gpu_steady_median_total_mib",
        "gpu_avg_utilization_pct",
        "gpu_samples",
        "failed",
        "returncode",
        "log_path",
    ]
    with output.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    print(f"wrote {len(rows)} rows to {output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
