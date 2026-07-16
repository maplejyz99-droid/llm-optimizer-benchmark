#!/usr/bin/env python3
"""Parse LR sweep logs into a compact CSV."""

from __future__ import annotations

import argparse
import csv
import json
import math
import re
from pathlib import Path


SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parents[2]

TRAIN_RE = re.compile(
    r"Train: Iter=(?P<iter>\d+).*?train_loss=(?P<loss>[-+0-9.eE]+).*?"
    r"iter_dt=(?P<iter_dt>[-+0-9.eE]+)s.*?lr=(?P<lr>[-+0-9.eE]+)"
)
EVAL_RE = re.compile(
    r">Eval: Iter=(?P<iter>\d+).*?val_loss=(?P<val_loss>[-+0-9.eE]+).*?"
    r"val_pp=(?P<val_pp>[-+0-9.eE]+)"
)
FAIL_PATTERNS = ("Traceback", "RuntimeError", "CUDA out of memory", "nan", "NaN")


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
                    "logged_lr": float(train_match.group("lr")),
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


def mean(values: list[float]) -> float:
    return sum(values) / len(values) if values else math.nan


def parse_gpu(path: Path) -> dict:
    if not path.exists():
        return {
            "gpu_peak_memory_mib": math.nan,
            "gpu_avg_utilization_pct": math.nan,
            "gpu_samples": 0,
        }
    memory_values: list[float] = []
    util_values: list[float] = []
    with path.open("r", encoding="utf-8", errors="ignore") as handle:
        for row in csv.reader(handle):
            if not row or row[0] == "timestamp" or len(row) < 4:
                continue
            try:
                memory_values.append(float(row[2].strip()))
                util_values.append(float(row[3].strip()))
            except ValueError:
                continue
    return {
        "gpu_peak_memory_mib": max(memory_values) if memory_values else math.nan,
        "gpu_avg_utilization_pct": mean(util_values),
        "gpu_samples": len(memory_values),
    }


def summarize(log_path: Path, tail_fraction: float) -> dict:
    meta_path = log_path.with_suffix(".meta.json")
    meta = json.loads(meta_path.read_text(encoding="utf-8")) if meta_path.exists() else {}
    trains, evals, failed, returncode = parse_log(log_path)
    gpu = parse_gpu(log_path.with_suffix(".gpu.csv"))
    tail_n = max(1, math.ceil(len(trains) * tail_fraction)) if trains else 0
    tail = trains[-tail_n:] if tail_n else []
    last_eval = evals[-1] if evals else {}
    return {
        "phase": meta.get("phase", log_path.parent.name),
        "alias": meta.get("alias", ""),
        "model": meta.get("model", ""),
        "n_embd": meta.get("n_embd", ""),
        "n_head": meta.get("n_head", ""),
        "n_layer": meta.get("extra_args", {}).get("n_layer", ""),
        "batch_size": meta.get("extra_args", {}).get("batch_size", ""),
        "sequence_length": meta.get("extra_args", {}).get("sequence_length", ""),
        "acc_steps": meta.get("extra_args", {}).get("acc_steps", ""),
        "lr": meta.get("lr", ""),
        "iterations": meta.get("iterations", ""),
        "seed": meta.get("seed", ""),
        "num_train_points": len(trains),
        "loss_tail_mean": mean([row["train_loss"] for row in tail]),
        "final_train_loss": trains[-1]["train_loss"] if trains else math.nan,
        "avg_iter_dt": mean([row["iter_dt"] for row in trains]),
        "final_val_loss": last_eval.get("val_loss", math.nan),
        **gpu,
        "failed": failed or not trains,
        "returncode": returncode if returncode is not None else "",
        "log_path": str(log_path.relative_to(REPO_ROOT)),
    }


def write_probe_summary(rows: list[dict], output: Path) -> None:
    probe_rows = [row for row in rows if row["phase"].startswith("probe_v2")]
    output.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "n_embd",
        "phase",
        "batch_size",
        "sequence_length",
        "acc_steps",
        "sp_peak_memory_mib",
        "mup_peak_memory_mib",
        "max_peak_memory_mib",
        "sp_failed",
        "mup_failed",
        "passes_70gb",
        "passes_75gb",
    ]
    grouped: dict[tuple[str, int], dict[str, dict]] = {}
    for row in probe_rows:
        grouped.setdefault((row["phase"], int(row["n_embd"])), {})[row["alias"]] = row
    with output.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for phase, width in sorted(grouped):
            sp = grouped[(phase, width)].get("sp", {})
            mup = grouped[(phase, width)].get("mup", {})
            sp_peak = float(sp.get("gpu_peak_memory_mib", math.nan))
            mup_peak = float(mup.get("gpu_peak_memory_mib", math.nan))
            peaks = [value for value in [sp_peak, mup_peak] if not math.isnan(value)]
            max_peak = max(peaks) if peaks else math.nan
            sp_failed = str(sp.get("failed", True)) == "True"
            mup_failed = str(mup.get("failed", True)) == "True"
            writer.writerow(
                {
                    "n_embd": width,
                    "phase": phase,
                    "batch_size": sp.get("batch_size") or mup.get("batch_size", ""),
                    "sequence_length": sp.get("sequence_length") or mup.get("sequence_length", ""),
                    "acc_steps": sp.get("acc_steps") or mup.get("acc_steps", ""),
                    "sp_peak_memory_mib": sp_peak,
                    "mup_peak_memory_mib": mup_peak,
                    "max_peak_memory_mib": max_peak,
                    "sp_failed": sp_failed,
                    "mup_failed": mup_failed,
                    "passes_70gb": (not sp_failed and not mup_failed and max_peak <= 70000),
                    "passes_75gb": (not sp_failed and not mup_failed and max_peak <= 75000),
                }
            )


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--log-root", default=str(REPO_ROOT / "logs/mup_readme_lr_sweep"))
    parser.add_argument("--output", default=str(REPO_ROOT / "logs/mup_readme_lr_sweep/results.csv"))
    parser.add_argument("--include-phases", default=None)
    parser.add_argument("--probe-summary", default=None)
    parser.add_argument("--tail-fraction", type=float, default=0.2)
    args = parser.parse_args()

    log_root = Path(args.log_root)
    include_phases = set(args.include_phases.split(",")) if args.include_phases else None
    rows = []
    for path in sorted(log_root.glob("*/*.log")):
        row = summarize(path, args.tail_fraction)
        if include_phases is None or row["phase"] in include_phases:
            rows.append(row)
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "phase",
        "alias",
        "model",
        "n_embd",
        "n_head",
        "n_layer",
        "batch_size",
        "sequence_length",
        "acc_steps",
        "lr",
        "iterations",
        "seed",
        "num_train_points",
        "loss_tail_mean",
        "final_train_loss",
        "avg_iter_dt",
        "final_val_loss",
        "gpu_peak_memory_mib",
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
    if args.probe_summary:
        write_probe_summary(rows, Path(args.probe_summary))
        print(f"wrote probe summary to {args.probe_summary}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
