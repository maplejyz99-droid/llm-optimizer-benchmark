#!/usr/bin/env python3
"""Generate an A800-vs-5090 comparison report from normalized results.csv files."""

from __future__ import annotations

import argparse
import csv
import math
from collections import defaultdict
from pathlib import Path


def to_float(value: str) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return math.nan


def load_rows(path: Path) -> list[dict[str, str]]:
    if not path.exists():
        return []
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def row_key(row: dict[str, str]) -> tuple[str, ...]:
    run_name = row.get("run_name", "")
    fallback = "fallback" if "fallback" in run_name else "exact"
    return (
        row.get("phase", ""),
        row.get("model", ""),
        row.get("seq", ""),
        row.get("effbs", ""),
        row.get("microbs", ""),
        row.get("acc", ""),
        row.get("world", ""),
        row.get("opt", ""),
        fallback,
    )


def best_by_key(rows: list[dict[str, str]]) -> dict[tuple[str, ...], dict[str, str]]:
    grouped: defaultdict[tuple[str, ...], list[dict[str, str]]] = defaultdict(list)
    for row in rows:
        grouped[row_key(row)].append(row)
    result = {}
    for key, group in grouped.items():
        ok = [row for row in group if row.get("failed") == "False"]
        candidates = ok or group
        result[key] = sorted(candidates, key=lambda row: row.get("run_name", ""))[-1]
    return result


def fmt(value: float, digits: int = 3) -> str:
    if math.isnan(value):
        return "n/a"
    return f"{value:.{digits}f}"


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--a800", default="/root/work/llmopt-results/a800-20260701/logs/results.csv")
    parser.add_argument("--gpu", default="/root/work/llmopt-results/5090-comparison-20260701/logs/results.csv")
    parser.add_argument("--gpu-label", default="5090")
    parser.add_argument("--output", default="/root/work/llmopt-results/5090-comparison-20260701/reports/a800-vs-5090-comparison.md")
    args = parser.parse_args()

    a800_rows = load_rows(Path(args.a800))
    gpu_rows = load_rows(Path(args.gpu))
    a800 = best_by_key(a800_rows)
    gpu = best_by_key(gpu_rows)
    keys = sorted(set(a800) | set(gpu))

    lines = [
        f"# A800 vs {args.gpu_label} Comparison",
        "",
        f"A800 source: `{args.a800}`",
        f"{args.gpu_label} source: `{args.gpu}`",
        "",
        "| Phase | Model | Seq | Effbs | Micro | Acc | World | Opt | Mode | A800 dt | GPU dt | dt ratio | A800 tok/s | GPU tok/s | tok/s ratio | A800 MiB | GPU MiB | Status |",
        "| --- | --- | ---: | ---: | ---: | ---: | ---: | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- |",
    ]
    for key in keys:
        ar = a800.get(key)
        gr = gpu.get(key)
        phase, model, seq, effbs, microbs, acc, world, opt, mode = key
        adt = to_float(ar.get("avg_iter_dt")) if ar else math.nan
        gdt = to_float(gr.get("avg_iter_dt")) if gr else math.nan
        ats = to_float(ar.get("tokens_per_second")) if ar else math.nan
        gts = to_float(gr.get("tokens_per_second")) if gr else math.nan
        amem = to_float(ar.get("gpu_peak_per_gpu_mib")) if ar else math.nan
        gmem = to_float(gr.get("gpu_peak_per_gpu_mib")) if gr else math.nan
        dt_ratio = gdt / adt if adt and not math.isnan(adt) and not math.isnan(gdt) else math.nan
        ts_ratio = gts / ats if ats and not math.isnan(ats) and not math.isnan(gts) else math.nan
        if ar and gr:
            status = f"A800 failed={ar.get('failed')}; {args.gpu_label} failed={gr.get('failed')}"
        elif ar:
            status = f"missing {args.gpu_label}"
        else:
            status = "missing A800"
        lines.append(
            f"| {phase} | {model} | {seq} | {effbs} | {microbs} | {acc} | {world} | {opt} | {mode} | "
            f"{fmt(adt)} | {fmt(gdt)} | {fmt(dt_ratio)} | {fmt(ats, 1)} | {fmt(gts, 1)} | {fmt(ts_ratio)} | "
            f"{fmt(amem, 0)} | {fmt(gmem, 0)} | {status} |"
        )

    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"wrote {output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
