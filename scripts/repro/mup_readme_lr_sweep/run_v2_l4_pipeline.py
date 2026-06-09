#!/usr/bin/env python3
"""Run the v2_l4 sweep, LR selection, long runs, and plots.

This is intentionally an orchestration wrapper around the existing repro
scripts. It keeps the experiment package external to src/ and makes long
remote runs resumable via --skip-existing.
"""

from __future__ import annotations

import argparse
import csv
import math
import subprocess
import sys
from collections import defaultdict
from pathlib import Path


SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parents[2]
RESULTS_CSV = REPO_ROOT / "logs/mup_readme_lr_sweep/results_v2_l4.csv"
PROBE_SUMMARY_CSV = REPO_ROOT / "logs/mup_readme_lr_sweep/probe_summary_v2_l4.csv"
SWEEP_WIDTHS = "128,256,512,768,1024,1536,2048"
LONG_MAIN_WIDTHS = "128,512,768,1024,2048"
LONG_EXTRA_WIDTHS = "256,1536"
ALL_PARSE_PHASES = ",".join(
    [
        "probe_v2_l4",
        "sweep_v2_l4",
        "long_v2_l4_global",
        "long_v2_l4_anchor128",
        "long_v2_l4_anchor768",
        "long_v2_l4_extra",
    ]
)


def run(cmd: list[str]) -> None:
    print("$ " + " ".join(cmd), flush=True)
    subprocess.run(cmd, cwd=REPO_ROOT, check=True)


def run_sweep_phase(
    phase: str,
    *,
    widths: str | None = None,
    sp_lr: float | None = None,
    mup_lr: float | None = None,
    dry_run: bool = False,
) -> None:
    cmd = [
        sys.executable,
        str(SCRIPT_DIR / "run_sweep.py"),
        "--phase",
        phase,
        "--skip-existing",
    ]
    if widths is not None:
        cmd += ["--widths", widths]
    if sp_lr is not None:
        cmd += ["--sp-lr", f"{sp_lr:g}"]
    if mup_lr is not None:
        cmd += ["--mup-lr", f"{mup_lr:g}"]
    if dry_run:
        cmd += ["--dry-run"]
    run(cmd)


def parse_results() -> None:
    run(
        [
            sys.executable,
            str(SCRIPT_DIR / "parse_logs.py"),
            "--output",
            str(RESULTS_CSV),
            "--include-phases",
            ALL_PARSE_PHASES,
            "--probe-summary",
            str(PROBE_SUMMARY_CSV),
        ]
    )


def plot_lr_sweep() -> None:
    run(
        [
            sys.executable,
            str(SCRIPT_DIR / "plot_results.py"),
            "--csv",
            str(RESULTS_CSV),
            "--phases",
            "sweep_v2_l4",
            "--lr-output-name",
            "sp_vs_mup_remote_lr_sweep_v2_l4.png",
        ]
    )


def plot_long(phase: str, output_name: str) -> None:
    run(
        [
            sys.executable,
            str(SCRIPT_DIR / "plot_results.py"),
            "--csv",
            str(RESULTS_CSV),
            "--phases",
            "sweep_v2_l4",
            "--plot-long",
            "--long-phases",
            phase,
            "--lr-output-name",
            "sp_vs_mup_remote_lr_sweep_v2_l4.png",
            "--long-output-name",
            output_name,
        ]
    )


def read_sweep_rows() -> list[dict]:
    rows: list[dict] = []
    with RESULTS_CSV.open("r", encoding="utf-8", newline="") as handle:
        for row in csv.DictReader(handle):
            if row["phase"] != "sweep_v2_l4" or row["failed"] != "False":
                continue
            row["n_embd"] = int(row["n_embd"])
            row["lr"] = float(row["lr"])
            row["loss_tail_mean"] = float(row["loss_tail_mean"])
            if not math.isnan(row["loss_tail_mean"]):
                rows.append(row)
    return rows


def best_by_alias(rows: list[dict], anchor_width: int | None = None) -> dict[str, float]:
    scores: dict[str, dict[float, list[float]]] = defaultdict(lambda: defaultdict(list))
    for row in rows:
        if anchor_width is not None and row["n_embd"] != anchor_width:
            continue
        scores[row["alias"]][row["lr"]].append(row["loss_tail_mean"])

    picks: dict[str, float] = {}
    for alias, by_lr in scores.items():
        means = {lr: sum(values) / len(values) for lr, values in by_lr.items() if values}
        if means:
            picks[alias] = min(means, key=means.get)
    return picks


def require_pair(name: str, picks: dict[str, float]) -> tuple[float, float]:
    if "sp" not in picks or "mup" not in picks:
        raise RuntimeError(f"{name} did not produce both sp and mup LR picks: {picks}")
    print(f"{name}: --mup-lr {picks['mup']:g} --sp-lr {picks['sp']:g}", flush=True)
    return picks["sp"], picks["mup"]


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--skip-probe", action="store_true")
    parser.add_argument("--skip-sweep", action="store_true")
    parser.add_argument("--skip-long", action="store_true")
    parser.add_argument("--run-extra", action="store_true")
    args = parser.parse_args()

    if not args.skip_probe:
        run_sweep_phase("probe_v2_l4", dry_run=args.dry_run)
    if not args.skip_sweep:
        run_sweep_phase("sweep_v2_l4", widths=SWEEP_WIDTHS, dry_run=args.dry_run)
    if args.dry_run:
        return 0

    parse_results()
    plot_lr_sweep()
    run([sys.executable, str(SCRIPT_DIR / "select_lrs.py"), "--csv", str(RESULTS_CSV), "--phase", "sweep_v2_l4"])

    if not args.skip_long:
        rows = read_sweep_rows()
        global_sp_lr, global_mup_lr = require_pair("global", best_by_alias(rows))
        anchor128_sp_lr, anchor128_mup_lr = require_pair("anchor128", best_by_alias(rows, anchor_width=128))
        anchor768_sp_lr, anchor768_mup_lr = require_pair("anchor768", best_by_alias(rows, anchor_width=768))

        run_sweep_phase("long_v2_l4_global", widths=LONG_MAIN_WIDTHS, sp_lr=global_sp_lr, mup_lr=global_mup_lr)
        run_sweep_phase("long_v2_l4_anchor128", widths=LONG_MAIN_WIDTHS, sp_lr=anchor128_sp_lr, mup_lr=anchor128_mup_lr)
        run_sweep_phase("long_v2_l4_anchor768", widths=LONG_MAIN_WIDTHS, sp_lr=anchor768_sp_lr, mup_lr=anchor768_mup_lr)
        if args.run_extra:
            run_sweep_phase("long_v2_l4_extra", widths=LONG_EXTRA_WIDTHS, sp_lr=global_sp_lr, mup_lr=global_mup_lr)

        parse_results()
        plot_long("long_v2_l4_global", "sp_vs_mup_remote_longrun_global_v2_l4.png")
        plot_long("long_v2_l4_anchor128", "sp_vs_mup_remote_longrun_anchor128_v2_l4.png")
        plot_long("long_v2_l4_anchor768", "sp_vs_mup_remote_longrun_anchor768_v2_l4.png")
        if args.run_extra:
            plot_long("long_v2_l4_extra", "sp_vs_mup_remote_longrun_extra_global_v2_l4.png")

    print("v2_l4 pipeline complete", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
