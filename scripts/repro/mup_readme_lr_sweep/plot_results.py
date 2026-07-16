#!/usr/bin/env python3
"""Plot the project-local analogue of microsoft/mup's README LR figure."""

from __future__ import annotations

import argparse
import csv
import math
import re
from collections import defaultdict
from pathlib import Path

import matplotlib.pyplot as plt


SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parents[2]
TRAIN_RE = re.compile(r"Train: Iter=(?P<iter>\d+).*?train_loss=(?P<loss>[-+0-9.eE]+)")


def read_rows(path: Path, phases: set[str]) -> list[dict]:
    rows: list[dict] = []
    with path.open("r", encoding="utf-8", newline="") as handle:
        for row in csv.DictReader(handle):
            if row["phase"] in phases and row["failed"] == "False":
                row["n_embd"] = int(row["n_embd"])
                row["lr"] = float(row["lr"])
                row["loss_tail_mean"] = float(row["loss_tail_mean"])
                rows.append(row)
    return rows


def plot_lr_sweep(rows: list[dict], output: Path) -> None:
    aliases = [("sp", "SP / llama"), ("mup", "muP / mup_llama")]
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.8), sharey=True)
    for axis, (alias, title) in zip(axes, aliases):
        axis.set_title(title)
        axis.set_xscale("log")
        axis.set_xlabel("nominal --lr")
        axis.grid(True, alpha=0.25)
        grouped: dict[int, list[dict]] = defaultdict(list)
        for row in rows:
            if row["alias"] == alias:
                grouped[row["n_embd"]].append(row)
        for width, width_rows in sorted(grouped.items()):
            width_rows = sorted(width_rows, key=lambda row: row["lr"])
            axis.plot(
                [row["lr"] for row in width_rows],
                [row["loss_tail_mean"] for row in width_rows],
                marker="o",
                label=f"width {width}",
            )
        axis.legend(fontsize=8)
    axes[0].set_ylabel("tail-mean training loss")
    fig.suptitle("Remote analogue of microsoft/mup README LR sweep")
    fig.tight_layout()
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output, dpi=180)
    print(f"wrote {output}")


def parse_train_series(log_path: Path) -> list[tuple[int, float]]:
    points: list[tuple[int, float]] = []
    for line in log_path.read_text(encoding="utf-8", errors="ignore").splitlines():
        match = TRAIN_RE.search(line)
        if match:
            points.append((int(match.group("iter")), float(match.group("loss"))))
    return points


def plot_long_runs(csv_path: Path, output: Path, phases: set[str]) -> None:
    rows = read_rows(csv_path, phases)
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.8), sharey=True)
    for axis, alias, title in zip(axes, ["sp", "mup"], ["SP / llama", "muP / mup_llama"]):
        axis.set_title(title)
        axis.set_xlabel("iteration")
        axis.grid(True, alpha=0.25)
        for row in sorted([row for row in rows if row["alias"] == alias], key=lambda r: r["n_embd"]):
            series = parse_train_series(REPO_ROOT / row["log_path"])
            if not series:
                continue
            axis.plot(
                [point[0] for point in series],
                [point[1] for point in series],
                label=f"width {row['n_embd']} lr={row['lr']:.0e}",
            )
        axis.legend(fontsize=8)
    axes[0].set_ylabel("training loss")
    fig.suptitle("Selected 5000-step runs")
    fig.tight_layout()
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output, dpi=180)
    print(f"wrote {output}")


def recommend_long_lrs(rows: list[dict]) -> dict[str, float]:
    scores: dict[str, dict[float, list[float]]] = defaultdict(lambda: defaultdict(list))
    for row in rows:
        if not math.isnan(row["loss_tail_mean"]):
            scores[row["alias"]][row["lr"]].append(row["loss_tail_mean"])
    picks: dict[str, float] = {}
    for alias, lr_values in scores.items():
        means = {lr: sum(values) / len(values) for lr, values in lr_values.items() if values}
        if means:
            picks[alias] = min(means, key=means.get)
    return picks


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--csv", default=str(REPO_ROOT / "logs/mup_readme_lr_sweep/results.csv"))
    parser.add_argument("--output-dir", default=str(REPO_ROOT / "logs/mup_readme_lr_sweep/figures"))
    parser.add_argument("--phases", default="coarse,fine")
    parser.add_argument("--plot-long", action="store_true")
    parser.add_argument("--lr-output-name", default="sp_vs_mup_remote_lr_sweep.png")
    parser.add_argument("--long-output-name", default="sp_vs_mup_remote_longrun.png")
    parser.add_argument("--long-phases", default="long")
    args = parser.parse_args()

    csv_path = Path(args.csv)
    output_dir = Path(args.output_dir)
    phases = set(args.phases.split(","))
    rows = read_rows(csv_path, phases)
    plot_lr_sweep(rows, output_dir / args.lr_output_name)
    picks = recommend_long_lrs(rows)
    if picks:
        print("recommended long-run LRs:", ", ".join(f"{k}={v:g}" for k, v in sorted(picks.items())))
    if args.plot_long:
        plot_long_runs(csv_path, output_dir / args.long_output_name, set(args.long_phases.split(",")))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
