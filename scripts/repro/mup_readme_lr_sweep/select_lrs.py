#!/usr/bin/env python3
"""Select global and anchor LRs from a parsed sweep CSV."""

from __future__ import annotations

import argparse
import csv
import math
from collections import defaultdict
from pathlib import Path


SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parents[2]


def read_rows(path: Path, phase: str) -> list[dict]:
    rows: list[dict] = []
    with path.open("r", encoding="utf-8", newline="") as handle:
        for row in csv.DictReader(handle):
            if row["phase"] != phase or row["failed"] != "False":
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


def format_pick(name: str, picks: dict[str, float]) -> str:
    return f"{name}: " + " ".join(f"--{alias}-lr {lr:g}" for alias, lr in sorted(picks.items()))


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--csv", default=str(REPO_ROOT / "logs/mup_readme_lr_sweep/results_v2_l4.csv"))
    parser.add_argument("--phase", default="sweep_v2_l4")
    parser.add_argument("--anchors", default="128,768")
    args = parser.parse_args()

    rows = read_rows(Path(args.csv), args.phase)
    print(format_pick("global", best_by_alias(rows)))
    for value in args.anchors.split(","):
        anchor = int(value)
        print(format_pick(f"anchor{anchor}", best_by_alias(rows, anchor_width=anchor)))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
