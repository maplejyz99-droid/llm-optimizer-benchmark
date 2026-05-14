#!/usr/bin/env python3

import csv
import re
import statistics
from collections import defaultdict
from pathlib import Path


RUNS = {
    "muon": {
        "smi": Path("logs/muon_124m_1gpu_bs32_acc1_500step_smi.csv"),
        "train": Path("logs/muon_124m_1gpu_bs32_acc1_500step.log"),
    },
    "newton-muon": {
        "smi": Path("logs/newton_muon_124m_1gpu_bs32_acc1_500step_smi.csv"),
        "train": Path("logs/newton_muon_124m_1gpu_bs32_acc1_500step.log"),
    },
}

MEM_RE = re.compile(
    r"^\[mem\]\[iter=(?P<iter>\d+)\]\[(?P<tag>[^\]]+)\] "
    r"curr_alloc_GiB=(?P<curr_alloc>[0-9.]+) "
    r"curr_resv_GiB=(?P<curr_resv>[0-9.]+) "
    r"max_alloc_GiB=(?P<max_alloc>[0-9.]+) "
    r"max_resv_GiB=(?P<max_resv>[0-9.]+)"
)


def load_series(path):
    per_timestamp = defaultdict(int)
    process_names = set()

    with path.open(newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            timestamp = row["timestamp"]
            memory = int(row["memory_used_mib"] or 0)
            per_timestamp[timestamp] += memory
            if row.get("process_name"):
                process_names.add(row["process_name"])

    series = list(per_timestamp.values())
    nonzero = [value for value in series if value > 0]
    steady_source = nonzero if nonzero else series
    return {
        "samples": len(series),
        "peak_mib": max(series) if series else 0,
        "steady_median_mib": int(statistics.median(steady_source)) if steady_source else 0,
        "process_names": ", ".join(sorted(process_names)) if process_names else "-",
    }


def load_torch_memory(path):
    by_tag = defaultdict(list)

    with path.open() as f:
        for line in f:
            match = MEM_RE.match(line.strip())
            if not match:
                continue
            item = {
                "iter": int(match.group("iter")),
                "curr_alloc": float(match.group("curr_alloc")),
                "curr_resv": float(match.group("curr_resv")),
                "max_alloc": float(match.group("max_alloc")),
                "max_resv": float(match.group("max_resv")),
            }
            by_tag[match.group("tag")].append(item)

    summary = {}
    for tag, rows in by_tag.items():
        summary[tag] = {
            "samples": len(rows),
            "peak_max_alloc": max(row["max_alloc"] for row in rows),
            "peak_max_resv": max(row["max_resv"] for row in rows),
            "median_curr_alloc": statistics.median(row["curr_alloc"] for row in rows),
            "median_curr_resv": statistics.median(row["curr_resv"] for row in rows),
        }
    return summary


def main():
    rows = {}
    torch_rows = {}
    for name, paths in RUNS.items():
        path = paths["smi"]
        if not path.exists():
            raise SystemExit(f"Missing {path}. Run the {name} probe first.")
        rows[name] = load_series(path)
        train_path = paths["train"]
        torch_rows[name] = load_torch_memory(train_path) if train_path.exists() else {}

    muon = rows["muon"]
    newton = rows["newton-muon"]

    print("## nvidia-smi process memory")
    print()
    print("| run | peak MiB | steady median MiB | samples | process names |")
    print("|---|---:|---:|---:|---|")
    for name in ("muon", "newton-muon"):
        row = rows[name]
        print(
            f"| {name} | {row['peak_mib']} | {row['steady_median_mib']} | "
            f"{row['samples']} | {row['process_names']} |"
        )

    print()
    print("| delta | peak MiB | steady median MiB |")
    print("|---|---:|---:|")
    print(
        "| newton-muon - muon | "
        f"{newton['peak_mib'] - muon['peak_mib']} | "
        f"{newton['steady_median_mib'] - muon['steady_median_mib']} |"
    )

    if newton["steady_median_mib"] > 0:
        ratio = newton["peak_mib"] / newton["steady_median_mib"]
        if ratio >= 1.10:
            print()
            print(
                "Newton-Muon peak is at least 10% above its steady median; "
                "inspect refresh-step windows around iterations 32, 64, 96, ... 480."
            )

    if torch_rows["muon"] and torch_rows["newton-muon"]:
        print()
        print("## torch cuda memory by training phase")
        print()
        print(
            "| tag | muon max alloc GiB | newton max alloc GiB | delta alloc GiB | "
            "muon max reserved GiB | newton max reserved GiB | delta reserved GiB |"
        )
        print("|---|---:|---:|---:|---:|---:|---:|")
        common_tags = sorted(set(torch_rows["muon"]) & set(torch_rows["newton-muon"]))
        for tag in common_tags:
            mu = torch_rows["muon"][tag]
            nm = torch_rows["newton-muon"][tag]
            print(
                f"| {tag} | "
                f"{mu['peak_max_alloc']:.3f} | {nm['peak_max_alloc']:.3f} | "
                f"{nm['peak_max_alloc'] - mu['peak_max_alloc']:.3f} | "
                f"{mu['peak_max_resv']:.3f} | {nm['peak_max_resv']:.3f} | "
                f"{nm['peak_max_resv'] - mu['peak_max_resv']:.3f} |"
            )


if __name__ == "__main__":
    main()
