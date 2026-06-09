#!/usr/bin/env python3

import csv
import statistics
from collections import defaultdict
from pathlib import Path


RUNS = {
    "muon": Path("logs/muon_124m_1gpu_bs32_acc1_500step_smi.csv"),
    "newton-muon": Path("logs/newton_muon_124m_1gpu_bs32_acc1_500step_smi.csv"),
}


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


def main():
    rows = {}
    for name, path in RUNS.items():
        if not path.exists():
            raise SystemExit(f"Missing {path}. Run the {name} probe first.")
        rows[name] = load_series(path)

    muon = rows["muon"]
    newton = rows["newton-muon"]

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


if __name__ == "__main__":
    main()
