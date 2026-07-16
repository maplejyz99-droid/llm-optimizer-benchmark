#!/usr/bin/env python3
"""Read-only FineWeb memmap preflight check for GPU comparison runs."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np


def resolve_fineweb_dir(datasets_dir: Path) -> Path:
    candidates = [
        datasets_dir,
        datasets_dir / "fineweb-30B",
        datasets_dir / "fineweb-100BT",
    ]
    seen: set[Path] = set()
    for candidate in candidates:
        candidate = candidate.expanduser().resolve()
        if candidate in seen:
            continue
        seen.add(candidate)
        if (candidate / "train.bin").exists() and (candidate / "val.bin").exists():
            return candidate
    checked = ", ".join(str(path) for path in candidates)
    raise FileNotFoundError(f"train.bin/val.bin not found under: {checked}")


def describe_bin(path: Path) -> dict:
    if not path.exists():
        raise FileNotFoundError(path)
    size_bytes = path.stat().st_size
    if size_bytes <= 0:
        raise ValueError(f"{path} is empty")
    itemsize = np.dtype(np.uint16).itemsize
    if size_bytes % itemsize != 0:
        raise ValueError(f"{path} size is not divisible by uint16 itemsize")
    token_count = size_bytes // itemsize
    data = np.memmap(path, dtype=np.uint16, mode="r")
    probe_index = min(1024, token_count - 1)
    result = {
        "path": str(path),
        "size_bytes": size_bytes,
        "tokens": token_count,
        "first_token": int(data[0]) if token_count else None,
        "probe_index": probe_index,
        "probe_token": int(data[probe_index]) if token_count else None,
    }
    del data
    return result


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--datasets-dir",
        default="/root/autodl-tmp/llmopt/datasets/fineweb-30B",
        help="FineWeb directory itself, or a parent containing fineweb-30B/ or fineweb-100BT/.",
    )
    parser.add_argument("--json", action="store_true", help="Print machine-readable JSON.")
    args = parser.parse_args()

    datasets_dir = Path(args.datasets_dir)
    try:
        fineweb_dir = resolve_fineweb_dir(datasets_dir)
        result = {
            "datasets_dir_arg": str(datasets_dir.expanduser()),
            "fineweb_dir": str(fineweb_dir),
            "train": describe_bin(fineweb_dir / "train.bin"),
            "val": describe_bin(fineweb_dir / "val.bin"),
        }
        for optional in ("meta.json", "build_state.json"):
            path = fineweb_dir / optional
            if path.exists():
                try:
                    result[optional] = json.loads(path.read_text(encoding="utf-8"))
                except json.JSONDecodeError:
                    result[optional] = path.read_text(encoding="utf-8", errors="replace")[:2000]
    except Exception as exc:
        print(f"ERROR: FineWeb preflight failed: {exc}", file=sys.stderr)
        print(
            "Expected one of: DATASETS_DIR/train.bin + val.bin, "
            "DATASETS_DIR/fineweb-30B/train.bin + val.bin, or "
            "DATASETS_DIR/fineweb-100BT/train.bin + val.bin.",
            file=sys.stderr,
        )
        return 1

    if args.json:
        print(json.dumps(result, indent=2, sort_keys=True))
        return 0

    print("FineWeb preflight OK")
    print(f"datasets_dir_arg: {result['datasets_dir_arg']}")
    print(f"fineweb_dir: {result['fineweb_dir']}")
    for split in ("train", "val"):
        info = result[split]
        print(
            f"{split}: tokens={info['tokens']:,} "
            f"size={info['size_bytes']:,} bytes path={info['path']}"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
