#!/usr/bin/env python3
"""Validate the versioned FineWeb-30B artifact without reading all tokens."""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path


TRAIN_TOKENS = 30_000_000_000
VAL_TOKENS = 100_000_000
ITEM_SIZE = 2


def default_datasets_dir() -> Path:
    return Path(os.environ.get("LLMOPT_DATASETS_DIR") or "./src/data/datasets/")


def resolve_fineweb_30b_dir(datasets_dir: Path) -> Path:
    candidates = [datasets_dir, datasets_dir / "fineweb-30B"]
    seen: set[Path] = set()
    for candidate in candidates:
        candidate = candidate.expanduser().resolve()
        if candidate in seen:
            continue
        seen.add(candidate)
        if all((candidate / name).is_file() for name in ("train.bin", "val.bin", "meta.json")):
            return candidate
    checked = ", ".join(str(path.expanduser()) for path in candidates)
    raise FileNotFoundError(
        "FineWeb-30B train.bin, val.bin, and meta.json were not found. "
        f"Checked: {checked}"
    )


def validate_fineweb_30b(datasets_dir: Path) -> dict:
    fineweb_dir = resolve_fineweb_30b_dir(datasets_dir)
    train_path = fineweb_dir / "train.bin"
    val_path = fineweb_dir / "val.bin"
    meta_path = fineweb_dir / "meta.json"
    meta = json.loads(meta_path.read_text(encoding="utf-8"))

    expected_meta = {
        "complete": True,
        "dtype": "uint16",
        "tokenizer": "gpt2",
        "train_tokens_target": TRAIN_TOKENS,
        "train_tokens_written": TRAIN_TOKENS,
        "val_tokens_target": VAL_TOKENS,
        "val_tokens_written": VAL_TOKENS,
    }
    mismatches = {
        key: {"expected": expected, "actual": meta.get(key)}
        for key, expected in expected_meta.items()
        if meta.get(key) != expected
    }
    expected_sizes = {
        "train.bin": TRAIN_TOKENS * ITEM_SIZE,
        "val.bin": VAL_TOKENS * ITEM_SIZE,
    }
    actual_sizes = {
        "train.bin": train_path.stat().st_size,
        "val.bin": val_path.stat().st_size,
    }
    for name, expected in expected_sizes.items():
        if actual_sizes[name] != expected:
            mismatches[f"{name}.size_bytes"] = {
                "expected": expected,
                "actual": actual_sizes[name],
            }
    if mismatches:
        raise ValueError(
            "FineWeb-30B artifact identity mismatch: "
            + json.dumps(mismatches, sort_keys=True)
        )

    return {
        "profile": "fineweb-30B-gpt2-uint16-v1",
        "fineweb_dir": str(fineweb_dir),
        "train": {"path": str(train_path), "tokens": TRAIN_TOKENS},
        "val": {"path": str(val_path), "tokens": VAL_TOKENS},
        "meta": meta,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--datasets-dir",
        type=Path,
        default=default_datasets_dir(),
        help=(
            "FineWeb-30B directory or its parent. Defaults to "
            "LLMOPT_DATASETS_DIR, then ./src/data/datasets/."
        ),
    )
    parser.add_argument("--json", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    try:
        result = validate_fineweb_30b(args.datasets_dir)
    except (FileNotFoundError, OSError, ValueError, json.JSONDecodeError) as exc:
        print(f"ERROR: FineWeb-30B preflight failed: {exc}", file=sys.stderr)
        return 1

    if args.json:
        print(json.dumps(result, indent=2, sort_keys=True))
    else:
        print("FineWeb-30B preflight OK")
        print(f"profile: {result['profile']}")
        print(f"fineweb_dir: {result['fineweb_dir']}")
        print(f"train_tokens: {result['train']['tokens']:,}")
        print(f"val_tokens: {result['val']['tokens']:,}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
