#!/usr/bin/env python3
"""Build a uint16 GPT-2-tokenized FineWeb binary dataset from cached parquet shards."""

from __future__ import annotations

import argparse
import json
import os
import time
from pathlib import Path
from typing import Iterable

import numpy as np
import pyarrow.parquet as pq
import tiktoken


SOURCE_PREFIX = "hf://datasets/HuggingFaceFW/fineweb@"
SOURCE_MARKER = "/sample/100BT/"
TOKENIZER_NAME = "gpt2"
EOT_TOKEN = 50256


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--downloads-dir", required=True, type=Path)
    parser.add_argument("--out-dir", required=True, type=Path)
    parser.add_argument("--train-tokens", required=True, type=int)
    parser.add_argument("--val-tokens", required=True, type=int)
    parser.add_argument("--dtype", default="uint16", choices=["uint16"])
    parser.add_argument("--tokenizer-threads", type=int, default=1)
    parser.add_argument("--log-interval-seconds", type=float, default=30.0)
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--resume", action="store_true")
    return parser.parse_args()


def atomic_write_json(path: Path, payload: dict) -> None:
    tmp_path = path.with_suffix(path.suffix + ".tmp")
    tmp_path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    tmp_path.replace(path)


def discover_parquet_files(downloads_dir: Path) -> list[tuple[str, Path]]:
    files: list[tuple[str, Path]] = []
    for meta_path in sorted(downloads_dir.glob("*.json")):
        try:
            meta = json.loads(meta_path.read_text(encoding="utf-8"))
        except json.JSONDecodeError:
            continue
        url = str(meta.get("url", ""))
        data_path = meta_path.with_suffix("")
        if not data_path.exists():
            continue
        if not (url.startswith(SOURCE_PREFIX) and SOURCE_MARKER in url and url.endswith(".parquet")):
            continue
        files.append((url, data_path))
    files.sort(key=lambda item: item[0])
    return files


def flatten_with_eot(encoded: Iterable[list[int]], total_tokens: int) -> np.ndarray:
    arr = np.empty(total_tokens, dtype=np.uint16)
    offset = 0
    for ids in encoded:
        n = len(ids)
        if n:
            arr[offset : offset + n] = ids
            offset += n
        arr[offset] = EOT_TOKEN
        offset += 1
    if offset != total_tokens:
        raise RuntimeError(f"token flatten mismatch: wrote {offset}, expected {total_tokens}")
    return arr


def encode_texts(enc: tiktoken.Encoding, texts: list[str], threads: int) -> np.ndarray:
    if hasattr(enc, "encode_ordinary_batch"):
        encoded = enc.encode_ordinary_batch(texts, num_threads=threads)
    else:
        encoded = [enc.encode_ordinary(text) for text in texts]
    total = sum(len(ids) + 1 for ids in encoded)
    return flatten_with_eot(encoded, total)


def open_outputs(args: argparse.Namespace) -> tuple[np.memmap, np.memmap, dict]:
    args.out_dir.mkdir(parents=True, exist_ok=True)
    state_path = args.out_dir / "build_state.json"
    train_path = args.out_dir / "train.bin"
    val_path = args.out_dir / "val.bin"
    existing = [p for p in [train_path, val_path, state_path] if p.exists()]

    if args.resume and state_path.exists():
        state = json.loads(state_path.read_text(encoding="utf-8"))
        mode = "r+"
    else:
        if existing and not args.overwrite:
            raise FileExistsError(
                f"{args.out_dir} already has output files; use --overwrite or --resume"
            )
        state = {
            "file_index": 0,
            "row_group": 0,
            "train_written": 0,
            "val_written": 0,
            "complete": False,
        }
        mode = "w+"

    train = np.memmap(train_path, dtype=np.uint16, mode=mode, shape=(args.train_tokens,))
    val = np.memmap(val_path, dtype=np.uint16, mode=mode, shape=(args.val_tokens,))
    return train, val, state


def write_tokens(
    tokens: np.ndarray,
    train: np.memmap,
    val: np.memmap,
    state: dict,
    train_target: int,
    val_target: int,
) -> None:
    offset = 0
    total = int(tokens.shape[0])
    if state["val_written"] < val_target:
        n = min(total, val_target - state["val_written"])
        val[state["val_written"] : state["val_written"] + n] = tokens[:n]
        state["val_written"] += n
        offset += n
    if offset < total and state["train_written"] < train_target:
        n = min(total - offset, train_target - state["train_written"])
        train[state["train_written"] : state["train_written"] + n] = tokens[offset : offset + n]
        state["train_written"] += n


def log_progress(start: float, state: dict, files_done: int, num_files: int, force: bool = False) -> float:
    now = time.time()
    elapsed = max(now - start, 1e-6)
    train = state["train_written"]
    val = state["val_written"]
    rate = (train + val) / elapsed
    print(
        f"[progress] files={files_done}/{num_files} "
        f"train={train:,} val={val:,} rate={rate:,.0f} tok/s elapsed={elapsed/3600:.2f}h",
        flush=True,
    )
    return now


def main() -> int:
    args = parse_args()
    start = time.time()

    files = discover_parquet_files(args.downloads_dir)
    if not files:
        raise FileNotFoundError(f"no FineWeb sample/100BT parquet cache files in {args.downloads_dir}")

    enc = tiktoken.get_encoding(TOKENIZER_NAME)
    train, val, state = open_outputs(args)
    state_path = args.out_dir / "build_state.json"
    meta_path = args.out_dir / "meta.json"

    print(
        f"[start] files={len(files)} out={args.out_dir} "
        f"train_target={args.train_tokens:,} val_target={args.val_tokens:,} dtype=uint16",
        flush=True,
    )
    last_log = log_progress(start, state, state["file_index"], len(files), force=True)

    for file_index, (url, parquet_path) in enumerate(files):
        if file_index < state["file_index"]:
            continue
        pf = pq.ParquetFile(parquet_path)
        row_group_start = state["row_group"] if file_index == state["file_index"] else 0
        for row_group in range(row_group_start, pf.num_row_groups):
            if state["train_written"] >= args.train_tokens and state["val_written"] >= args.val_tokens:
                state["complete"] = True
                break
            table = pf.read_row_group(row_group, columns=["text"])
            texts = table.column("text").to_pylist()
            tokens = encode_texts(enc, texts, args.tokenizer_threads)
            write_tokens(tokens, train, val, state, args.train_tokens, args.val_tokens)
            state.update(
                {
                    "file_index": file_index,
                    "row_group": row_group + 1,
                    "current_url": url,
                    "current_file": str(parquet_path),
                    "updated_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
                }
            )
            if time.time() - last_log >= args.log_interval_seconds:
                train.flush()
                val.flush()
                atomic_write_json(state_path, state)
                last_log = log_progress(start, state, file_index, len(files))
        if state.get("complete"):
            break
        state["file_index"] = file_index + 1
        state["row_group"] = 0
        atomic_write_json(state_path, state)

    train.flush()
    val.flush()
    state["complete"] = state["train_written"] >= args.train_tokens and state["val_written"] >= args.val_tokens
    state["finished_at"] = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
    atomic_write_json(state_path, state)

    meta = {
        "source": "HuggingFaceFW/fineweb/sample/100BT cached parquet",
        "downloads_dir": str(args.downloads_dir),
        "tokenizer": TOKENIZER_NAME,
        "eot_token": EOT_TOKEN,
        "dtype": "uint16",
        "train_tokens_target": args.train_tokens,
        "val_tokens_target": args.val_tokens,
        "train_tokens_written": state["train_written"],
        "val_tokens_written": state["val_written"],
        "complete": state["complete"],
        "num_source_files": len(files),
        "created_at": state["finished_at"],
        "format": "raw contiguous np.memmap token ids, no header",
    }
    atomic_write_json(meta_path, meta)
    log_progress(start, state, state["file_index"], len(files), force=True)
    print(f"[done] complete={state['complete']} meta={meta_path}", flush=True)
    return 0 if state["complete"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
