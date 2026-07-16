#!/usr/bin/env python3
"""No-card CLI and optimizer assembly check for SoftEq K=2000 Muon."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from types import SimpleNamespace

import torch

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT / "src"))

from main import get_args  # noqa: E402
from optim.experimental.softeq_muon import SoftEqK2000Muon  # noqa: E402
from optim.muon import CombinedScheduler  # noqa: E402


def parse_main_args() -> object:
    old_argv = sys.argv[:]
    try:
        sys.argv = [
            "main.py",
            "--config_format", "base",
            "--opt", "softeq-k2000-muon",
            "--model", "llama",
            "--dataset", "fineweb",
            "--datasets_dir", "/root/autodl-tmp/llmopt/datasets/fineweb-30B",
            "--device", "cpu",
            "--iterations", "10",
            "--warmup_steps", "1",
            "--batch_size", "1",
            "--acc_steps", "1",
        ]
        args, _ = get_args()
    finally:
        sys.argv = old_argv
    return args


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--json", action="store_true")
    args = parser.parse_args()

    parsed = parse_main_args()
    if parsed.opt != "softeq-k2000-muon":
        raise AssertionError(f"Unexpected opt: {parsed.opt}")

    matrix = torch.nn.Parameter(torch.randn(4, 4))
    vector = torch.nn.Parameter(torch.randn(4))
    opt = SoftEqK2000Muon(
        muon_params=[matrix, vector],
        lr=1e-2,
        momentum=0.95,
        weight_decay=0.1,
        adamw_lr=1e-3,
        adamw_betas=(0.9, 0.95),
        adamw_wd=0.1,
    )
    matrix.grad = torch.ones_like(matrix)
    vector.grad = torch.ones_like(vector)
    opt.step()

    cfg = SimpleNamespace(
        scheduler="wsd",
        lr=1e-2,
        iterations=10,
        warmup_steps=1,
        final_div_factor=1,
        cos_inf_steps=0,
        wsd_fract_decay=0.1,
        wsd_final_lr_scale=0.0,
        decay_type="linear",
    )
    scheduler = CombinedScheduler(opt, cfg)

    result = {
        "cli_opt": parsed.opt,
        "optimizer": type(opt).__name__,
        "global_step": opt.global_step,
        "matrix_use_muon": bool(opt.state[matrix]["use_muon"]),
        "vector_use_muon": bool(opt.state[vector]["use_muon"]),
        "scheduler": type(scheduler).__name__,
    }
    if args.json:
        print(json.dumps(result, indent=2, sort_keys=True))
    else:
        print("MuonEq CLI/optimizer/scheduler check OK")
        print(result)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
