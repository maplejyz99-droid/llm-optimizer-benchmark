#!/usr/bin/env python3
import argparse
import math
import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[2]
SRC = ROOT / "src"
sys.path.insert(0, str(SRC))

from config.base import parse_args  # noqa: E402
from models.utils import get_model  # noqa: E402
from optim.muon import DistributedMuon, Muon  # noqa: E402


def make_args(model_name, extra=None):
    base = [
        "--model",
        model_name,
        "--n_embd",
        "16",
        "--n_layer",
        "2",
        "--n_head",
        "2",
        "--sequence_length",
        "8",
        "--vocab_size",
        "128",
        "--multiple_of",
        "4",
        "--scale_base_model",
        "8",
        "--scale_emb",
        "10",
        "--scale_depth",
        "1.4",
        "--lr",
        "0.01",
        "--muon_lr_factor",
        "0.02",
        "--warmup_steps",
        "2",
        "--iterations",
        "10",
    ]
    return parse_args(argparse.ArgumentParser(), base + (extra or []), None)


def specs_to_param_groups(model, specs):
    param_dict = dict(model.named_parameters())
    groups = []
    for spec in specs:
        group = {k: v for k, v in spec.items() if k != "params"}
        group["params"] = [param_dict[name] for name in spec["params"]]
        groups.append(group)
    return groups


def assert_close(actual, expected, msg):
    if not math.isclose(actual, expected, rel_tol=1e-12, abs_tol=1e-12):
        raise AssertionError(f"{msg}: got {actual}, expected {expected}")


def check_specs():
    llama_args = make_args("llama")
    llama = get_model(llama_args)
    llama_specs = llama.get_parameter_group_specs(llama_args)
    if any("lr" in group for group in llama_specs):
        raise AssertionError("plain llama unexpectedly has a per-group lr")

    mup_args = make_args("mup_llama")
    mup_llama = get_model(mup_args)
    mup_specs = mup_llama.get_parameter_group_specs(mup_args)
    width_mult = mup_args.n_embd / mup_args.scale_base_model
    assert_close(mup_specs[0]["lr"], mup_args.lr / width_mult, "mup_llama decay lr")
    if "lr" in mup_specs[1]:
        raise AssertionError("mup_llama no_decay group should inherit optimizer lr")
    if mup_specs[1].get("weight_decay") != 0.0:
        raise AssertionError("mup_llama no_decay group should set weight_decay=0.0")
    names = [name for group in mup_specs for name in group["params"]]
    if len(names) != len(set(names)):
        raise AssertionError("mup_llama parameter groups contain duplicate names")
    if set(names) != set(dict(mup_llama.named_parameters()).keys()):
        missing = set(dict(mup_llama.named_parameters()).keys()) - set(names)
        extra = set(names) - set(dict(mup_llama.named_parameters()).keys())
        raise AssertionError(f"mup_llama parameter coverage mismatch missing={missing} extra={extra}")

    mup_gpt_args = make_args("mup_gpt")
    mup_gpt = get_model(mup_gpt_args)
    mup_gpt_specs = mup_gpt.get_parameter_group_specs(mup_gpt_args)
    assert_close(mup_gpt_specs[0]["lr"], mup_gpt_args.lr / width_mult, "mup_gpt decay lr")
    print("parameter group specs: ok")


def check_optimizers():
    args = make_args("mup_llama")
    model = get_model(args)
    groups = specs_to_param_groups(model, model.get_parameter_group_specs(args))
    width_mult = args.n_embd / args.scale_base_model

    adamw = torch.optim.AdamW(groups, lr=args.lr, weight_decay=args.weight_decay)
    assert_close(adamw.param_groups[0]["lr"], args.lr / width_mult, "AdamW decay lr")
    assert_close(adamw.param_groups[1]["lr"], args.lr, "AdamW no_decay lr")
    scheduler = torch.optim.lr_scheduler.OneCycleLR(
        adamw,
        max_lr=[group.get("lr", args.lr) for group in groups],
        total_steps=args.iterations,
        pct_start=args.warmup_steps / args.iterations,
        cycle_momentum=False,
        div_factor=1e2,
        final_div_factor=args.final_div_factor,
    )
    assert_close(adamw.param_groups[0]["max_lr"], args.lr / width_mult, "AdamW max_lr decay")
    assert_close(adamw.param_groups[1]["max_lr"], args.lr, "AdamW max_lr no_decay")
    scheduler.step()

    dmuon_groups = specs_to_param_groups(model, model.get_parameter_group_specs(args))
    dmuon = DistributedMuon(
        dmuon_groups,
        lr=args.lr,
        momentum=args.momentum,
        nesterov=args.nesterov,
        ns_steps=args.muon_ns_steps,
        adamw_betas=(args.beta1, args.beta2),
        adamw_eps=1e-8,
        weight_decay=args.weight_decay,
    )
    assert_close(dmuon.param_groups[0]["lr"], args.lr / width_mult, "D-Muon decay lr")
    assert_close(dmuon.param_groups[1]["lr"], args.lr, "D-Muon no_decay lr")

    muon_lr = args.muon_lr_factor / width_mult
    muon = Muon(
        muon_params=list(model.parameters()),
        lr=muon_lr,
        momentum=args.momentum,
        nesterov=args.nesterov,
        ns_steps=args.muon_ns_steps,
        adamw_params=None,
        adamw_lr=args.lr,
        adamw_betas=(args.beta1, args.beta2),
        adamw_eps=1e-8,
        adamw_wd=args.weight_decay,
    )
    assert_close(muon.param_groups[0]["lr"], muon_lr, "Muon matrix lr")
    backup_lr = muon.param_groups[0]["adamw_lr_ratio"] * muon.param_groups[0]["lr"]
    assert_close(backup_lr, args.lr, "Muon AdamW backup lr")
    print("optimizer lr checks: ok")


if __name__ == "__main__":
    check_specs()
    check_optimizers()
