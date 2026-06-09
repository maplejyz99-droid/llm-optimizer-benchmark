#!/usr/bin/env python3
import argparse
import math
import sys
from collections import defaultdict
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[2]
SRC = ROOT / "src"
sys.path.insert(0, str(SRC))

from config.base import parse_args  # noqa: E402
from models.utils import get_model  # noqa: E402


WIDTHS = (128, 256, 512)
KEY_SUFFIXES = (
    "attn.c_attn.weight",
    "attn.c_proj.weight",
    "mlp.w1.weight",
    "mlp.w2.weight",
    "mlp.c_proj.weight",
    "transformer.wte.weight",
    "ln_1.weight",
    "ln_2.weight",
    "ln_f.weight",
)


def rms(tensor):
    return tensor.detach().float().square().mean().sqrt().item()


def make_args(width):
    n_head = width // 64
    argv = [
        "--model",
        "mup_llama",
        "--n_embd",
        str(width),
        "--n_layer",
        "2",
        "--n_head",
        str(n_head),
        "--sequence_length",
        "16",
        "--vocab_size",
        "256",
        "--multiple_of",
        "64",
        "--scale_base_model",
        "256",
        "--scale_emb",
        "10",
        "--scale_depth",
        "1.4",
        "--lr",
        "0.001",
        "--dropout",
        "0.0",
        "--device",
        "cpu",
    ]
    return parse_args(argparse.ArgumentParser(), argv, None)


def group_lrs(model, args):
    mapping = {}
    for group in model.get_parameter_group_specs(args):
        lr = group.get("lr", args.lr)
        for name in group["params"]:
            mapping[name] = lr
    return mapping


def selected_params(model):
    params = {}
    for name, param in model.named_parameters():
        if any(name.endswith(suffix) for suffix in KEY_SUFFIXES):
            params[name] = param
    return params


def check_finite(name, value):
    if not math.isfinite(value):
        raise AssertionError(f"{name} is not finite: {value}")


def summarize_width(width):
    torch.manual_seed(1337)
    args = make_args(width)
    model = get_model(args)
    model.train()
    residual_rms = []
    hooks = []

    def capture_residual(_module, _inputs, output):
        hidden = output[0] if isinstance(output, tuple) else output
        residual_rms.append(rms(hidden))

    for block in model.transformer.h:
        hooks.append(block.register_forward_hook(capture_residual))

    idx = torch.randint(0, args.vocab_size, (2, args.sequence_length), dtype=torch.long)
    targets = torch.randint(0, args.vocab_size, (2, args.sequence_length), dtype=torch.long)
    out = model(idx, targets=targets, get_logits=True, full_logits=True)
    loss = out["loss"]
    loss.backward()
    for hook in hooks:
        hook.remove()

    metrics = {
        "loss": loss.item(),
        "logits_rms": rms(out["logits"]),
        "max_residual_rms": max(residual_rms),
    }
    for metric_name, value in metrics.items():
        check_finite(f"width={width} {metric_name}", value)

    tracked_params = selected_params(model)
    before_step = {name: param.detach().clone() for name, param in tracked_params.items()}
    param_groups = []
    param_dict = dict(model.named_parameters())
    for group in model.get_parameter_group_specs(args):
        concrete_group = {k: v for k, v in group.items() if k != "params"}
        concrete_group["params"] = [param_dict[name] for name in group["params"]]
        param_groups.append(concrete_group)
    opt = torch.optim.AdamW(param_groups, lr=args.lr, weight_decay=args.weight_decay)
    opt.step()

    param_metrics = {}
    for name, param in tracked_params.items():
        if param.grad is None:
            continue
        param_rms = rms(before_step[name])
        grad_rms = rms(param.grad)
        update = param.detach() - before_step[name]
        update_rms = rms(update)
        update_ratio = update_rms / max(param_rms, 1e-12)
        values = {
            "param_rms": param_rms,
            "grad_rms": grad_rms,
            "update_rms": update_rms,
            "update_to_param": update_ratio,
        }
        for metric_name, value in values.items():
            check_finite(f"width={width} {name} {metric_name}", value)
        param_metrics[name] = values

    return metrics, param_metrics


def ratio(values):
    positives = [value for value in values if value > 0.0]
    if len(positives) < 2:
        return 1.0
    return max(positives) / min(positives)


def check_ratios(all_metrics, all_param_metrics):
    failures = []
    warnings = []
    scalar_series = defaultdict(list)
    for metrics in all_metrics.values():
        for key, value in metrics.items():
            scalar_series[key].append(value)
    for key, values in scalar_series.items():
        r = ratio(values)
        print(f"{key}: ratio={r:.3f} values={[round(v, 6) for v in values]}")
        if r > 10.0:
            failures.append((key, r))
        elif r > 3.0:
            warnings.append((key, r))

    param_names = sorted(set().union(*[metrics.keys() for metrics in all_param_metrics.values()]))
    for name in param_names:
        values = [
            all_param_metrics[width][name]["update_to_param"]
            for width in WIDTHS
            if name in all_param_metrics[width]
        ]
        r = ratio(values)
        print(f"{name} update_to_param: ratio={r:.3f} values={[round(v, 8) for v in values]}")
        if r > 10.0:
            failures.append((f"{name} update_to_param", r))
        elif r > 3.0:
            warnings.append((f"{name} update_to_param", r))

    for key, r in warnings:
        print(f"WARNING: {key} changed by {r:.3f}x across widths")
    if failures:
        detail = ", ".join(f"{key}={r:.3f}x" for key, r in failures)
        raise AssertionError(f"coordinate check failed: {detail}")


def main():
    all_metrics = {}
    all_param_metrics = {}
    for width in WIDTHS:
        metrics, param_metrics = summarize_width(width)
        all_metrics[width] = metrics
        all_param_metrics[width] = param_metrics
    check_ratios(all_metrics, all_param_metrics)
    print("coordinate check: ok")


if __name__ == "__main__":
    main()
