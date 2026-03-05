import math
from collections import OrderedDict
from dataclasses import dataclass
from typing import Dict, Iterable, List, Tuple

import torch
import torch.nn.functional as F
from torch.func import functional_call, jvp, vjp


@dataclass
class GNStepMetrics:
    loss: float
    base_loss: float
    accuracy: float
    gradient_norm: float
    param_norm: float


def current_param_dict(model: torch.nn.Module) -> "OrderedDict[str, torch.Tensor]":
    return OrderedDict(
        (name, param) for name, param in model.named_parameters() if param.requires_grad
    )


@torch.no_grad()
def clone_param_dict(model: torch.nn.Module) -> "OrderedDict[str, torch.Tensor]":
    return OrderedDict(
        (name, param.detach().clone())
        for name, param in model.named_parameters()
        if param.requires_grad
    )


def clone_param_dict_from_named_params(
    params: "OrderedDict[str, torch.Tensor]",
) -> "OrderedDict[str, torch.Tensor]":
    return OrderedDict((name, tensor.detach().clone()) for name, tensor in params.items())


def named_buffers_dict(model: torch.nn.Module) -> "OrderedDict[str, torch.Tensor]":
    return OrderedDict(model.named_buffers())


@torch.no_grad()
def apply_param_dict_(
    params: "OrderedDict[str, torch.Tensor]",
    new_values: "OrderedDict[str, torch.Tensor]",
) -> None:
    for name, param in params.items():
        param.copy_(new_values[name])


def sub_param_dict(
    left: "OrderedDict[str, torch.Tensor]",
    right: "OrderedDict[str, torch.Tensor]",
) -> "OrderedDict[str, torch.Tensor]":
    return OrderedDict((name, left[name] - right[name]) for name in left.keys())


def add_scaled_param_dict(
    base: "OrderedDict[str, torch.Tensor]",
    direction: "OrderedDict[str, torch.Tensor]",
    scale: float,
) -> "OrderedDict[str, torch.Tensor]":
    return OrderedDict((name, base[name] + scale * direction[name]) for name in base.keys())


def _cross_entropy_from_logits(logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
    return F.cross_entropy(
        logits.view(-1, logits.size(-1)),
        targets.view(-1),
        ignore_index=-1,
    )


def _accuracy_from_logits(logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
    valid = targets.ne(-1)
    if not torch.any(valid):
        return logits.new_tensor(0.0)
    pred = logits.argmax(dim=-1)
    correct = pred.eq(targets) & valid
    return correct.float().sum() / valid.float().sum()


def _param_norm_from_tensors(tensors: Iterable[torch.Tensor]) -> float:
    total = torch.tensor(0.0, device="cpu")
    for tensor in tensors:
        total = total + tensor.detach().float().pow(2).sum().cpu()
    return torch.sqrt(total).item()


def _mean_squared_delta(delta: Dict[str, torch.Tensor]) -> torch.Tensor:
    total = None
    for tensor in delta.values():
        term = tensor.float().pow(2).mean()
        total = term if total is None else total + term
    if total is None:
        raise ValueError("Delta dictionary is empty.")
    return total


def compute_gn_step(
    model: torch.nn.Module,
    params0: "OrderedDict[str, torch.Tensor]",
    x: torch.Tensor,
    y: torch.Tensor,
    mode: str,
    prox_weight_decay: float,
    moe: bool = False,
) -> Tuple[List[torch.Tensor], GNStepMetrics]:
    """
    Compute one GN-prox or GN-full gradient step.
    """
    if mode not in {"prox", "full"}:
        raise ValueError(f"Unknown GN mode: {mode}")

    params = current_param_dict(model)
    buffers = named_buffers_dict(model)
    delta = sub_param_dict(params, params0)

    def logits_fn(pdict: "OrderedDict[str, torch.Tensor]") -> torch.Tensor:
        out = functional_call(
            model,
            (pdict, buffers),
            args=(x,),
            kwargs={"targets": None, "get_logits": True, "moe": moe},
        )
        return out["logits"]

    logits0, jvp_delta = jvp(logits_fn, (params0,), (delta,))

    if mode == "prox":
        logits_linearized = logits0.detach() + jvp_delta
        base_loss = _cross_entropy_from_logits(logits_linearized, y)
        prox = (
            prox_weight_decay * _mean_squared_delta(delta)
            if prox_weight_decay > 0.0
            else logits_linearized.new_tensor(0.0)
        )
        loss = base_loss + prox
        grads = torch.autograd.grad(loss, tuple(params.values()))
        accuracy = _accuracy_from_logits(logits_linearized.detach(), y)
        metrics_loss = loss.detach().item()
        metrics_base_loss = base_loss.detach().item()
    else:
        def base_loss_on_logits(logits: torch.Tensor) -> torch.Tensor:
            return _cross_entropy_from_logits(logits, y)

        logits0_for_grad = logits0.detach().requires_grad_(True)
        jvp_delta_fixed = jvp_delta.detach()
        base_loss_tensor = base_loss_on_logits(logits0_for_grad)
        g0 = torch.autograd.grad(base_loss_tensor, logits0_for_grad)[0]
        hv = torch.autograd.functional.hvp(
            base_loss_on_logits,
            logits0_for_grad,
            jvp_delta_fixed,
        )[1]
        _, vjp_fn = vjp(logits_fn, params0)
        pullback = vjp_fn((g0 + hv).detach())[0]

        grads = []
        for name in params.keys():
            grad = pullback[name]
            if prox_weight_decay > 0.0:
                grad = grad + (2.0 * prox_weight_decay / delta[name].numel()) * delta[name]
            grads.append(grad)
        grads = tuple(grads)

        base_loss = base_loss_tensor.detach()
        surrogate = base_loss + (g0.detach() * jvp_delta_fixed).sum() + 0.5 * (
            jvp_delta_fixed * hv.detach()
        ).sum()
        accuracy = _accuracy_from_logits((logits0 + jvp_delta).detach(), y)
        metrics_loss = surrogate.item()
        metrics_base_loss = base_loss.item()

    gradient_norm = _param_norm_from_tensors(grads)
    param_norm = _param_norm_from_tensors(params.values())
    metrics = GNStepMetrics(
        loss=metrics_loss,
        base_loss=metrics_base_loss,
        accuracy=accuracy.detach().item(),
        gradient_norm=gradient_norm,
        param_norm=param_norm,
    )
    return list(grads), metrics


@torch.no_grad()
def line_search_over_direction(
    model: torch.nn.Module,
    anchor_params: "OrderedDict[str, torch.Tensor]",
    direction: "OrderedDict[str, torch.Tensor]",
    batches: List[Tuple[torch.Tensor, torch.Tensor]],
    moe: bool,
    ls_range: int,
) -> Tuple[float, float]:
    """
    Evaluate candidate step sizes in [1, 1/sqrt(2), 1/2, ...], set the best params.
    """
    if not batches:
        return 1.0, float("nan")

    params = current_param_dict(model)
    best_step = 1.0
    best_loss = float("inf")
    best_params = clone_param_dict_from_named_params(params)

    for i in range(ls_range):
        step = 1.0 / (math.sqrt(2.0) ** i)
        candidate = add_scaled_param_dict(anchor_params, direction, step)
        apply_param_dict_(params, candidate)

        total = 0.0
        for x, y in batches:
            out = model(x, targets=None, get_logits=True, moe=moe)
            logits = out["logits"]
            total += _cross_entropy_from_logits(logits, y).item()
        avg_loss = total / len(batches)
        if avg_loss < best_loss:
            best_loss = avg_loss
            best_step = step
            best_params = clone_param_dict_from_named_params(params)

    apply_param_dict_(params, best_params)
    return best_step, best_loss
