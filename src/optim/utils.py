import json
import math
import os
import random
import tempfile
import uuid
from contextlib import nullcontext
from copy import deepcopy
from dataclasses import dataclass
from pathlib import Path
from typing import Mapping

import numpy as np
import torch
import torch.distributed as dist

try:
    import wandb
except ImportError:
    wandb = None


CHECKPOINT_KIND = "llm-optimizer-benchmark-training"
CHECKPOINT_FORMAT_VERSION = 3
WORKER_STATE_KIND = "llm-optimizer-benchmark-worker"
WORKER_STATE_FORMAT_VERSION = 2


class CheckpointCompatibilityError(RuntimeError):
    """Raised before restore when a checkpoint cannot preserve training semantics."""


@dataclass(frozen=True)
class CheckpointLoadResult:
    iteration: int
    training_state: dict
    run_identity: object | None
    world_size: int
    format_version: int
    averager_names: tuple[str, ...]
    snapshot_id: str | None


def get_batch(datareader, device="cpu"):
    x, y = datareader.sample_batch()
    if "cuda" in torch.device(device).type:
        # pin arrays x,y, which allows us to move them to GPU asynchronously (non_blocking=True)
        x = x.pin_memory().to(device, non_blocking=True)
        y = y.pin_memory().to(device, non_blocking=True)
    else:
        x = x.to(device)
        y = y.to(device)
    return x, y


@torch.no_grad()
def eval(
    model,
    reader,
    device="cpu",
    max_num_batches=24,
    ctx=nullcontext(),
    moe=False,
    get_router_logits=False,
    cfg=None,
):
    assert model.training == False

    loss_list_val, loss_list_aux_val = [], {}
    num_correct_targets = 0
    num_effective_targets = 0
    router_logits = []

    for idx in range(max_num_batches):
        x, y = get_batch(reader, device=device)
        with ctx:
            outputs = model(x, targets=y, get_logits=True, moe=moe)
        val_loss = outputs["loss"]

        loss_list_val.append(val_loss)
        valid_targets = y != -1
        predictions = outputs["logits"].argmax(-1)
        num_correct_targets += int(
            ((predictions == y) & valid_targets).sum().item()
        )
        num_effective_targets += int(valid_targets.sum().item())

        # auxiliary losses are optional
        for k, v in outputs["aux_losses"].items():
            loss_list_aux_val[k] = loss_list_aux_val.get(k, [])
            loss_list_aux_val[k].append(v)

        # router logits for MoE visualization
        if get_router_logits:
            # shape [layers, batch_size * sequence_length, num_experts]
            logits = outputs["router_logits"]
            # shape [max_batches, layers, batch_size * sequence_length, num_experts]
            router_logits.append(logits)

    if num_effective_targets == 0:
        raise ValueError("validation accuracy has no effective targets (all targets are -1).")
    val_acc = num_correct_targets / num_effective_targets
    val_loss = torch.stack(loss_list_val).mean().item()
    val_perplexity = 2.71828**val_loss
    val_aux_losses = {
        f"val/{k}": torch.stack(v).mean().item() for k, v in loss_list_aux_val.items()
    }

    if get_router_logits:
        # filter out the router logits that are not of the expected shape (happens for the last batch in
        # dataloader has a different batch size than the others)
        if cfg:
            intended_size = cfg.batch_size * cfg.sequence_length
        else:
            intended_size = x.shape[0] * x.shape[1]
        # shape [batches - 1, layers, batch_size * sequence_length, num_experts]
        router_logits = (
            torch.stack(
                [rl for rl in router_logits if rl.shape[1] == intended_size],
                dim=0,
            )
            .detach()
            .cpu()
        )

    return val_acc, val_loss, val_perplexity, val_aux_losses, router_logits


@torch.no_grad()
def eval_sweep_dropk(
    model,
    data_tensor,
    sequence_length,
    batch_size,
    n_heads,
    device="cpu",
    max_num_batches=24,
    ctx=nullcontext(),
):
    assert model.training == False

    x_axis, y_axis_pp, y_axis_acc, y_axis_loss = (
        torch.linspace(0.0, 0.95, 15),
        [],
        [],
        [],
    )
    loss_list_val, acc_list = [], []

    for frac in x_axis:
        drop_k = int(sequence_length * frac * n_heads)
        for _ in range(max_num_batches):
            x, y = get_batch(data_tensor, sequence_length, batch_size, device=device)
            with ctx:
                outputs = model(
                    x, targets=y, alpha_th=None, drop_k=drop_k, get_logits=True
                )
            loss_list_val.append(outputs["ce_loss"])
            acc_list.append((outputs["logits"].argmax(-1) == y).float().mean())

        y_axis_acc.append(torch.stack(acc_list).mean().item())
        y_axis_loss.append(np.mean(loss_list_val))
        y_axis_pp.append(2.71828 ** y_axis_loss[-1])

    return x_axis, y_axis_acc, y_axis_pp, y_axis_loss


@torch.no_grad()
def eval_sweep_alphath(
    model,
    data_tensor,
    sequence_length,
    batch_size,
    device="cpu",
    max_num_batches=24,
    ctx=nullcontext(),
):
    assert model.training == False

    alpha_ths, y_axis_pp, y_axis_acc, y_axis_loss = (
        [0, 1e-4, 1e-3, 1e-2, 1e-1, 2e-1, 3e-1, 4e-1, 5e-1],
        [],
        [],
        [],
    )
    loss_list_val, acc_list, x_axis = [], [], []

    for alpha_th in alpha_ths:
        frac_heads_pruned_list = []
        for _ in range(max_num_batches):
            x, y = get_batch(data_tensor, sequence_length, batch_size, device=device)
            with ctx:
                outputs = model(
                    x, targets=y, alpha_th=alpha_th, drop_k=None, get_logits=True
                )
            nph, nh = (
                outputs["num_head_pruned_per_layer"],
                outputs["num_heads_per_layer"],
            )
            frac_heads_pruned = np.sum(nph) / np.sum(
                nh
            )  # fractions of heads removed given alpha_th
            frac_heads_pruned_list.append(frac_heads_pruned)
            loss_list_val.append(outputs["ce_loss"])
            acc_list.append((outputs["logits"].argmax(-1) == y).float().mean())

        x_axis.append(np.mean(frac_heads_pruned_list))
        y_axis_acc.append(torch.stack(acc_list).mean().item())
        y_axis_loss.append(np.mean(loss_list_val))
        y_axis_pp.append(2.71828 ** y_axis_loss[-1])

    return x_axis, y_axis_acc, y_axis_pp, y_axis_loss


def _distributed_rank():
    if dist.is_available() and dist.is_initialized():
        return dist.get_rank()
    return int(os.environ.get("RANK", 0))


def _distributed_world_size():
    if dist.is_available() and dist.is_initialized():
        return dist.get_world_size()
    return int(os.environ.get("WORLD_SIZE", 1))


def _copy_mapping(value, name):
    if value is None:
        return {}
    if not isinstance(value, Mapping):
        raise TypeError(f"{name} must be a mapping, got {type(value).__name__}.")
    return deepcopy(dict(value))


def _copy_run_identity(value, name="run_identity"):
    if value is None:
        return None
    try:
        encoded = json.dumps(value, sort_keys=True, allow_nan=False)
        return json.loads(encoded)
    except (TypeError, ValueError) as error:
        raise TypeError(f"{name} must be JSON-serializable.") from error


def new_checkpoint_snapshot_id():
    """Return an opaque identifier shared by every file in one checkpoint."""
    return uuid.uuid4().hex


def _validate_snapshot_id(value, *, name="snapshot_id", error_type=ValueError):
    if not isinstance(value, str) or not value.strip():
        raise error_type(f"{name} must be a non-empty string.")
    return value


def atomic_torch_save(value, path):
    """Save a torch payload without exposing a partially written target file."""
    path = Path(path)
    path.parent.mkdir(exist_ok=True, parents=True)
    file_descriptor, temporary_name = tempfile.mkstemp(
        dir=path.parent,
        prefix=f".{path.name}.",
        suffix=".tmp",
    )
    temporary_path = Path(temporary_name)
    try:
        with os.fdopen(file_descriptor, "wb") as temporary_file:
            torch.save(value, temporary_file)
            temporary_file.flush()
            os.fsync(temporary_file.fileno())
        os.replace(temporary_path, path)
    except BaseException:
        try:
            os.close(file_descriptor)
        except OSError:
            pass
        temporary_path.unlink(missing_ok=True)
        raise


def optimizer_requires_rank_local_state(opt, world_size=None):
    """Return whether this optimizer owns different state on each active rank."""
    if opt is None or not getattr(opt, "requires_rank_local_state", False):
        return False
    world_size = _distributed_world_size() if world_size is None else int(world_size)
    optimizer_world_size = int(getattr(opt, "world_size", world_size))
    return world_size > 1 and optimizer_world_size > 1


def _averager_state_dicts(averagers):
    if averagers is None:
        return {}
    if not isinstance(averagers, Mapping):
        raise TypeError("averagers must be a mapping from name to averager.")
    states = {}
    for name, averager in averagers.items():
        if not isinstance(name, str) or not name:
            raise ValueError("averager names must be non-empty strings.")
        if not hasattr(averager, "state_dict"):
            raise TypeError(f"averager {name!r} does not provide state_dict().")
        states[name] = averager.state_dict()
    return states


def save_checkpoint(
    model,
    opt,
    scheduler,
    itr,
    ckpt_dir: Path,
    *,
    training_state=None,
    run_identity=None,
    averagers=None,
    world_size=None,
    snapshot_id=None,
):
    if isinstance(model, torch.nn.parallel.DistributedDataParallel):
        model = model.module

    iteration = int(itr)
    world_size = (
        _distributed_world_size() if world_size is None else int(world_size)
    )
    if world_size <= 0:
        raise ValueError(f"world_size must be positive, got {world_size}.")
    if snapshot_id is None:
        snapshot_id = new_checkpoint_snapshot_id()
    snapshot_id = _validate_snapshot_id(snapshot_id)
    saved_training_state = _copy_mapping(training_state, "training_state")
    recorded_iteration = saved_training_state.setdefault("iteration", iteration)
    if int(recorded_iteration) != iteration:
        raise ValueError(
            "training_state iteration does not match the checkpoint iteration: "
            f"{recorded_iteration!r} != {iteration}."
        )

    rank_local_optimizer = optimizer_requires_rank_local_state(opt, world_size)
    checkpoint = {
        "checkpoint_kind": CHECKPOINT_KIND,
        "format_version": CHECKPOINT_FORMAT_VERSION,
        "snapshot_id": snapshot_id,
        "world_size": world_size,
        "optimizer_state_scope": (
            "rank-local-workers" if rank_local_optimizer else "global"
        ),
        "model": model.state_dict(),
        "optimizer": None if rank_local_optimizer else opt.state_dict(),
        "scheduler": scheduler.state_dict() if scheduler is not None else None,
        "itr": iteration,
        "training_state": saved_training_state,
        "run_identity": _copy_run_identity(run_identity),
        "averagers": _averager_state_dicts(averagers),
    }
    atomic_torch_save(checkpoint, Path(ckpt_dir) / "main.pt")
    return snapshot_id


def _validate_versioned_checkpoint(ckpt):
    if not isinstance(ckpt, Mapping):
        raise CheckpointCompatibilityError("checkpoint payload must be a mapping.")
    if ckpt.get("checkpoint_kind") != CHECKPOINT_KIND:
        raise CheckpointCompatibilityError(
            "checkpoint kind is missing or incompatible; this is a legacy checkpoint."
        )
    version = ckpt.get("format_version")
    if version != CHECKPOINT_FORMAT_VERSION:
        raise CheckpointCompatibilityError(
            f"unsupported checkpoint format version {version!r}; "
            f"expected {CHECKPOINT_FORMAT_VERSION}."
        )
    required = {
        "snapshot_id",
        "world_size",
        "optimizer_state_scope",
        "model",
        "optimizer",
        "scheduler",
        "itr",
        "training_state",
        "run_identity",
        "averagers",
    }
    missing = sorted(required.difference(ckpt))
    if missing:
        raise CheckpointCompatibilityError(
            f"checkpoint is incomplete; missing fields: {', '.join(missing)}."
        )
    _validate_snapshot_id(
        ckpt["snapshot_id"],
        name="checkpoint snapshot_id",
        error_type=CheckpointCompatibilityError,
    )
    if not isinstance(ckpt["training_state"], Mapping):
        raise CheckpointCompatibilityError("checkpoint training_state must be a mapping.")
    try:
        _copy_run_identity(ckpt["run_identity"])
    except TypeError as error:
        raise CheckpointCompatibilityError(
            "checkpoint run_identity must be JSON-serializable."
        ) from error
    if not isinstance(ckpt["averagers"], Mapping):
        raise CheckpointCompatibilityError("checkpoint averagers must be a mapping.")
    saved_world_size = int(ckpt["world_size"])
    if saved_world_size <= 0:
        raise CheckpointCompatibilityError(
            f"checkpoint world size must be positive, got {saved_world_size}."
        )
    scope = ckpt["optimizer_state_scope"]
    if scope not in {"global", "rank-local-workers"}:
        raise CheckpointCompatibilityError(
            f"unsupported optimizer state scope: {scope!r}."
        )
    if scope == "rank-local-workers" and saved_world_size == 1:
        raise CheckpointCompatibilityError(
            "rank-local optimizer state is invalid for a single-rank checkpoint."
        )
    if scope == "rank-local-workers" and ckpt["optimizer"] is not None:
        raise CheckpointCompatibilityError(
            "rank-local optimizer state must be stored only in worker checkpoints."
        )
    if scope == "global" and ckpt["optimizer"] is None:
        raise CheckpointCompatibilityError(
            "global optimizer state is missing from the main checkpoint."
        )
    try:
        training_iteration = int(ckpt["training_state"]["iteration"])
    except (KeyError, TypeError, ValueError) as error:
        raise CheckpointCompatibilityError(
            "checkpoint training_state has no valid iteration."
        ) from error
    if training_iteration != int(ckpt["itr"]):
        raise CheckpointCompatibilityError(
            "checkpoint iteration fields disagree: "
            f"itr={ckpt['itr']!r}, training_state={training_iteration}."
        )


def _validate_scheduler_state(scheduler, saved_state):
    if scheduler is None and saved_state is not None:
        raise CheckpointCompatibilityError(
            "checkpoint contains scheduler state but no scheduler was provided."
        )
    if scheduler is not None and saved_state is None:
        raise CheckpointCompatibilityError(
            "checkpoint has no scheduler state for the provided scheduler."
        )


def _restore_averagers(averagers, saved_states):
    if averagers is None:
        if saved_states:
            raise CheckpointCompatibilityError(
                "checkpoint contains averager state but no averagers were provided."
            )
        return
    if not isinstance(averagers, Mapping):
        raise TypeError("averagers must be a mapping from name to averager.")
    expected_names = set(averagers)
    saved_names = set(saved_states)
    if expected_names != saved_names:
        missing = sorted(expected_names.difference(saved_names))
        extra = sorted(saved_names.difference(expected_names))
        raise CheckpointCompatibilityError(
            "checkpoint averager set does not match the active run: "
            f"missing={missing}, extra={extra}."
        )
    for name, averager in averagers.items():
        if not hasattr(averager, "load_state_dict"):
            raise TypeError(f"averager {name!r} does not provide load_state_dict().")
        averager.load_state_dict(saved_states[name])


def load_checkpoint(
    model,
    opt,
    scheduler,
    ckpt_path,
    device,
    *,
    averagers=None,
    expected_run_identity=None,
    expected_world_size=None,
    allow_legacy=False,
    return_metadata=False,
):
    if isinstance(model, torch.nn.parallel.DistributedDataParallel):
        model = model.module

    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
    has_schema_marker = isinstance(ckpt, Mapping) and (
        "checkpoint_kind" in ckpt or "format_version" in ckpt
    )
    is_versioned = has_schema_marker and ckpt.get("checkpoint_kind") == CHECKPOINT_KIND
    if not is_versioned:
        if has_schema_marker:
            _validate_versioned_checkpoint(ckpt)
        if not allow_legacy:
            raise CheckpointCompatibilityError(
                "legacy checkpoint has no supported schema; pass allow_legacy=True "
                "only for an explicitly non-exact resume."
            )
        required = {"model", "optimizer", "scheduler", "itr"}
        if not isinstance(ckpt, Mapping) or not required.issubset(ckpt):
            raise CheckpointCompatibilityError("legacy checkpoint is incomplete.")
        _validate_scheduler_state(scheduler, ckpt["scheduler"])
        model.load_state_dict(ckpt["model"])
        opt.load_state_dict(ckpt["optimizer"])
        if scheduler is not None:
            scheduler.load_state_dict(ckpt["scheduler"])
        result = CheckpointLoadResult(
            iteration=int(ckpt["itr"]),
            training_state={},
            run_identity=None,
            world_size=1,
            format_version=0,
            averager_names=(),
            snapshot_id=None,
        )
        return result if return_metadata else result.iteration

    _validate_versioned_checkpoint(ckpt)
    saved_world_size = int(ckpt["world_size"])
    if expected_world_size is None:
        expected_world_size = _distributed_world_size()
    if saved_world_size != int(expected_world_size):
        raise CheckpointCompatibilityError(
            "checkpoint world size does not match the active run: "
            f"saved={saved_world_size}, expected={int(expected_world_size)}."
        )
    active_scope = (
        "rank-local-workers"
        if optimizer_requires_rank_local_state(opt, expected_world_size)
        else "global"
    )
    if ckpt["optimizer_state_scope"] != active_scope:
        raise CheckpointCompatibilityError(
            "checkpoint optimizer state scope does not match the active optimizer: "
            f"saved={ckpt['optimizer_state_scope']!r}, expected={active_scope!r}."
        )

    if expected_run_identity is not None:
        expected_identity = _copy_run_identity(
            expected_run_identity,
            "expected_run_identity",
        )
        saved_identity = _copy_run_identity(ckpt["run_identity"])
        if saved_identity != expected_identity:
            raise CheckpointCompatibilityError(
                "checkpoint run identity does not match the active run."
            )

    _validate_scheduler_state(scheduler, ckpt["scheduler"])
    _restore_averagers(averagers, ckpt["averagers"])
    model.load_state_dict(ckpt["model"])
    if ckpt["optimizer_state_scope"] == "global":
        opt.load_state_dict(ckpt["optimizer"])
    if scheduler is not None:
        scheduler.load_state_dict(ckpt["scheduler"])

    result = CheckpointLoadResult(
        iteration=int(ckpt["itr"]),
        training_state=deepcopy(dict(ckpt["training_state"])),
        run_identity=_copy_run_identity(ckpt["run_identity"]),
        world_size=saved_world_size,
        format_version=int(ckpt["format_version"]),
        averager_names=tuple(sorted(ckpt["averagers"])),
        snapshot_id=ckpt["snapshot_id"],
    )
    return result if return_metadata else result.iteration



def extend_onecycle_total_steps(scheduler, new_total_steps):
    if scheduler is None:
        return
    schedulers = (
        scheduler.schedulers if hasattr(scheduler, "schedulers") else [scheduler]
    )
    for sched in schedulers:
        if not hasattr(sched, "total_steps"):
            continue
        if new_total_steps <= sched.total_steps:
            continue
        if hasattr(sched, "_schedule_phases") and sched._schedule_phases:
            pct_start = (sched._schedule_phases[0]["end_step"] + 1) / sched.total_steps
            three_phase = len(sched._schedule_phases) == 3
            sched.total_steps = new_total_steps
            if three_phase:
                sched._schedule_phases = [
                    {
                        "end_step": float(pct_start * new_total_steps) - 1,
                        "start_lr": "initial_lr",
                        "end_lr": "max_lr",
                        "start_momentum": "max_momentum",
                        "end_momentum": "base_momentum",
                    },
                    {
                        "end_step": float(2 * pct_start * new_total_steps) - 2,
                        "start_lr": "max_lr",
                        "end_lr": "initial_lr",
                        "start_momentum": "base_momentum",
                        "end_momentum": "max_momentum",
                    },
                    {
                        "end_step": new_total_steps - 1,
                        "start_lr": "initial_lr",
                        "end_lr": "min_lr",
                        "start_momentum": "max_momentum",
                        "end_momentum": "max_momentum",
                    },
                ]
            else:
                sched._schedule_phases = [
                    {
                        "end_step": float(pct_start * new_total_steps) - 1,
                        "start_lr": "initial_lr",
                        "end_lr": "max_lr",
                        "start_momentum": "max_momentum",
                        "end_momentum": "base_momentum",
                    },
                    {
                        "end_step": new_total_steps - 1,
                        "start_lr": "max_lr",
                        "end_lr": "min_lr",
                        "start_momentum": "base_momentum",
                        "end_momentum": "max_momentum",
                    },
                ]
        else:
            sched.total_steps = new_total_steps


def save_worker_state(
    ckpt_dir: Path,
    opt=None,
    training_state=None,
    *,
    rank=None,
    world_size=None,
    snapshot_id=None,
):
    rank = _distributed_rank() if rank is None else int(rank)
    world_size = (
        _distributed_world_size() if world_size is None else int(world_size)
    )
    if world_size <= 0:
        raise ValueError(f"world_size must be positive, got {world_size}.")
    if rank < 0 or rank >= world_size:
        raise ValueError(f"rank {rank} is outside world size {world_size}.")

    if snapshot_id is None:
        raise ValueError(
            "snapshot_id is required so worker state can be bound to main.pt."
        )
    snapshot_id = _validate_snapshot_id(snapshot_id)

    rank_local_optimizer = optimizer_requires_rank_local_state(opt, world_size)
    worker_state = {
        "checkpoint_kind": WORKER_STATE_KIND,
        "format_version": WORKER_STATE_FORMAT_VERSION,
        "snapshot_id": snapshot_id,
        "rank": rank,
        "world_size": world_size,
        "requires_rank_local_optimizer_state": rank_local_optimizer,
        "training_state": _copy_mapping(training_state, "training_state"),
        "rng_torch_cpu": torch.random.get_rng_state(),
        "rng_torch_gpu": torch.cuda.get_rng_state() if torch.cuda.is_available() else None,
        "rng_np": np.random.get_state(),
        "rng_python": random.getstate(),
    }
    if rank_local_optimizer:
        worker_state["optimizer"] = opt.state_dict()
    atomic_torch_save(worker_state, Path(ckpt_dir) / f"worker_{rank}.pt")


def _validate_worker_state(worker_state, rank, expected_world_size):
    if not isinstance(worker_state, Mapping):
        raise CheckpointCompatibilityError("worker checkpoint payload must be a mapping.")
    if worker_state.get("checkpoint_kind") != WORKER_STATE_KIND:
        raise CheckpointCompatibilityError(
            "worker checkpoint kind is missing or incompatible."
        )
    if worker_state.get("format_version") != WORKER_STATE_FORMAT_VERSION:
        raise CheckpointCompatibilityError(
            "unsupported worker checkpoint format version "
            f"{worker_state.get('format_version')!r}."
        )
    required = {
        "snapshot_id",
        "rank",
        "world_size",
        "requires_rank_local_optimizer_state",
        "training_state",
        "rng_torch_cpu",
        "rng_torch_gpu",
        "rng_np",
        "rng_python",
    }
    missing = sorted(required.difference(worker_state))
    if missing:
        raise CheckpointCompatibilityError(
            f"worker checkpoint is incomplete; missing fields: {', '.join(missing)}."
        )
    _validate_snapshot_id(
        worker_state["snapshot_id"],
        name="worker checkpoint snapshot_id",
        error_type=CheckpointCompatibilityError,
    )
    if int(worker_state["rank"]) != rank:
        raise CheckpointCompatibilityError(
            f"worker checkpoint rank mismatch: saved={worker_state['rank']}, expected={rank}."
        )
    if int(worker_state["world_size"]) != int(expected_world_size):
        raise CheckpointCompatibilityError(
            "worker checkpoint world size does not match the active run: "
            f"saved={worker_state['world_size']}, expected={int(expected_world_size)}."
        )
    if not isinstance(worker_state["training_state"], Mapping):
        raise CheckpointCompatibilityError(
            "worker checkpoint training_state must be a mapping."
        )


def load_worker_state(
    ckpt_dir: Path,
    opt=None,
    *,
    rank=None,
    expected_world_size=None,
    expected_snapshot_id=None,
    allow_legacy=False,
):
    rank = _distributed_rank() if rank is None else int(rank)
    if expected_world_size is None:
        expected_world_size = _distributed_world_size()
    worker_state = torch.load(
        Path(ckpt_dir) / f"worker_{rank}.pt",
        weights_only=False,
    )
    has_schema_marker = isinstance(worker_state, Mapping) and (
        "checkpoint_kind" in worker_state or "format_version" in worker_state
    )
    is_versioned = (
        has_schema_marker
        and worker_state.get("checkpoint_kind") == WORKER_STATE_KIND
    )
    if not is_versioned:
        if has_schema_marker:
            _validate_worker_state(worker_state, rank, expected_world_size)
        if not allow_legacy:
            raise CheckpointCompatibilityError(
                "legacy worker checkpoint has no supported schema; pass "
                "allow_legacy=True only for an explicitly non-exact resume."
            )
        required = {"rng_torch_cpu", "rng_torch_gpu", "rng_np", "rng_python"}
        if not isinstance(worker_state, Mapping) or not required.issubset(worker_state):
            raise CheckpointCompatibilityError("legacy worker checkpoint is incomplete.")
        training_state = {}
    else:
        _validate_worker_state(worker_state, rank, expected_world_size)
        if expected_snapshot_id is None:
            raise ValueError(
                "expected_snapshot_id is required to prevent restoring worker "
                "state from a different main checkpoint."
            )
        expected_snapshot_id = _validate_snapshot_id(
            expected_snapshot_id,
            name="expected_snapshot_id",
        )
        if worker_state["snapshot_id"] != expected_snapshot_id:
            raise CheckpointCompatibilityError(
                "worker checkpoint belongs to a different snapshot: "
                f"worker={worker_state['snapshot_id']!r}, "
                f"main={expected_snapshot_id!r}."
            )
        saved_requires_local = bool(
            worker_state["requires_rank_local_optimizer_state"]
        )
        active_requires_local = optimizer_requires_rank_local_state(
            opt,
            expected_world_size,
        )
        if saved_requires_local != active_requires_local:
            raise CheckpointCompatibilityError(
                "worker optimizer state scope does not match the active optimizer."
            )
        if saved_requires_local:
            if opt is None or "optimizer" not in worker_state:
                raise CheckpointCompatibilityError(
                    "worker checkpoint is missing required rank-local optimizer state."
                )
            opt.load_state_dict(worker_state["optimizer"])
        training_state = deepcopy(dict(worker_state["training_state"]))

    torch.random.set_rng_state(worker_state["rng_torch_cpu"])
    if torch.cuda.is_available() and worker_state["rng_torch_gpu"] is not None:
        torch.cuda.set_rng_state(worker_state["rng_torch_gpu"])
    np.random.set_state(worker_state["rng_np"])
    random.setstate(worker_state["rng_python"])
    return training_state


def get_parameter_norms(model, order=2):
    model_norm = 0
    for p in model.parameters():
        param_data = p.detach().data
        if order == float("inf"):
            param_norm = param_data.norm(p=order)
            model_norm = max(model_norm, param_norm.item())
        else:
            param_norm = param_data.norm(p=order)
            model_norm += param_norm.item() ** order

    if order != float("inf"):
        model_norm = model_norm ** (1.0 / order)

    return model_norm


def log_prodigy_lr(opt):
    effective_lrs = []

    for group in opt.param_groups:
        d = group["d"]
        lr = group["lr"]
        if group["use_bias_correction"]:
            k = group["k"]
            beta1, beta2 = group["betas"]
            bias_correction = ((1 - beta2 ** (k + 1)) ** 0.5) / (1 - beta1 ** (k + 1))
        else:
            bias_correction = 1
        effective_lr = d * lr * bias_correction
        effective_lrs.append(effective_lr)

    return effective_lrs


def visualize_routing(router_logits, extra_args):
    # router_logits: [batches, layers, batch_size * sequence_length, num_experts]
    logs = {}

    n_layers = extra_args.n_layer
    num_experts = extra_args.moe_num_experts
    num_experts_per_tok = extra_args.moe_num_experts_per_tok

    # histogram over all logits to see distribution
    logs["router/logits"] = wandb.Histogram(
        router_logits.type(torch.float32).flatten().cpu().numpy()
    )

    # distribution over experts for layer 0, layer n/2, n-1
    for layer in [0, n_layers // 2, n_layers - 1]:
        router_logits_layer = router_logits[:, layer]
        # shape [batches, batch_size * sequence_length, num_experts_per_tok]
        weights, selected_experts = torch.topk(
            router_logits_layer, num_experts_per_tok, dim=-1
        )
        # shape [batches, batch_size * sequence_length, num_experts_per_tok, num_experts]
        expert_mask = torch.nn.functional.one_hot(selected_experts, num_experts)
        # For a given token, determine if it was routed to a given expert.
        # Shape: [batches, batch_size * sequence_length, num_experts]
        expert_mask, _ = torch.max(expert_mask, dim=-2)
        # shape [num_experts]
        tokens_per_expert = torch.mean(expert_mask, dim=(0, 1), dtype=torch.float32)
        layer_token_routing = {
            f"router/layer_{layer}_expert_{i}_selection": tokens_per_expert[i].item()
            for i in range(num_experts)
        }
        logs.update(layer_token_routing)
    return logs
