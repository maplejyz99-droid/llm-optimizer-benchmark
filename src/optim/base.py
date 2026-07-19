import copy
import hashlib
import math
import os
import random
import time
import uuid
from contextlib import contextmanager, nullcontext
from pathlib import Path

import numpy as np
import torch
import yaml

try:
    import wandb
except ImportError:
    wandb = None

from logger.logger import DynamicsLogger
from notify import maybe_notify
from optim.weight_averaging import (ExponentialWeightAverager, WeightAverager,
                                    eval_ewa, eval_wa)

from .gn import (clone_param_dict, clone_param_dict_from_named_params,
                 compute_gn_step, current_param_dict, line_search_over_direction,
                 sub_param_dict)
from .utils import (atomic_torch_save, eval, extend_onecycle_total_steps, get_batch,
                    get_parameter_norms, load_checkpoint, load_worker_state,
                    log_prodigy_lr, save_checkpoint, save_worker_state,
                    visualize_routing)


def _distributed_barrier(distributed_backend):
    barrier = getattr(distributed_backend, "barrier", None)
    if barrier is not None:
        barrier()
        return
    if distributed_backend.get_world_size() != 1:
        raise RuntimeError("Distributed backend does not provide barrier().")

def _broadcast_object(distributed_backend, value, src=0):
    broadcast = getattr(distributed_backend, "broadcast_object", None)
    if broadcast is not None:
        return broadcast(value, src=src)
    if distributed_backend.get_world_size() == 1:
        return value
    raise RuntimeError("Distributed backend does not provide broadcast_object().")


def _all_gather_object(distributed_backend, value):
    all_gather = getattr(distributed_backend, "all_gather_object", None)
    if all_gather is not None:
        return all_gather(value)
    if distributed_backend.get_world_size() == 1:
        return [value]
    raise RuntimeError("Distributed backend does not provide all_gather_object().")


def _distributed_reduce_mean(distributed_backend, value):
    reduce_mean = getattr(distributed_backend, "reduce_mean", None)
    if reduce_mean is not None:
        return reduce_mean(value)
    if distributed_backend.get_world_size() == 1:
        return value
    raise RuntimeError("Distributed backend does not provide reduce_mean().")


def _run_synchronized_action(
    distributed_backend,
    action_name,
    action,
    *,
    master_only=False,
):
    """Run an action and make every rank fail before entering a new collective."""
    local_exception = None
    result = None
    if not master_only or distributed_backend.is_master_process():
        try:
            result = action()
        except BaseException as exc:
            local_exception = exc

    failure = None
    if local_exception is not None:
        failure = {
            "rank": int(getattr(distributed_backend, "rank", 0)),
            "type": type(local_exception).__name__,
            "message": str(local_exception),
        }
    failures = [
        item
        for item in _all_gather_object(distributed_backend, failure)
        if item is not None
    ]
    if failures:
        if local_exception is not None:
            raise local_exception
        first = failures[0]
        raise RuntimeError(
            f"{action_name} failed on rank {first['rank']}: "
            f"{first['type']}: {first['message']}"
        )
    return result


@contextmanager
def _preserve_rng_state():
    """Keep evaluation and text generation from changing training RNG streams."""
    python_state = random.getstate()
    numpy_state = np.random.get_state()
    random_module = getattr(torch, "random", None)
    get_cpu_state = getattr(random_module, "get_rng_state", None)
    set_cpu_state = getattr(random_module, "set_rng_state", None)
    cpu_state = get_cpu_state() if get_cpu_state is not None else None

    cuda_module = getattr(torch, "cuda", None)
    cuda_available = (
        cuda_module is not None
        and hasattr(cuda_module, "is_available")
        and cuda_module.is_available()
    )
    get_cuda_states = getattr(cuda_module, "get_rng_state_all", None)
    set_cuda_states = getattr(cuda_module, "set_rng_state_all", None)
    cuda_states = (
        get_cuda_states()
        if cuda_available and get_cuda_states is not None
        else None
    )
    try:
        yield
    finally:
        random.setstate(python_state)
        np.random.set_state(numpy_state)
        if cpu_state is not None and set_cpu_state is not None:
            set_cpu_state(cpu_state)
        if cuda_states is not None and set_cuda_states is not None:
            set_cuda_states(cuda_states)


def _reader_step(train_reader, fallback):
    return int(getattr(train_reader, "step", fallback))


def _save_training_checkpoints(
    model,
    opt,
    scheduler,
    curr_iter,
    exp_dir,
    distributed_backend,
    cfg,
    train_reader,
    substep,
    averagers,
):
    def save_at(ckpt_dir):
        def build_training_state():
            reader_step = _reader_step(train_reader, substep)
            if reader_step != substep:
                raise RuntimeError(
                    "Training reader step diverged from substep before checkpoint: "
                    f"reader={reader_step}, substep={substep}."
                )
            return {
                "iteration": int(curr_iter),
                "train_reader_step": reader_step,
                "substep": int(substep),
            }

        training_state = _run_synchronized_action(
            distributed_backend,
            "checkpoint preparation",
            build_training_state,
        )
        world_size = distributed_backend.get_world_size()
        rank = getattr(distributed_backend, "rank", 0)

        def save_main_checkpoint():
            return save_checkpoint(
                model,
                opt,
                scheduler,
                curr_iter,
                ckpt_dir,
                training_state=training_state,
                run_identity=getattr(cfg, "run_identity", None),
                world_size=world_size,
                averagers=averagers or None,
            )

        snapshot_id = _run_synchronized_action(
            distributed_backend,
            "main checkpoint save",
            save_main_checkpoint,
            master_only=True,
        )
        snapshot_id = _broadcast_object(
            distributed_backend,
            snapshot_id,
            src=0,
        )
        # The main checkpoint creates the directory and must be durable before
        # rank-local RNG/optimizer files are published into the same snapshot.
        _distributed_barrier(distributed_backend)

        _run_synchronized_action(
            distributed_backend,
            "worker checkpoint save",
            lambda: save_worker_state(
                ckpt_dir,
                opt=opt,
                training_state=training_state,
                world_size=world_size,
                rank=rank,
                snapshot_id=snapshot_id,
            ),
        )
        _distributed_barrier(distributed_backend)

    # Preserve the existing behavior: checkpoints are considered at iter 0,
    # at matching intervals, and at the final iteration for latest checkpoints.
    if cfg.permanent_ckpt_interval > 0:
        if curr_iter % cfg.permanent_ckpt_interval == 0:
            ckpt_dir = exp_dir / "ckpts" / str(curr_iter)
            save_at(ckpt_dir)

    if cfg.latest_ckpt_interval > 0:
        if curr_iter % cfg.latest_ckpt_interval == 0 or curr_iter == cfg.iterations:
            ckpt_dir = exp_dir / "ckpts" / "latest"
            save_at(ckpt_dir)


def _is_full_eval(curr_iter, cfg):
    return curr_iter in cfg.full_eval_at


def _should_run_eval(curr_iter, cfg):
    return (
        curr_iter % cfg.eval_interval == 0
        or curr_iter == cfg.iterations
        or _is_full_eval(curr_iter, cfg)
    )


def _is_schedulefree(cfg):
    return cfg.opt in {"sf-sgd", "sf-adamw"}


def _enter_train_mode(model, opt, cfg):
    model.train()
    if _is_schedulefree(cfg):
        opt.train()


def _clip_grad_norm(model, cfg):
    if isinstance(model, torch.nn.parallel.DistributedDataParallel):
        return torch.nn.utils.clip_grad_norm_(
            model.module.parameters(), cfg.grad_clip
        )
    return torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.grad_clip)


def _resume_training_state(
    model,
    opt,
    scheduler,
    cfg,
    averagers,
    expected_world_size,
    rank,
    use_gn,
):
    if cfg.resume_from:
        # This is a full resume including the model weights, optimizer, state
        # dataloader state, random seed, etc. Not indended for fine tuning or
        # other scenarios where some of these should change.
        print(f"\nResuming Training From {cfg.resume_from}")
        ckpt_dir = Path(cfg.resume_from)
        allow_legacy = getattr(cfg, "allow_legacy_checkpoint_resume", False)
        checkpoint = load_checkpoint(
            model,
            opt,
            scheduler,
            ckpt_dir / "main.pt",
            cfg.device,
            averagers=averagers or None,
            expected_run_identity=getattr(cfg, "run_identity", None),
            expected_world_size=expected_world_size,
            allow_legacy=allow_legacy,
            return_metadata=True,
        )
        worker_training_state = load_worker_state(
            ckpt_dir,
            opt=opt,
            rank=rank,
            expected_world_size=expected_world_size,
            expected_snapshot_id=checkpoint.snapshot_id,
            allow_legacy=allow_legacy,
        )
        extend_onecycle_total_steps(scheduler, cfg.iterations)

        curr_iter = int(checkpoint.iteration)
        if checkpoint.format_version == 0:
            if use_gn:
                raise RuntimeError(
                    "GN cannot resume from a legacy checkpoint because its exact "
                    "training reader position was not recorded."
                )
            # Explicit legacy opt-in preserves the old approximation for standard
            # optimizers only. It is intentionally not considered exact resume.
            return curr_iter, curr_iter * cfg.acc_steps

        training_state = checkpoint.training_state
        try:
            saved_iteration = int(training_state["iteration"])
            reader_step = int(training_state["train_reader_step"])
            substep = int(training_state["substep"])
        except (KeyError, TypeError, ValueError) as exc:
            raise RuntimeError(
                "Checkpoint training_state must contain integer iteration, "
                "train_reader_step, and substep values."
            ) from exc
        if saved_iteration != curr_iter:
            raise RuntimeError(
                "Checkpoint training_state iteration does not match checkpoint "
                f"iteration: {saved_iteration} != {curr_iter}."
            )
        if reader_step < 0 or substep < 0:
            raise RuntimeError("Checkpoint reader position cannot be negative.")
        if reader_step != substep:
            raise RuntimeError(
                "Checkpoint train_reader_step does not match substep: "
                f"{reader_step} != {substep}."
            )

        if worker_training_state:
            for key in ("iteration", "train_reader_step", "substep"):
                if key in worker_training_state and int(worker_training_state[key]) != int(
                    training_state[key]
                ):
                    raise RuntimeError(
                        f"Worker checkpoint {key} does not match main checkpoint."
                    )
        return curr_iter, reader_step
    return 0, 0


def _should_log_train_step(curr_iter, cfg, distributed_backend):
    return (
        cfg.log_interval
        and curr_iter % cfg.log_interval == 0
        and distributed_backend.is_master_process()
    )


def _metric_to_float(value):
    if hasattr(value, "detach"):
        value = value.detach()
    if hasattr(value, "cpu"):
        value = value.cpu()
    if hasattr(value, "item"):
        value = value.item()
    return float(value)


def _log_training_step(
    curr_iter,
    model,
    epoch,
    tokens,
    loss,
    outputs,
    opt,
    distributed_backend,
    cfg,
    dt,
    elapsed_seconds,
    train_step_seconds_total,
    avg_iter_dt,
    grad_norms,
    use_gn,
    gn_step_size,
):
    train_loss = _metric_to_float(loss)
    last_train_loss = train_loss
    train_aux_losses = {
        f"train/{k}": _metric_to_float(v)
        for k, v in outputs["aux_losses"].items()
    }

    current_lrs = [param_group["lr"] for param_group in opt.param_groups]
    last_lr = current_lrs[0]

    if cfg.opt == "prodigy":
        prodigy_efective_lrs = log_prodigy_lr(opt)

    print(
        f"Train: Iter={curr_iter} ({epoch:0.3f} epochs) "
        f"train_loss={train_loss:.3f} iter_dt={dt:.2e}s "
        f"elapsed={elapsed_seconds:.2f}s "
        f"train_step_total={train_step_seconds_total:.2f}s "
        f"avg_iter_dt={avg_iter_dt:.2e}s "
        f"lr={current_lrs[0]:.2e}"
    )
    if cfg.opt == "prodigy":
        print(f"effective_lr={prodigy_efective_lrs[0]:.2e}")

    if cfg.wandb:
        wandb_logs = {
            "tokens": tokens,
            "iter": curr_iter,
            "train/loss": train_loss,
            "train/perplexity": 2.71828**train_loss,
            "lr": current_lrs[0],
            "iter_dt": dt,
            "wall_clock/elapsed_seconds": elapsed_seconds,
            "train/step_seconds": dt,
            "train/step_seconds_total": train_step_seconds_total,
            "train/avg_step_seconds": avg_iter_dt,
            "max_grad_norm": max(grad_norms).item() if grad_norms else 0,
            "mean_grad_norm": (
                torch.tensor(grad_norms).mean().item() if grad_norms else 0
            ),
            **train_aux_losses,
        }

        if cfg.opt == "prodigy":
            wandb_logs["effective_lr"] = prodigy_efective_lrs[0]
        if use_gn and gn_step_size is not None:
            wandb_logs["train/gn_step_size"] = gn_step_size

        if cfg.log_parameter_norms:
            raw_model = distributed_backend.get_raw_model(model)
            model_norm = get_parameter_norms(raw_model, order=cfg.norm_order)
            wandb_logs["model_norm"] = model_norm

        try:
            wandb.log(wandb_logs)
        except Exception as exc:
            print(f"Warning: training telemetry failed: {exc}")

    return last_train_loss, last_lr, []


def _notify_training_progress(
    curr_iter,
    epoch,
    exp_dir,
    opt,
    distributed_backend,
    cfg,
    last_train_loss,
    last_val_loss,
    last_val_pp,
    last_val_acc,
    last_lr,
    last_iter_dt,
):
    if distributed_backend.is_master_process():
        current_lrs = [param_group["lr"] for param_group in opt.param_groups]
        maybe_notify(
            cfg,
            curr_iter=curr_iter,
            epoch=epoch,
            train_loss=last_train_loss,
            val_loss=last_val_loss,
            val_pp=last_val_pp,
            val_acc=last_val_acc,
            lr=current_lrs[0] if current_lrs else last_lr,
            iter_dt=last_iter_dt,
            run_name=exp_dir.name,
        )


def _run_weight_average_evals(
    curr_iter,
    not_compiled_model,
    weight_averager,
    ewa,
    val_reader,
    type_ctx,
    distributed_backend,
    cfg,
):
    evaluation_counts = {}
    if curr_iter > cfg.wa_interval and cfg.weight_average:
        counts = eval_wa(
            curr_iter,
            not_compiled_model,
            weight_averager,
            val_reader,
            type_ctx,
            distributed_backend,
            cfg,
            full_eval=_is_full_eval(curr_iter, cfg),
        )
        if counts is not None:
            evaluation_counts["wa"] = counts

    if cfg.exponential_weight_average:
        counts = eval_ewa(
            curr_iter,
            not_compiled_model,
            ewa,
            val_reader,
            type_ctx,
            distributed_backend,
            cfg,
            full_eval=_is_full_eval(curr_iter, cfg),
        )
        if counts is not None:
            evaluation_counts["ewa"] = counts

    return evaluation_counts


def _step_weight_averagers(
    not_compiled_model,
    weight_averager,
    ewa,
    distributed_backend,
    cfg,
):
    if cfg.weight_average:
        step_weight_average = lambda: weight_averager.step(
            not_compiled_model, distributed_backend.is_master_process()
        )
        if weight_averager.publishes_on_next_step():
            _run_synchronized_action(
                distributed_backend,
                "weight-average horizon publish",
                step_weight_average,
            )
        else:
            step_weight_average()
    if cfg.exponential_weight_average:
        ewa.step(not_compiled_model, distributed_backend.is_master_process())


def _run_accumulation_microsteps(
    model,
    opt,
    train_reader,
    type_ctx,
    distributed_backend,
    cfg,
):
    has_precond_flag = hasattr(opt, "precond_flag_for_step")
    precond_flag = opt.precond_flag_for_step() if has_precond_flag else False
    x = y = outputs = None
    microstep_losses = []
    aux_losses = {}
    local_effective_targets = 0
    accumulation_batches = []

    for microstep_idx in range(cfg.acc_steps):  # gradient accumulation
        x, y = get_batch(train_reader, device=cfg.device)
        accumulation_batches.append((x, y))
        with type_ctx:
            with distributed_backend.get_context_for_microstep_forward(
                model=model,
                microstep_idx=microstep_idx,
                gradient_accumulation_steps=cfg.acc_steps,
            ):
                if has_precond_flag:
                    outputs = model(
                        x,
                        targets=y,
                        moe=cfg.moe,
                        precond_flag=precond_flag,
                    )
                else:
                    outputs = model(x, targets=y, moe=cfg.moe)

        raw_loss = outputs["loss"]
        (raw_loss / cfg.acc_steps).backward()
        microstep_losses.append(raw_loss.detach())
        local_effective_targets += _count_effective_targets(y, cfg)
        for name, value in outputs["aux_losses"].items():
            aux_losses.setdefault(name, []).append(value.detach())

    local_loss = _mean_microstep_metrics(microstep_losses)
    aux_names = sorted(aux_losses)
    local_metrics = [local_loss]
    local_metrics.extend(
        _mean_microstep_metrics(aux_losses[name]) for name in aux_names
    )
    local_metrics.append(
        torch.tensor(float(local_effective_targets), device=cfg.device)
    )
    reduced_metrics = _distributed_reduce_mean(
        distributed_backend,
        _stack_metrics(local_metrics),
    )
    step_loss = reduced_metrics[0]
    step_aux_losses = {
        name: reduced_metrics[index + 1]
        for index, name in enumerate(aux_names)
    }
    global_effective_targets = int(
        round(
            _metric_to_float(reduced_metrics[-1])
            * distributed_backend.get_world_size()
        )
    )
    outputs = {
        "loss": step_loss,
        "aux_losses": step_aux_losses,
    }
    return (
        x,
        y,
        outputs,
        step_loss,
        cfg.acc_steps,
        global_effective_targets,
        accumulation_batches,
    )


def _mean_microstep_metrics(values):
    if not values:
        raise ValueError("cannot average an empty microstep metric list")
    if hasattr(torch, "stack"):
        return torch.stack(values).mean()
    # The behavior harness provides a minimal fake torch without stack().
    return torch.tensor(values).mean()


def _stack_metrics(values):
    if hasattr(torch, "stack"):
        return torch.stack(values)
    return torch.tensor(values)


def _count_effective_targets(targets, cfg):
    try:
        count = (targets != -1).sum()
    except (AttributeError, TypeError):
        # The lightweight behavior harness uses string batches. Production
        # readers always return tensors, so this preserves the declared shape.
        return int(cfg.batch_size * cfg.sequence_length)
    return int(_metric_to_float(count))


def _global_effective_target_count(targets, cfg, distributed_backend):
    local_count = _count_effective_targets(targets, cfg)
    world_size = distributed_backend.get_world_size()
    if world_size == 1:
        return local_count
    mean_count = _distributed_reduce_mean(
        distributed_backend,
        torch.tensor(float(local_count), device=cfg.device)
    )
    return int(round(_metric_to_float(mean_count) * world_size))


def _global_count_from_local(local_count, cfg, distributed_backend):
    world_size = distributed_backend.get_world_size()
    if world_size == 1:
        return int(local_count)
    mean_count = _distributed_reduce_mean(
        distributed_backend,
        torch.tensor(float(local_count), device=cfg.device),
    )
    return int(round(_metric_to_float(mean_count) * world_size))


def _count_batch_examples(batch, cfg):
    shape = getattr(batch, "shape", None)
    if shape is not None and len(shape) > 0:
        return int(shape[0])
    return int(cfg.batch_size)


def _run_sophia_hessian_estimator(
    model,
    opt,
    accumulation_batches,
    distributed_backend,
    cfg,
):
    mode = getattr(cfg, "sophia_estimator_mode", "legacy_last_microbatch")
    estimator_batches = (
        accumulation_batches
        if mode == "global_accum"
        else accumulation_batches[-1:]
    )
    local_examples = 0
    local_tokens = 0
    for estimator_idx, (batch_x, batch_y) in enumerate(estimator_batches):
        sync_context = (
            distributed_backend.get_context_for_microstep_forward(
                model=model,
                microstep_idx=estimator_idx,
                gradient_accumulation_steps=len(estimator_batches),
            )
            if mode == "global_accum"
            else nullcontext()
        )
        with sync_context:
            sample_again = model(batch_x, targets=batch_y, get_logits=True)
            samp_dist = torch.distributions.Categorical(
                logits=sample_again["logits"]
            )
            y_sample = samp_dist.sample()
            loss_sampled = torch.nn.functional.cross_entropy(
                sample_again["logits"].view(
                    -1, sample_again["logits"].size(-1)
                ),
                y_sample.view(-1),
                ignore_index=-1,
            )
            (loss_sampled / cfg.acc_steps).backward()
        local_examples += _count_batch_examples(batch_x, cfg)
        local_tokens += _count_effective_targets(batch_y, cfg)

    opt.update_hessian()
    opt.zero_grad(set_to_none=True)
    model.zero_grad()
    return {
        "examples": _global_count_from_local(
            local_examples, cfg, distributed_backend
        ),
        "tokens": _global_count_from_local(local_tokens, cfg, distributed_backend),
    }


def _update_digest_with_tensor(digest, label, tensor):
    digest.update(label.encode("utf-8"))
    digest.update(str(tensor.dtype).encode("ascii"))
    digest.update(repr(tuple(tensor.shape)).encode("ascii"))
    byte_view = tensor.detach().contiguous().view(torch.uint8).cpu().numpy()
    digest.update(memoryview(byte_view))


def _sophia_rank_state_digest(model, opt, distributed_backend):
    raw_model = distributed_backend.get_raw_model(model)
    digest = hashlib.sha256()
    parameter_names = {id(param): name for name, param in raw_model.named_parameters()}
    for name, param in sorted(raw_model.named_parameters()):
        _update_digest_with_tensor(digest, f"model:{name}", param)
    optimizer_states = []
    for param, state in opt.state.items():
        name = parameter_names.get(id(param))
        if name is None:
            continue
        for state_name in ("exp_avg", "hessian"):
            value = state.get(state_name)
            if value is not None:
                optimizer_states.append((name, state_name, value))
    for name, state_name, value in sorted(
        optimizer_states, key=lambda item: (item[0], item[1])
    ):
        _update_digest_with_tensor(
            digest, f"optimizer:{name}:{state_name}", value
        )
    local_digest = digest.hexdigest()
    digests = _all_gather_object(distributed_backend, local_digest)
    if len(set(digests)) != 1:
        raise RuntimeError(
            "SophiaG model/optimizer state diverged across ranks: "
            + ", ".join(digests)
        )
    return local_digest


def _step_standard_optimizer(opt, scheduler, cfg):
    opt.step()
    if scheduler is not None:
        scheduler.step()


def train(
    model,
    opt,
    datareaders,
    scheduler,
    exp_dir,
    distributed_backend,
    cfg,
):
    use_gn = cfg.opt in {"gn-prox", "gn-full"}
    active_world_size = distributed_backend.get_world_size()
    expected_world_size = getattr(
        cfg,
        "expected_world_size",
        getattr(cfg, "world_size", active_world_size),
    )
    if int(expected_world_size) != int(active_world_size):
        raise RuntimeError(
            "Configured checkpoint world size does not match the active backend: "
            f"expected={expected_world_size}, active={active_world_size}."
        )
    if use_gn and active_world_size > 1:
        raise RuntimeError(
            "Benchmark GN is currently single-rank only; its manually assigned "
            "gradients do not have verified DDP synchronization semantics."
        )
    sophia_estimator_stats = None
    if cfg.opt == "sophiag":
        estimator_semantics = copy.deepcopy(
            getattr(cfg, "training_semantics", {}).get(
                "sophia_hessian_estimator", {}
            )
        )
        estimator_mode = getattr(
            cfg, "sophia_estimator_mode", "legacy_last_microbatch"
        )
        if estimator_mode == "global_accum":
            expected_sophia_bs = (
                active_world_size * int(cfg.batch_size) * int(cfg.acc_steps)
            )
            if int(cfg.sophia_bs) != expected_sophia_bs:
                raise ValueError(
                    "global_accum SophiaG requires sophia_bs == "
                    "world_size * local_batch * local_acc_steps: "
                    f"received {cfg.sophia_bs}, expected {expected_sophia_bs}."
                )
        sophia_estimator_stats = {
            **estimator_semantics,
            "mode": estimator_mode,
            "version": estimator_semantics.get(
                "version",
                {
                    "legacy_last_microbatch": "legacy_last_microbatch_v1",
                    "global_accum": "global_accum_gnb_v1",
                }[estimator_mode],
            ),
            "refresh_count": 0,
            "actual_examples": 0,
            "actual_tokens": 0,
            "rank_state_consistent": None,
            "rank_state_digest": None,
        }

    not_compiled_model = model
    if cfg.compile:
        print(f"Compiling model ...")
        model = torch.compile(model)

    if "cuda" in cfg.device:
        type_ctx = torch.amp.autocast(
            device_type="cuda",
            dtype={
                "float32": torch.float32,
                "float16": torch.float16,
                "bfloat16": torch.bfloat16,
            }[cfg.dtype],
        )
    else:
        type_ctx = nullcontext()

    weight_averager = None
    if cfg.weight_average:
        weight_averager = WeightAverager(
            not_compiled_model,
            horizon=cfg.wa_horizon,
            interval=cfg.wa_interval,
            save_dir=None if cfg.wa_use_temp_dir else exp_dir / "avgs",
            dtype={
                "float32": torch.float32,
                "float64": torch.float64,
            }[cfg.wa_dtype],
            count=0,
        )

    ewa = None
    if cfg.exponential_weight_average:
        ewa = ExponentialWeightAverager(
            not_compiled_model,
            interval=cfg.ewa_interval,
            decay=cfg.ewa_decay,
            warmup=cfg.warmup_steps if cfg.ewa_after_warmup else 0,
            dtype={
                "float32": torch.float32,
                "float64": torch.float64,
            }[cfg.wa_dtype],
        )

    averagers = {}
    if weight_averager is not None:
        averagers["wa"] = weight_averager
    if ewa is not None:
        averagers["ewa"] = ewa

    resume_args = (
        model,
        opt,
        scheduler,
        cfg,
        averagers,
        expected_world_size,
        getattr(distributed_backend, "rank", 0),
        use_gn,
    )
    if cfg.resume_from:
        curr_iter, substep = _run_synchronized_action(
            distributed_backend,
            "checkpoint restore",
            lambda: _resume_training_state(*resume_args),
        )
    else:
        curr_iter, substep = _resume_training_state(*resume_args)

    if cfg.log_dynamics:
        def initialize_dynamics_logger():
            with open(cfg.dynamics_logger_cfg, "r") as stream:
                dlcfg = yaml.safe_load(stream)
            dlogger = DynamicsLogger(
                model, opt, dlcfg, cfg.results_base_folder, wandb=cfg.wandb
            )
            dlogger.iteration = curr_iter
            return dlogger

        _run_synchronized_action(
            distributed_backend,
            "dynamics logger initialization",
            initialize_dynamics_logger,
            master_only=True,
        )

    train_reader, val_reader = datareaders["train"], datareaders["val"]
    train_reader.set_step(substep)
    metric_semantics = copy.deepcopy(
        getattr(
            cfg,
            "metric_semantics",
            {
                "validation_loss": "next_token_cross_entropy",
                "validation_accuracy": "next_token_accuracy",
            },
        )
    )
    stats = {
        "train_loss": [],
        "val_loss": [],
        "val_pp": [],
        "val_acc": [],
        "train_records": [],
        "validation_records": [],
        "metric_semantics": metric_semantics,
    }
    last_train_loss = None
    last_iter_dt = None
    last_lr = None
    last_val_loss = None
    last_val_pp = None
    last_val_acc = None
    grad_norms = []
    wall_clock_start = time.perf_counter()
    train_step_seconds_total = 0.0
    completed_iterations = 0
    _enter_train_mode(model, opt, cfg)
    while curr_iter <= cfg.iterations:
        _save_training_checkpoints(
            model,
            opt,
            scheduler,
            curr_iter,
            exp_dir,
            distributed_backend,
            cfg,
            train_reader,
            substep,
            averagers,
        )

        ws = distributed_backend.get_world_size()
        tokens = ws * substep * cfg.sequence_length * cfg.batch_size
        epoch = tokens / train_reader.num_tokens
        if _should_run_eval(curr_iter, cfg):
            def run_evaluations():
                result = eval_and_log(
                    tokens,
                    curr_iter,
                    epoch,
                    model,
                    val_reader,
                    type_ctx,
                    distributed_backend,
                    cfg,
                    opt,
                    full_eval=_is_full_eval(curr_iter, cfg),
                    exp_dir=exp_dir,
                    manage_modes=False,
                )
                average_evaluation_counts = _run_weight_average_evals(
                    curr_iter,
                    not_compiled_model,
                    weight_averager,
                    ewa,
                    val_reader,
                    type_ctx,
                    distributed_backend,
                    cfg,
                )
                return result, average_evaluation_counts

            with _preserve_rng_state():
                try:
                    # Schedule-Free mode changes parameters in place. Every rank
                    # must execute the same eval/train round-trip even though only
                    # rank 0 computes validation metrics, otherwise finite-precision
                    # round-off can leave DDP replicas at different parameter values.
                    _run_synchronized_action(
                        distributed_backend,
                        "enter evaluation mode",
                        lambda: _enter_eval_mode(model, opt, cfg),
                    )
                    evaluation_result = _run_synchronized_action(
                        distributed_backend,
                        "evaluation",
                        run_evaluations,
                        master_only=True,
                    )
                finally:
                    _run_synchronized_action(
                        distributed_backend,
                        "restore training mode",
                        lambda: _enter_train_mode(model, opt, cfg),
                    )
            if evaluation_result is None:
                last_val_loss = last_val_pp = last_val_acc = None
                evaluation_counts = average_evaluation_counts = None
            else:
                (
                    raw_evaluation_result,
                    average_evaluation_counts,
                ) = evaluation_result
                (
                    last_val_loss,
                    last_val_pp,
                    last_val_acc,
                    evaluation_counts,
                ) = raw_evaluation_result
            if last_val_loss is not None:
                stats["val_loss"].append(float(last_val_loss))
                stats["val_pp"].append(float(last_val_pp))
                stats["val_acc"].append(float(last_val_acc))
                validation_record = {
                    "iteration": int(curr_iter),
                    "tokens": int(tokens),
                    "loss": float(last_val_loss),
                    "perplexity": float(last_val_pp),
                    "token_accuracy": float(last_val_acc),
                    **evaluation_counts,
                }
                if average_evaluation_counts:
                    validation_record["weight_average_evaluations"] = copy.deepcopy(
                        average_evaluation_counts
                    )
                stats["validation_records"].append(validation_record)
            # eval_and_log/eval_wa/eval_ewa are master-only. Keep every rank at
            # the same control-flow point before another training collective.
            _distributed_barrier(distributed_backend)

        notify_interval = getattr(cfg, "notify_interval", 0)
        if (
            curr_iter > 0
            and completed_iterations > 0
            and notify_interval
            and notify_interval > 0
            and curr_iter % notify_interval == 0
        ):
            _run_synchronized_action(
                distributed_backend,
                "training progress notification",
                lambda: _notify_training_progress(
                    curr_iter,
                    epoch,
                    exp_dir,
                    opt,
                    distributed_backend,
                    cfg,
                    last_train_loss,
                    last_val_loss,
                    last_val_pp,
                    last_val_acc,
                    last_lr,
                    last_iter_dt,
                ),
                master_only=True,
            )

        if curr_iter == cfg.iterations:
            # Save checkpoints and evaluate at final iteration, but no need to train further
            break

        # Train model
        t_start = time.perf_counter_ns()
        gn_step_size = None
        step_effective_targets = None
        if use_gn:
            raw_model = distributed_backend.get_raw_model(model)
            gn_mode = "full" if cfg.opt == "gn-full" else "prox"
            params = current_param_dict(raw_model)
            params0 = clone_param_dict(raw_model)
            gn_metrics = None
            gn_effective_targets = 0

            for inner_idx in range(cfg.gn_inner_iters):
                x, y = get_batch(train_reader, device=cfg.device)
                gn_effective_targets += _count_effective_targets(y, cfg)
                with type_ctx:
                    grads, gn_metrics = compute_gn_step(
                        model=raw_model,
                        params0=params0,
                        x=x,
                        y=y,
                        mode=gn_mode,
                        prox_weight_decay=cfg.gn_inner_wd,
                        moe=cfg.moe,
                    )

                for param, grad in zip(params.values(), grads):
                    param.grad = grad.detach()

                if cfg.grad_clip != 0.0:
                    grad_norm = _clip_grad_norm(model, cfg)
                    grad_norms.append(grad_norm)

                opt.step()
                opt.zero_grad(set_to_none=True)
                substep += 1

                if (
                    cfg.gn_log_inner_steps
                    and cfg.wandb
                    and distributed_backend.is_master_process()
                    and gn_metrics is not None
                ):
                    wandb.log(
                        {
                            "iter": curr_iter,
                            "train/gn_inner_step": curr_iter * cfg.gn_inner_iters + inner_idx,
                            "train/gn_inner_loss": gn_metrics.loss,
                            "train/gn_inner_base_loss": gn_metrics.base_loss,
                            "train/gn_inner_grad_norm": gn_metrics.gradient_norm,
                            "train/gn_inner_param_norm": gn_metrics.param_norm,
                        }
                    )

            if gn_metrics is None:
                raise RuntimeError("GN step did not produce metrics.")

            if cfg.gn_linesearch:
                current_params = clone_param_dict_from_named_params(params)
                direction = sub_param_dict(current_params, params0)
                line_search_batches = [
                    get_batch(train_reader, device=cfg.device)
                    for _ in range(cfg.gn_inner_iters)
                ]
                gn_effective_targets += sum(
                    _count_effective_targets(batch_y, cfg)
                    for _batch_x, batch_y in line_search_batches
                )
                substep += len(line_search_batches)
                gn_step_size, _ = line_search_over_direction(
                    model=raw_model,
                    anchor_params=params0,
                    direction=direction,
                    batches=line_search_batches,
                    moe=cfg.moe,
                    ls_range=cfg.gn_ls_range,
                )

            loss = torch.tensor(gn_metrics.base_loss, device=cfg.device)
            step_effective_targets = gn_effective_targets
            outputs = {
                "loss": loss,
                "aux_losses": {},
            }
            grad_norms.append(torch.tensor(gn_metrics.gradient_norm))
        else:
            (
                x,
                y,
                outputs,
                loss,
                microsteps_run,
                step_effective_targets,
                accumulation_batches,
            ) = _run_accumulation_microsteps(
                model,
                opt,
                train_reader,
                type_ctx,
                distributed_backend,
                cfg,
            )
            substep += microsteps_run

            if cfg.grad_clip != 0.0:
                grad_norm = _clip_grad_norm(model, cfg)
                grad_norms.append(grad_norm)

            if cfg.opt in {"sophiag", "mars"}:
                (
                    opt.step()
                    if cfg.opt != "sophiag"
                    else opt.step(bs=cfg.sophia_bs * cfg.sequence_length)
                )
                if cfg.scheduler != "none":
                    scheduler.step()
            else:
                _step_standard_optimizer(opt, scheduler, cfg)
            if cfg.opt == "sophiag":
                opt.zero_grad(set_to_none=True)
                if curr_iter % cfg.precondition_frequency == cfg.precondition_frequency - 1:
                    estimator_counts = _run_sophia_hessian_estimator(
                        model,
                        opt,
                        accumulation_batches,
                        distributed_backend,
                        cfg,
                    )
                    sophia_estimator_stats["refresh_count"] += 1
                    sophia_estimator_stats["actual_examples"] += (
                        estimator_counts["examples"]
                    )
                    sophia_estimator_stats["actual_tokens"] += estimator_counts[
                        "tokens"
                    ]
            elif cfg.opt == "mars":
                opt.zero_grad(set_to_none=True)
                opt.update_last_grad()
            else:
                opt.zero_grad(set_to_none=True)

        if cfg.scheduler != "none" and use_gn:
            scheduler.step()

        _step_weight_averagers(
            not_compiled_model,
            weight_averager,
            ewa,
            distributed_backend,
            cfg,
        )

        dt = (time.perf_counter_ns() - t_start) / 1e9

        curr_iter += 1
        completed_iterations += 1
        train_step_seconds_total += dt
        elapsed_seconds = time.perf_counter() - wall_clock_start
        avg_iter_dt = train_step_seconds_total / max(1, completed_iterations)
        last_iter_dt = dt
        tokens = ws * substep * cfg.sequence_length * cfg.batch_size
        epoch = tokens / train_reader.num_tokens

        train_loss_value = loss.detach().cpu().item()
        last_train_loss = float(train_loss_value)
        stats["train_loss"].append(float(train_loss_value))
        train_record = {
            "iteration": int(curr_iter),
            "tokens": int(
                ws * substep * cfg.sequence_length * cfg.batch_size
            ),
            "loss": float(train_loss_value),
            "effective_target_tokens": int(step_effective_targets),
        }
        if outputs["aux_losses"]:
            train_record["aux_losses"] = {
                name: _metric_to_float(value)
                for name, value in outputs["aux_losses"].items()
            }
        stats["train_records"].append(train_record)

        if _should_log_train_step(curr_iter, cfg, distributed_backend):
            last_train_loss, last_lr, grad_norms = _log_training_step(
                curr_iter,
                model,
                epoch,
                tokens,
                loss,
                outputs,
                opt,
                distributed_backend,
                cfg,
                dt,
                elapsed_seconds,
                train_step_seconds_total,
                avg_iter_dt,
                grad_norms,
                use_gn,
                gn_step_size,
            )
        # Only rank 0 retains the bounded window needed for periodic telemetry.
        if not distributed_backend.is_master_process() or not cfg.log_interval:
            grad_norms = []

    stats["wall_clock_seconds"] = time.perf_counter() - wall_clock_start
    stats["train_step_seconds_total"] = train_step_seconds_total
    stats["avg_iter_dt"] = (
        train_step_seconds_total / completed_iterations
        if completed_iterations
        else 0.0
    )
    stats["completed_iterations"] = completed_iterations
    if sophia_estimator_stats is not None:
        if getattr(cfg, "sophia_verify_rank_state", False):
            sophia_estimator_stats["rank_state_digest"] = (
                _sophia_rank_state_digest(model, opt, distributed_backend)
            )
            sophia_estimator_stats["rank_state_consistent"] = True
        stats["sophia_estimator"] = sophia_estimator_stats
    return stats


def _get_eval_limits(curr_iter, val_reader, cfg, full_eval):
    if curr_iter != cfg.iterations and not full_eval:
        return cfg.eval_batches, None

    available_batches = val_reader.num_batches()
    final_eval_batches = getattr(cfg, "final_eval_batches", None)
    max_num_batches = (
        available_batches
        if final_eval_batches is None
        else min(available_batches, final_eval_batches)
    )
    return max_num_batches, getattr(cfg, "final_eval_tokens", None)


def _get_eval_batch_count(curr_iter, val_reader, cfg, full_eval):
    max_num_batches, _max_num_tokens = _get_eval_limits(
        curr_iter, val_reader, cfg, full_eval
    )
    return max_num_batches


def _evaluation_count_logs(counts, *, final):
    prefix = "final-val" if final else "val"
    return {
        f"{prefix}/evaluated_batches": counts["evaluated_batches"],
        f"{prefix}/evaluated_tokens": counts["evaluated_tokens"],
    }


def _save_final_evaluation_model(
    eval_model,
    cfg,
    curr_iter,
    exp_dir,
    val_loss,
    val_perplexity,
    val_acc,
    evaluation_counts,
):
    if (
        exp_dir is None
        or curr_iter != cfg.iterations
        or not (
            _is_schedulefree(cfg)
            or getattr(cfg, "save_final_model", False)
        )
    ):
        return
    evaluation_protocol = copy.deepcopy(
        getattr(cfg, "evaluation_protocol", None)
    )
    payload = {
        "checkpoint_kind": "llm-optimizer-benchmark-evaluation-model",
        "format_version": 1,
        "iteration": int(curr_iter),
        "optimizer": cfg.opt,
        "parameterization": (
            "schedulefree-eval" if _is_schedulefree(cfg) else "train-final"
        ),
        "run_identity": copy.deepcopy(getattr(cfg, "run_identity", None)),
        "evaluation_protocol": evaluation_protocol,
        "metrics": {
            "loss": float(val_loss),
            "perplexity": float(val_perplexity),
            "token_accuracy": float(val_acc),
            **copy.deepcopy(evaluation_counts),
        },
        "model": eval_model.state_dict(),
    }
    exp_dir = Path(exp_dir)
    protocol_id = (
        evaluation_protocol.get("identity")
        if isinstance(evaluation_protocol, dict)
        else None
    )
    if not protocol_id:
        atomic_torch_save(payload, exp_dir / "model_eval.pt")
        return

    versioned_path = (
        exp_dir / "evaluations" / str(protocol_id) / "model_eval.pt"
    )
    atomic_torch_save(payload, versioned_path)

    # Keep the legacy root path as an atomic latest alias without serializing
    # the potentially large model twice. Both paths remain on the same run fs.
    latest_path = exp_dir / "model_eval.pt"
    temporary_link = exp_dir / f".model_eval.pt.{uuid.uuid4().hex}"
    try:
        os.link(versioned_path, temporary_link)
        temporary_link.replace(latest_path)
    except OSError:
        if temporary_link.exists():
            temporary_link.unlink()
        atomic_torch_save(payload, latest_path)


def _build_eval_logs(
    tokens,
    curr_iter,
    val_loss,
    val_perplexity,
    val_acc,
    val_aux_losses,
    evaluation_counts,
    cfg,
    full_eval,
):
    final = curr_iter == cfg.iterations or full_eval
    if final:
        return {
            "tokens": tokens,
            "iter": curr_iter,
            "final-val/loss": val_loss,
            "final-val/perplexity": val_perplexity,
            "final-val/acc": val_acc,
            **_evaluation_count_logs(evaluation_counts, final=True),
            **val_aux_losses,
        }
    return {
        "tokens": tokens,
        "iter": curr_iter,
        "val/loss": val_loss,
        "val/perplexity": val_perplexity,
        "val/acc": val_acc,
        **_evaluation_count_logs(evaluation_counts, final=False),
        **val_aux_losses,
    }


def _add_router_logs(logs, router_logits, cfg):
    if cfg.moe and cfg.plot_router_logits:
        routing_logs = visualize_routing(router_logits, cfg)
        logs = {**logs, **routing_logs}
    return logs


def _maybe_log_generated_text(
    curr_iter,
    val_perplexity,
    model,
    distributed_backend,
    cfg,
):
    if cfg.eval_seq_prefix != "none" and (
        curr_iter % (cfg.eval_interval * 5) == 0 or curr_iter == cfg.iterations
    ):
        text_table = wandb.Table(columns=["itr", "val-pp", "text"])

        out_str = distributed_backend.get_raw_model(model).generate_from_string(
            cfg.eval_seq_prefix,
            max_new_tokens=40,
            temperature=0.9,
            top_k=None,
        )
        text_table.add_data(curr_iter, val_perplexity, out_str)
        # why a copy? see github.com/wandb/wandb/issues/2981
        wandb.log({f"generated-text-{wandb.run.name}": copy.copy(text_table)})


def _enter_eval_mode(model, opt, cfg):
    model.eval()
    if _is_schedulefree(cfg):
        opt.eval()


def eval_and_log(
    tokens,
    curr_iter,
    epoch,
    model,
    val_reader,
    type_ctx,
    distributed_backend,
    cfg,
    opt,
    full_eval=False,
    exp_dir=None,
    manage_modes=True,
):
    if not distributed_backend.is_master_process():
        # Only evaluate and log on master rank
        return None, None, None, None

    try:
        if manage_modes:
            _enter_eval_mode(model, opt, cfg)
        max_num_batches, max_num_tokens = _get_eval_limits(
            curr_iter, val_reader, cfg, full_eval
        )

        # to make sure we start from the beginning of the validation set,
        # i.e. repeat the same batches
        val_reader.set_step(0)
        eval_model = distributed_backend.get_raw_model(model)
        (
            val_acc,
            val_loss,
            val_perplexity,
            val_aux_losses,
            router_logits,
            evaluation_counts,
        ) = eval(
            eval_model,
            val_reader,
            cfg.device,
            max_num_batches=max_num_batches,
            max_num_tokens=max_num_tokens,
            ctx=type_ctx,
            moe=cfg.moe,
            get_router_logits=cfg.moe and cfg.plot_router_logits,
            cfg=cfg,
            return_counts=True,
        )

        print(
            f">Eval: Iter={curr_iter} ({epoch:0.3f} epochs) "
            f"val_loss={val_loss:.3f} "
            f"val_pp={val_perplexity:.3f} "
            f"val_acc={val_acc:3f}"
        )

        _save_final_evaluation_model(
            eval_model,
            cfg,
            curr_iter,
            exp_dir,
            val_loss,
            val_perplexity,
            val_acc,
            evaluation_counts,
        )

        if cfg.wandb:
            logs = _build_eval_logs(
                tokens,
                curr_iter,
                val_loss,
                val_perplexity,
                val_acc,
                val_aux_losses,
                evaluation_counts,
                cfg,
                full_eval,
            )
            logs = _add_router_logs(logs, router_logits, cfg)
            wandb.log(logs)
            _maybe_log_generated_text(
                curr_iter,
                val_perplexity,
                model,
                distributed_backend,
                cfg,
            )
        return val_loss, val_perplexity, val_acc, evaluation_counts
    finally:
        if manage_modes:
            _enter_train_mode(model, opt, cfg)
