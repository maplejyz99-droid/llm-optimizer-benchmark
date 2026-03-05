import copy
import math
import time
from contextlib import nullcontext
from pathlib import Path

import torch
import wandb
import yaml

from logger.logger import DynamicsLogger
from notify import maybe_notify
from optim.weight_averaging import (ExponentialWeightAverager, WeightAverager,
                                    eval_ewa, eval_wa)

from .gn import (clone_param_dict, clone_param_dict_from_named_params,
                 compute_gn_step, current_param_dict, line_search_over_direction,
                 sub_param_dict)
from .utils import (eval, extend_onecycle_total_steps, get_batch,
                    get_parameter_norms, load_checkpoint, load_worker_state,
                    log_prodigy_lr, save_checkpoint, save_worker_state,
                    visualize_routing)


def train(
    model,
    opt,
    datareaders,
    scheduler,
    exp_dir,
    distributed_backend,
    cfg,
):
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

    if cfg.resume_from:
        # This is a full resume including the model weights, optimizer, state
        # dataloader state, random seed, etc. Not indended for fine tuning or
        # other scenarios where some of these should change.
        print(f"\nResuming Training From {cfg.resume_from}")
        ckpt_dir = Path(cfg.resume_from)
        curr_iter = load_checkpoint(
            model,
            opt,
            scheduler,
            ckpt_dir / "main.pt",
            cfg.device,
        )
        load_worker_state(ckpt_dir)
        extend_onecycle_total_steps(scheduler, cfg.iterations)
    else:
        curr_iter = 0

    if cfg.weight_average:
        # This does generally not support resuming training, but will work if
        # cfg.wa_interval perfectly divides the iteration number of the chkpt.
        # Otherwise, the first avg will not be correctly computed, with a bias
        # towards the first sample and missing values for earlier iterations.
        weight_averager = WeightAverager(
            not_compiled_model,
            horizon=cfg.wa_horizon,
            interval=cfg.wa_interval,
            save_dir=None if cfg.wa_use_temp_dir else exp_dir / "avgs",
            dtype={
                "float32": torch.float32,
                "float64": torch.float64,
            }[cfg.wa_dtype],
            count=curr_iter,
        )
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

    if distributed_backend.is_master_process() and cfg.log_dynamics:
        with open(cfg.dynamics_logger_cfg, "r") as f:
            dlcfg = yaml.safe_load(f)

        # Hooks into optimizer
        dlogger = DynamicsLogger(
            model, opt, dlcfg, cfg.results_base_folder, wandb=cfg.wandb
        )
        dlogger.iteration = curr_iter

    substep = curr_iter * cfg.acc_steps
    train_reader, val_reader = datareaders["train"], datareaders["val"]
    train_reader.set_step(substep)
    stats = {"train_loss": [], "val_loss": [], "val_pp": [], "val_acc": []}
    last_train_loss = None
    last_iter_dt = None
    last_lr = None
    last_val_loss = None
    last_val_pp = None
    last_val_acc = None
    grad_norms = []
    model.train()
    use_gn = cfg.opt in {"gn-prox", "gn-full"}

    while curr_iter <= cfg.iterations:
        # Save permanent checkpoint
        if cfg.permanent_ckpt_interval > 0:
            if curr_iter % cfg.permanent_ckpt_interval == 0:
                ckpt_dir = exp_dir / "ckpts" / str(curr_iter)
                if distributed_backend.is_master_process():
                    save_checkpoint(model, opt, scheduler, curr_iter, ckpt_dir)
                save_worker_state(ckpt_dir)

        # Save temporary checkpoint for resuming training
        if cfg.latest_ckpt_interval > 0:
            if curr_iter % cfg.latest_ckpt_interval == 0 or curr_iter == cfg.iterations:
                ckpt_dir = exp_dir / "ckpts" / "latest"
                if distributed_backend.is_master_process():
                    save_checkpoint(model, opt, scheduler, curr_iter, ckpt_dir)
                save_worker_state(ckpt_dir)

        ws = distributed_backend.get_world_size()
        tokens = ws * substep * cfg.sequence_length * cfg.batch_size
        epoch = tokens / train_reader.num_tokens
        if (
            curr_iter % cfg.eval_interval == 0
            or curr_iter == cfg.iterations
            or (curr_iter in cfg.full_eval_at)
        ):
            (
                last_val_loss,
                last_val_pp,
                last_val_acc,
            ) = eval_and_log(
                tokens,
                curr_iter,
                epoch,
                model,
                val_reader,
                type_ctx,
                distributed_backend,
                cfg,
                opt,
                full_eval=(curr_iter in cfg.full_eval_at),
            )

            if curr_iter > cfg.wa_interval and cfg.weight_average:
                eval_wa(
                    curr_iter,
                    not_compiled_model,
                    weight_averager,
                    val_reader,
                    type_ctx,
                    distributed_backend,
                    cfg,
                    full_eval=(curr_iter in cfg.full_eval_at),
                )

            if cfg.exponential_weight_average:
                eval_ewa(
                    curr_iter,
                    not_compiled_model,
                    ewa,
                    val_reader,
                    type_ctx,
                    distributed_backend,
                    cfg,
                    full_eval=(curr_iter in cfg.full_eval_at),
                )

        if curr_iter == cfg.iterations:
            # Save checkpoints and evaluate at final iteration, but no need to train further
            break

        # Train model
        t_start = time.perf_counter_ns()
        gn_step_size = None
        if use_gn:
            raw_model = distributed_backend.get_raw_model(model)
            gn_mode = "full" if cfg.opt == "gn-full" else "prox"
            params = current_param_dict(raw_model)
            params0 = clone_param_dict(raw_model)
            gn_metrics = None

            for inner_idx in range(cfg.gn_inner_iters):
                x, y = get_batch(train_reader, device=cfg.device)
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
                    if isinstance(model, torch.nn.parallel.DistributedDataParallel):
                        grad_norm = torch.nn.utils.clip_grad_norm_(
                            model.module.parameters(), cfg.grad_clip
                        )
                    else:
                        grad_norm = torch.nn.utils.clip_grad_norm_(
                            model.parameters(), cfg.grad_clip
                        )
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
            outputs = {
                "loss": loss,
                "aux_losses": {},
            }
            grad_norms.append(torch.tensor(gn_metrics.gradient_norm))
        else:
            for microstep_idx in range(cfg.acc_steps):  # gradient accumulation
                x, y = get_batch(train_reader, device=cfg.device)
                with type_ctx:
                    with distributed_backend.get_context_for_microstep_forward(
                        model=model,
                        microstep_idx=microstep_idx,
                        gradient_accumulation_steps=cfg.acc_steps,
                    ):
                        outputs = model(x, targets=y, moe=cfg.moe)

                loss = outputs["loss"] / cfg.acc_steps
                loss.backward()
                substep += 1

            if cfg.grad_clip != 0.0:
                if isinstance(model, torch.nn.parallel.DistributedDataParallel):
                    grad_norm = torch.nn.utils.clip_grad_norm_(
                        model.module.parameters(), cfg.grad_clip
                    )
                else:
                    grad_norm = torch.nn.utils.clip_grad_norm_(
                        model.parameters(), cfg.grad_clip
                    )
                grad_norms.append(grad_norm)

            if cfg.opt == "sf-sgd" or cfg.opt == "sf-adamw":
                opt.train()
            (
                opt.step()
                if cfg.opt != "sophiag"
                else opt.step(bs=cfg.sophia_bs * cfg.sequence_length)
            )
            if cfg.scheduler != "none":
                scheduler.step()
            if cfg.opt == "sophiag":
                opt.zero_grad(set_to_none=True)
                if curr_iter % cfg.precondition_frequency == cfg.precondition_frequency - 1:
                    sample_again = model(x, targets=y, get_logits=True)
                    samp_dist = torch.distributions.Categorical(
                        logits=sample_again["logits"]
                    )
                    y_sample = samp_dist.sample()
                    loss_sampled = torch.nn.functional.cross_entropy(
                        sample_again["logits"].view(-1, sample_again["logits"].size(-1)),
                        y_sample.view(-1),
                        ignore_index=-1,
                    )
                    (loss_sampled / cfg.acc_steps).backward()
                    opt.update_hessian()
                    opt.zero_grad(set_to_none=True)
                    model.zero_grad()
            elif cfg.opt == "mars":
                opt.zero_grad(set_to_none=True)
                opt.update_last_grad()
            else:
                opt.zero_grad(set_to_none=True)

        if cfg.scheduler != "none" and use_gn:
            scheduler.step()

        if cfg.weight_average:
            weight_averager.step(
                not_compiled_model, distributed_backend.is_master_process()
            )
        if cfg.exponential_weight_average:
            ewa.step(not_compiled_model, distributed_backend.is_master_process())

        dt = (time.perf_counter_ns() - t_start) / 1e9

        curr_iter += 1
        last_iter_dt = dt

        if (
            cfg.log_interval
            and curr_iter % cfg.log_interval == 0
            and distributed_backend.is_master_process()  # Only log on master rank
        ):
            train_loss = loss.detach().cpu().item() * cfg.acc_steps
            if use_gn:
                train_loss = loss.detach().cpu().item()
            last_train_loss = train_loss
            train_aux_losses = {
                f"train/{k}": v for k, v in outputs["aux_losses"].items()
            }

            current_lrs = [param_group["lr"] for param_group in opt.param_groups]
            last_lr = current_lrs[0]

            if cfg.opt == "prodigy":
                prodigy_efective_lrs = log_prodigy_lr(opt)

            print(
                f"Train: Iter={curr_iter} ({epoch:0.3f} epochs) "
                f"train_loss={train_loss:.3f} iter_dt={dt:.2e}s "
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

                wandb.log(wandb_logs)

            grad_norms = []

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

    return stats


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
):
    if not distributed_backend.is_master_process():
        # Only evaluate and log on master rank
        return None, None, None

    model.eval()
    if cfg.opt == "sf-sgd" or cfg.opt == "sf-adamw":
        opt.eval()

    if curr_iter == cfg.iterations or full_eval:
        max_num_batches = val_reader.num_batches()
    else:
        max_num_batches = cfg.eval_batches

    # to make sure we start from the beginning of the validation set,
    # i.e. repeat the same batches
    val_reader.set_step(0)
    val_acc, val_loss, val_perplexity, val_aux_losses, router_logits = eval(
        model,
        val_reader,
        cfg.device,
        max_num_batches=max_num_batches,
        ctx=type_ctx,
        moe=cfg.moe,
        get_router_logits=cfg.moe and cfg.plot_router_logits,
        cfg=cfg,
    )

    print(
        f">Eval: Iter={curr_iter} ({epoch:0.3f} epochs) "
        f"val_loss={val_loss:.3f} "
        f"val_pp={val_perplexity:.3f} "
        f"val_acc={val_acc:3f}"
    )

    if cfg.wandb:
        if curr_iter == cfg.iterations or full_eval:
            logs = {
                "tokens": tokens,
                "iter": curr_iter,
                "final-val/loss": val_loss,
                "final-val/perplexity": val_perplexity,
                "final-val/acc": val_acc,
                **val_aux_losses,
            }
        else:
            logs = {
                "tokens": tokens,
                "iter": curr_iter,
                "val/loss": val_loss,
                "val/perplexity": val_perplexity,
                "val/acc": val_acc,
                **val_aux_losses,
            }
        if cfg.moe and cfg.plot_router_logits:
            routing_logs = visualize_routing(router_logits, cfg)
            logs = {**logs, **routing_logs}

        wandb.log(logs)
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
    model.train()
    return val_loss, val_perplexity, val_acc
