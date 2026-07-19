import argparse
import copy
import inspect
import random
import sys
from pathlib import Path

import numpy as np
import torch

try:
    import wandb
except ImportError:
    wandb = None

import config
import distributed
from data.utils import DataReader, get_dataset
from models.utils import get_model
from optim.adafactor import Adafactor
from optim.ademamix import AdEMAMix
from optim.adopt import ADOPT
from optim.base import train
from optim.cadamw import CAdamW
from optim.lamb import Lamb
from optim.lion import Lion
from optim.mars import MARS
from optim.magma import MagmaAdamW, MagmaMuon
from optim.muon import CombinedScheduler, DistributedMuon, Muon
from optim.newton_muon import NewtonMuon
from optim.prodigy import Prodigy
from optim.schedule import cos_inf_schedule, wsd_schedule
from optim.schedulefree import AdamWScheduleFree, SGDScheduleFree
from optim.scion import Scion, ScionLight, scion_partitions
from optim.sign import Signum
from optim.soap import SOAP
from optim.experimental.softeq_muon import SoftEqK2000Muon
from optim.sophia import SophiaG
from run_manifest import (
    build_data_manifest,
    build_evaluation_protocol,
    build_run_manifest,
    collect_runtime_identity,
    reconcile_run_manifest,
    require_resolved_run_manifest,
    resolve_optimization_plan,
    sanitized_config,
    write_json_atomic,
)


def _run_synchronized_action(
    distributed_backend,
    action_name,
    action,
    *,
    master_only=False,
):
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
    gather = getattr(distributed_backend, "all_gather_object", None)
    if gather is None:
        if distributed_backend.get_world_size() != 1:
            raise RuntimeError(
                "Distributed backend does not provide all_gather_object()."
            )
        failures = [failure]
    else:
        failures = gather(failure)
    failures = [item for item in failures if item is not None]
    if failures:
        if local_exception is not None:
            raise local_exception
        first = failures[0]
        raise RuntimeError(
            f"{action_name} failed on rank {first['rank']}: "
            f"{first['type']}: {first['message']}"
        )
    return result


def _require_consistent_distributed_value(distributed_backend, label, value):
    gather = getattr(distributed_backend, "all_gather_object", None)
    if gather is None:
        if distributed_backend.get_world_size() != 1:
            raise RuntimeError(
                "Distributed backend does not provide all_gather_object()."
            )
        values = [value]
    else:
        values = gather(value)
    if any(candidate != values[0] for candidate in values[1:]):
        raise RuntimeError(f"{label} differs across distributed ranks.")
    return values[0]


MUON_SCHEDULER_OPTS = {"muon", "muon-magma", "newton-muon", "softeq-k2000-muon"}
MAGMA_OPT_LABELS = {
    "adamw-magma": "AdamW Magma",
    "muon-magma": "Muon Magma",
}


def get_mup_width_mult(args):
    return args.n_embd / args.scale_base_model


def get_muon_effective_backup_lr(group):
    return group.get("adamw_lr_ratio", 1.0) * group["lr"]


def get_optimizer_param_list(args, model):
    if args.opt == "newton-muon":
        return list(model.parameters())
    return (
        list(model.parameters())
        if args.distributed_backend is None
        else list(model.module.parameters())
    )


def build_adamw_optimizer(args, group_specs, lr, betas, weight_decay, fused_label):
    device_type = "cuda" if "cuda" in args.device else "cpu"
    use_fused = (device_type == "cuda") and (
        "fused" in inspect.signature(torch.optim.AdamW).parameters
    )
    print(f"using fused {fused_label}: {use_fused}")
    extra_args = dict(fused=True) if use_fused else dict()
    return torch.optim.AdamW(
        group_specs,
        lr=lr,
        betas=betas,
        weight_decay=weight_decay,
        **extra_args,
    )


def build_optimizer(args, model, group_specs, magma_param_ids):
    if args.opt in MAGMA_OPT_LABELS and getattr(args, "world_size", 1) > 1:
        raise ValueError(
            f"{MAGMA_OPT_LABELS[args.opt]} requires world_size=1 because its "
            "stochastic masks are not synchronized across ranks."
        )
    if args.opt == "adamw":
        return build_adamw_optimizer(
            args,
            group_specs,
            lr=args.lr,
            betas=(args.beta1, args.beta2),
            weight_decay=args.weight_decay,
            fused_label="AdamW",
        )
    elif args.opt == "gn-prox" or args.opt == "gn-full":
        return build_adamw_optimizer(
            args,
            group_specs,
            lr=args.gn_inner_lr,
            betas=(args.gn_inner_b1, args.gn_inner_b2),
            # Proximal regularization is applied explicitly in the GN objective.
            weight_decay=0.0,
            fused_label="GN inner AdamW",
        )
    elif args.opt == "cadamw":
        return CAdamW(
            group_specs,
            lr=args.lr,
            betas=(args.beta1, args.beta2),
            weight_decay=args.weight_decay,
            eps=1e-8,
            cautious_xi=args.cautious_xi,
        )
    elif args.opt == "soap":
        return SOAP(
            group_specs,
            lr=args.lr,
            betas=(args.beta1, args.beta2),
            shampoo_beta=args.shampoo_beta,
            weight_decay=args.weight_decay,
            precondition_frequency=args.precondition_frequency,
            max_precond_dim=args.max_precond_dim,
            merge_dims=args.merge_dims,
            precondition_1d=args.precondition_1d,
            normalize_grads=args.normalize_grads,
            data_format=args.soap_data_format,
            correct_bias=args.correct_bias,
        )
    elif args.opt == "muon":
        param_list = get_optimizer_param_list(args, model)
        muon_lr = args.muon_lr_factor
        if args.model == "mup_llama":
            muon_lr = args.muon_lr_factor / get_mup_width_mult(args)
            print(
                "muP Llama Muon mode: scaling Muon matrix lr "
                f"from {args.muon_lr_factor} to {muon_lr}. "
                "AdamW backup lr remains args.lr; this is an engineering "
                "training policy, not a theoretical muP-Muon proof."
            )
        return Muon(
            muon_params=param_list,
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
    elif args.opt == "softeq-k2000-muon":
        param_list = get_optimizer_param_list(args, model)
        muon_lr = args.muon_lr_factor
        if args.model == "mup_llama":
            muon_lr = args.muon_lr_factor / get_mup_width_mult(args)
            print(
                "muP Llama SoftEq K=2000 Muon mode: scaling matrix lr "
                f"from {args.muon_lr_factor} to {muon_lr}. "
                "AdamW backup lr remains args.lr; this is an engineering "
                "training policy, not a theoretical muP-Muon proof."
            )
        return SoftEqK2000Muon(
            muon_params=param_list,
            lr=muon_lr,
            momentum=args.momentum,
            weight_decay=args.weight_decay,
            adamw_params=None,
            adamw_lr=args.lr,
            adamw_betas=(args.beta1, args.beta2),
            adamw_eps=1e-8,
            adamw_wd=args.weight_decay,
        )
    elif args.opt == "newton-muon":
        param_list = get_optimizer_param_list(args, model)
        opt = NewtonMuon(
            muon_params=param_list,
            lr=args.muon_lr_factor,
            momentum=args.momentum,
            nesterov=args.nesterov,
            ns_steps=args.muon_ns_steps,
            adamw_params=None,
            adamw_lr=args.lr,
            adamw_betas=(args.beta1, args.beta2),
            adamw_eps=1e-8,
            adamw_wd=args.weight_decay,
            precond_every=args.newton_muon_precond_every,
            precond_ewma=args.newton_muon_precond_ewma,
            precond_init_diag=args.newton_muon_precond_init_diag,
            precond_ridge_mult=args.newton_muon_precond_ridge_mult,
            precond_eps=args.newton_muon_precond_eps,
        )
        opt.attach_preconditioner(model)
        return opt
    elif args.opt == "muon-magma":
        param_list = get_optimizer_param_list(args, model)
        return MagmaMuon(
            muon_params=param_list,
            lr=args.muon_lr_factor,
            momentum=args.momentum,
            nesterov=args.nesterov,
            ns_steps=args.muon_ns_steps,
            adamw_params=None,
            adamw_lr=args.lr,
            adamw_betas=(args.beta1, args.beta2),
            adamw_eps=1e-8,
            adamw_wd=args.weight_decay,
            magma_survival_p=args.magma_survival_p,
            magma_tau=args.magma_tau,
            magma_beta=args.magma_beta,
            magma_param_ids=magma_param_ids,
        )
    elif args.opt == "adamw-magma":
        return MagmaAdamW(
            group_specs,
            lr=args.lr,
            betas=(args.beta1, args.beta2),
            weight_decay=args.weight_decay,
            eps=1e-8,
            magma_survival_p=args.magma_survival_p,
            magma_tau=args.magma_tau,
            magma_beta=args.magma_beta,
            magma_param_ids=magma_param_ids,
        )
    elif args.opt == "d-muon":
        return DistributedMuon(
            group_specs,
            lr=args.lr,
            momentum=args.momentum,
            nesterov=args.nesterov,
            ns_steps=args.muon_ns_steps,
            adamw_betas=(args.beta1, args.beta2),
            adamw_eps=1e-8,
            weight_decay=args.weight_decay,
        )
    elif args.opt == "ademamix":
        return AdEMAMix(
            group_specs,
            lr=args.lr,
            betas=(args.beta1, args.beta2, args.adema_beta3),
            alpha=args.adema_alpha,
            beta3_warmup=args.adema_beta3_warmup,
            alpha_warmup=args.adema_alpha_warmup,
            weight_decay=args.weight_decay,
        )
    elif args.opt == "lion":
        return Lion(
            group_specs,
            lr=args.lr,
            betas=(args.beta1, args.beta2),
            weight_decay=args.weight_decay,
        )
    elif args.opt == "sf-adamw":
        return AdamWScheduleFree(
            group_specs,
            lr=args.lr,
            betas=(args.beta1, args.beta2),
            weight_decay=args.weight_decay,
            warmup_steps=args.warmup_steps,
            r=args.schedulefree_r,
            weight_lr_power=args.weight_lr_power,
        )  # without foreach argument
    elif args.opt == "sf-sgd":
        return SGDScheduleFree(
            group_specs,
            lr=args.lr,
            momentum=args.momentum,
            weight_decay=args.weight_decay,
            warmup_steps=args.warmup_steps,
            r=args.schedulefree_r,
            weight_lr_power=args.weight_lr_power,
        )  # without foreach argument
    elif args.opt == "signsgd":
        return Signum(
            group_specs,
            lr=args.lr,
            momentum=0.0,  # always use zero momentum because its signSGD
            dampening=args.dampening,
            weight_decay=args.weight_decay,
            nesterov=args.nesterov,
            sign_update=True,
        )
    elif args.opt == "signum":
        return Signum(
            group_specs,
            lr=args.lr,
            momentum=args.momentum,
            weight_decay=args.weight_decay,
            dampening=args.dampening,
            nesterov=args.nesterov,
            sign_update=True,
        )
    elif args.opt == "prodigy":
        return Prodigy(
            group_specs,
            lr=args.lr,
            betas=(args.beta1, args.beta2),
            beta3=args.prodigy_beta3,
            weight_decay=args.weight_decay,
            decouple=args.prodigy_decouple,
            use_bias_correction=args.prodigy_use_bias_correction,
            safeguard_warmup=args.prodigy_safeguard_warmup,
            fsdp_in_use=args.prodigy_fsdp_in_use,
        )
    elif args.opt == "sophiag":
        return SophiaG(
            group_specs,
            lr=args.lr,
            betas=(args.beta1, args.beta2),
            weight_decay=args.weight_decay,
            rho=args.sophia_rho,
        )
    elif args.opt == "adopt":
        return ADOPT(
            group_specs,
            lr=args.lr,
            betas=(args.beta1, args.beta2),
            eps=args.adopt_eps,  # 1e-6
            weight_decay=args.weight_decay,
            decouple=args.adopt_decouple,
        )
    elif args.opt == "mars":
        return MARS(
            group_specs,
            lr=args.mars_lr,
            betas=(args.mars_beta1, args.mars_beta2),
            weight_decay=args.weight_decay,
            amsgrad=False,
            gamma=args.mars_vr_gamma,
            is_approx=args.mars_is_approx,
            mars_type=args.mars_type,
            optimize_1d=False,  # we set in order to optimize 1D parameters with AdamW
            lr_1d=args.lr,  # AdamW's lr when optimize_1d=False
            betas_1d=(args.beta1, args.beta2),  # AdamW's betas when optimize_1d=False
            weight_decay_1d=args.weight_decay,  # AdamW's weight decay
        )
    elif args.opt == "adafactor":
        return Adafactor(
            group_specs,
            lr=args.lr,
            decay_rate=args.adafactor_decay_rate,
            beta1=args.beta1,
            clip_threshold=1.0,
            weight_decay=args.weight_decay,
        )
    elif args.opt == "lamb":
        return Lamb(
            group_specs,
            lr=args.lr,
            betas=(args.beta1, args.beta2),
            weight_decay=args.weight_decay,
            adam=False,
            bias_correction=args.lamb_use_bias_correction,
        )
    elif args.opt == "scion":
        scion_param_groups = scion_partitions(group_specs, model, args)
        scion_params_cnt = sum(
            p.numel() for group in scion_param_groups for p in group["params"]
        )
        print(f"Optimized parameters: {scion_params_cnt}")
        return Scion(
            scion_param_groups,
            lr=args.lr,
            momentum=args.momentum,
        )
    elif args.opt == "scion-light":
        scion_param_groups = scion_partitions(group_specs, model, args)
        scion_params_cnt = sum(
            p.numel() for group in scion_param_groups for p in group["params"]
        )
        print(f"Optimized parameters: {scion_params_cnt}")
        return ScionLight(
            scion_param_groups,
            lr=args.lr,
            momentum=args.momentum,
        )
    elif args.opt == "muon-pytorch":
        return torch.optim.Muon(
            group_specs,
            lr=args.lr,
            momentum=args.momentum,
            nesterov=args.nesterov,
            ns_steps=args.muon_ns_steps,
            ns_coefficients=(
                3.4445,
                -4.775,
                2.0315,
            ),  # someone might try to change it later
            eps=1e-7,  # muon pytorch uses smaller eps
            adjust_lr_fn=None,  # to make the orthogonalized update have a consistent RMS across rectangular matrices
        )
    elif args.opt == "sgd":
        return torch.optim.SGD(
            group_specs,
            lr=args.lr,
            momentum=args.momentum,
            weight_decay=args.weight_decay,
            nesterov=args.nesterov,
        )
    raise ValueError(f"Unknown optimizer: {args.opt!r}")


def uses_combined_scheduler(args):
    return args.opt in MUON_SCHEDULER_OPTS


def build_scheduler(args, opt, group_specs):
    if args.opt == "adafactor":
        # Adafactor is configured with relative_step=True, so its learning-rate
        # schedule is owned by the optimizer and no external scheduler state
        # should be created or checkpointed.
        return None
    if args.scheduler == "none":
        return None

    assert (
        args.warmup_steps < args.iterations
    ), "Warmup steps must be < iterations."  # from schedules-and-scaling
    sched_base_lr = args.gn_inner_lr if args.opt in {"gn-prox", "gn-full"} else args.lr
    if args.scheduler in ["cos", "linear"]:
        # initial lr is args.lr / div_factor
        # final lr is initial_lr/final_div_factor = args.lr / div_factor / final_div_factor
        return (
            torch.optim.lr_scheduler.OneCycleLR(
                optimizer=opt,
                max_lr=[
                    group.get("lr", sched_base_lr) for group in group_specs
                ],  # it was args.lr
                total_steps=args.iterations,
                pct_start=args.warmup_steps
                / args.iterations,  # it was args.warmup_percent
                anneal_strategy=args.scheduler,
                cycle_momentum=False,
                div_factor=1e2,
                final_div_factor=args.final_div_factor,
            )
            if not uses_combined_scheduler(args)
            else CombinedScheduler(opt, args)
        )
    elif args.scheduler == "cos_inf":
        lambda_schedule = cos_inf_schedule(
            n_iterations=args.iterations,
            n_warmup=args.warmup_steps,
            n_inf=args.cos_inf_steps,
            div_factor=1e2,
            final_div_factor=0.1,
        )
        return (
            torch.optim.lr_scheduler.LambdaLR(opt, lambda_schedule)
            if not uses_combined_scheduler(args)
            else CombinedScheduler(opt, args)
        )
    elif args.scheduler == "wsd":
        lambda_schedule = wsd_schedule(
            n_iterations=args.iterations,
            n_warmup=args.warmup_steps,
            fract_decay=args.wsd_fract_decay,
            init_div_factor=1e2,
            final_lr_factor=args.wsd_final_lr_scale,  # should be 0 here
            decay_type=args.decay_type,
        )
        return (
            torch.optim.lr_scheduler.LambdaLR(opt, lambda_schedule)
            if not uses_combined_scheduler(args)
            else CombinedScheduler(opt, args)
        )
    else:
        raise NotImplementedError(f"Unknown scheduler type: {args.scheduler}.")


def log_optimizer_groups(opt, param_to_name, label):
    print(f"\nOptimizer parameter groups ({label}):")
    for group_idx, group in enumerate(opt.param_groups):
        lr = group.get("lr")
        weight_decay = group.get("weight_decay", group.get("adamw_wd"))
        adamw_lr = (
            get_muon_effective_backup_lr(group)
            if "adamw_lr_ratio" in group and lr is not None
            else group.get("adamw_lr")
        )
        print(
            f"  group {group_idx}: lr={lr}, "
            f"adamw_backup_lr={adamw_lr}, weight_decay={weight_decay}, "
            f"num_params={len(group['params'])}"
        )
        for param in group["params"]:
            name = param_to_name.get(id(param), "<unnamed>")
            branch = "default"
            if param in opt.state and "use_muon" in opt.state[param]:
                branch = "muon" if opt.state[param]["use_muon"] else "adamw_backup"
            effective_lr = adamw_lr if branch == "adamw_backup" and adamw_lr is not None else lr
            print(
                f"    {branch}: {name} shape={tuple(param.shape)} "
                f"lr={effective_lr} weight_decay={weight_decay}"
            )


def get_logged_parameter_counts(raw_model):
    params_cnt = raw_model.get_num_params()
    try:
        nonemb_param_cnt = raw_model.get_num_params(non_embedding=True)
    except TypeError:
        nonemb_param_cnt = params_cnt
    if nonemb_param_cnt < 0:
        raise ValueError(f"Non-embedding parameter count is negative: {nonemb_param_cnt}.")
    return params_cnt, nonemb_param_cnt


def get_args():
    parser = argparse.ArgumentParser(allow_abbrev=False)
    parser.add_argument(
        "--config_format", default="base", choices=config.registered_formats()
    )

    args, rem_args = parser.parse_known_args()

    final_args = config.parse_args_with_format(
        format=args.config_format, base_parser=parser, args=rem_args, namespace=args
    )

    return final_args, parser


def main(args, parser):
    if args.opt == "newton-muon" and args.distributed_backend is not None:
        raise ValueError("Newton-Muon v1 only supports single-device dense Llama.")
    if args.opt in MAGMA_OPT_LABELS and args.distributed_backend is not None:
        raise ValueError(
            f"{MAGMA_OPT_LABELS[args.opt]} currently only supports single-device "
            "execution because its stochastic masks are not synchronized across ranks."
        )
    if args.opt in {"gn-prox", "gn-full"} and args.distributed_backend is not None:
        raise ValueError(
            "Benchmark GN currently only supports single-device execution; "
            "multi-rank gradients are not synchronized."
        )
    args.run_seed = args.seed
    distributed_backend = distributed.make_backend_from_args(args)
    try:
        result = _main_with_backend(args, parser, distributed_backend)
    except BaseException as primary_error:
        try:
            distributed_backend.finalize()
        except BaseException as finalize_error:
            raise primary_error from finalize_error
        raise
    else:
        distributed_backend.finalize()
        return result


def _main_with_backend(args, parser, distributed_backend):
    args = distributed_backend.get_adjusted_args_for_process(args)
    args.world_size = distributed_backend.get_world_size()
    if args.opt in MAGMA_OPT_LABELS and args.world_size > 1:
        raise ValueError(
            f"{MAGMA_OPT_LABELS[args.opt]} requires world_size=1 because its "
            "stochastic masks are not synchronized across ranks."
        )
    if args.opt == "newton-muon":
        if args.world_size != 1:
            raise ValueError("Newton-Muon v1 only supports world_size=1.")
        if args.moe:
            raise ValueError("Newton-Muon v1 does not support MoE models.")
        if args.model != "llama":
            raise ValueError("Newton-Muon v1 only supports --model llama.")
    if args.opt == "softeq-k2000-muon":
        if args.moe:
            raise ValueError("SoftEq K=2000 Muon v1 does not support MoE models.")
        if args.model not in {"llama", "mup_llama"}:
            raise ValueError(
                "SoftEq K=2000 Muon v1 only supports --model llama or mup_llama."
            )
    if args.opt in {"sf-adamw", "sf-sgd"} and args.scheduler != "none":
        raise ValueError("Schedule-free optimizers require --scheduler none.")
    if args.wandb and wandb is None:
        raise ImportError("wandb is not installed; rerun without --wandb or install wandb.")

    if args.full_eval_at is None:
        args.full_eval_at = []

    # NOTE args.seed is offset per worker in get_adjusted_args_for_process
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    torch.manual_seed(args.seed)
    random.seed(args.seed)
    np.random.seed(args.seed)
    if "cuda" in args.device:
        torch.cuda.set_device(torch.device(args.device))
    # torch.use_deterministic_algorithms(True)  # CUBLAS_WORKSPACE_CONFIG=:4096:8

    exp_name = get_exp_name(args, parser, distributed_backend)
    exp_dir = Path(args.results_base_folder) / exp_name
    manifest_path = exp_dir / "run_manifest.json"
    latest_ckpt_dir = exp_dir / "ckpts" / "latest"

    print(f"Starting Experiment: {exp_name}")
    print(f"Experiment Directory: {exp_dir}")

    completed_summary_exists = _require_consistent_distributed_value(
        distributed_backend,
        "Completed summary availability",
        (exp_dir / "summary.json").is_file(),
    )
    versioned_summary_exists = _require_consistent_distributed_value(
        distributed_backend,
        "Versioned evaluation summary availability",
        any(
            path.is_file()
            for path in (exp_dir / "evaluations").glob("*/summary.json")
        ),
    )
    if completed_summary_exists or versioned_summary_exists:
        raise ValueError(
            f"The experiment dir {exp_dir} already has a completed summary "
            "artifact and cannot be resumed or overwritten."
        )
    latest_checkpoint_exists = _require_consistent_distributed_value(
        distributed_backend,
        "Latest checkpoint availability",
        (latest_ckpt_dir / "main.pt").is_file(),
    )
    if latest_checkpoint_exists and args.resume_from is None:
        if not args.auto_resume:
            raise ValueError(
                f"The experiment dir {exp_dir} already has a checkpoint. "
                "To resume training, set auto_resume=True. Otherwise, "
                "specify a different experiment name."
            )
        args.resume_from = str(latest_ckpt_dir)
    resume_source = _require_consistent_distributed_value(
        distributed_backend,
        "Resume source",
        args.resume_from,
    )
    if resume_source is not None:
        expected_resume_path = latest_ckpt_dir.resolve(strict=False)
        actual_resume_path = Path(resume_source).expanduser().resolve(strict=False)
        if actual_resume_path != expected_resume_path:
            raise ValueError(
                "--resume_from may only point to this run's ckpts/latest "
                f"directory: expected {expected_resume_path}, got "
                f"{actual_resume_path}."
            )
        if not latest_checkpoint_exists:
            raise ValueError(
                "--resume_from requires this run's ckpts/latest/main.pt."
            )
        existing_manifest = _run_synchronized_action(
            distributed_backend,
            "resume manifest validation",
            lambda: require_resolved_run_manifest(manifest_path),
        )
        current_protocol = _run_synchronized_action(
            distributed_backend,
            "requested evaluation protocol construction",
            lambda: build_evaluation_protocol(args),
        )
        existing_protocol = existing_manifest.get("evaluation_protocol", {})
        existing_protocol_identity = _require_consistent_distributed_value(
            distributed_backend,
            "Existing evaluation protocol identity",
            existing_protocol.get("identity"),
        )
        current_protocol_identity = _require_consistent_distributed_value(
            distributed_backend,
            "Requested evaluation protocol identity",
            current_protocol["identity"],
        )
        if existing_protocol_identity != current_protocol_identity:
            raise ValueError(
                "Existing resolved evaluation protocol identity does not match "
                "the requested evaluation settings; resume with the original "
                "settings or choose a different experiment directory."
            )

    print(f"Loading dataset: '{args.dataset}'")
    datareaders = _run_synchronized_action(
        distributed_backend,
        "data reader initialization",
        lambda: get_data_readers(args),
    )
    data_manifest = datareaders.get("_data_manifest")
    if data_manifest is None:
        data_manifest = build_data_manifest(args.dataset, {})
    runtime_identity = collect_runtime_identity()
    if hasattr(torch, "__version__"):
        runtime_identity["torch_runtime"] = {
            "version": str(torch.__version__),
            "cuda": str(getattr(getattr(torch, "version", None), "cuda", None)),
        }
    preflight_manifest = build_run_manifest(
        args,
        data_manifest,
        runtime=runtime_identity,
    )
    _require_consistent_distributed_value(
        distributed_backend,
        "Preflight identity",
        preflight_manifest["preflight_identity"],
    )
    args.metric_semantics = preflight_manifest["metric_semantics"]
    args.evaluation_protocol = preflight_manifest["evaluation_protocol"]
    args.training_semantics = preflight_manifest["training_semantics"]

    _run_synchronized_action(
        distributed_backend,
        "run manifest preflight reconciliation",
        lambda: reconcile_run_manifest(
            manifest_path,
            preflight_manifest,
            phase="preflight",
        ),
        master_only=True,
    )
    distributed_backend.barrier()

    model = _run_synchronized_action(
        distributed_backend,
        "model construction",
        lambda: get_model(args).to(args.device),
    )
    if args.opt in {"gn-prox", "gn-full"}:
        # GN uses torch.func JVP; force math SDP backend to avoid Flash forward-AD errors.
        if "cuda" in args.device:
            if hasattr(torch.backends.cuda, "enable_flash_sdp"):
                torch.backends.cuda.enable_flash_sdp(False)
            if hasattr(torch.backends.cuda, "enable_mem_efficient_sdp"):
                torch.backends.cuda.enable_mem_efficient_sdp(False)
            if hasattr(torch.backends.cuda, "enable_math_sdp"):
                torch.backends.cuda.enable_math_sdp(True)
        print("GN mode: force math SDP backend for forward-AD compatibility.")
    print(f"\nModel:\n{model}")

    model = _run_synchronized_action(
        distributed_backend,
        "distributed model transformation",
        lambda: distributed_backend.transform_model(model),
    )

    def prepare_parameter_groups():
        raw_model = distributed_backend.get_raw_model(model)
        group_specs = raw_model.get_parameter_group_specs(config=args)
        param_name_mapping = {
            parameter_name: parameter
            for parameter_name, parameter in model.named_parameters()
        }
        param_to_name = {}
        magma_param_ids = set()

        def use_magma_for_param(parameter_name):
            if args.magma_scope == "all":
                return True
            lowered = parameter_name.lower()
            return ("attn" in lowered) or ("mlp" in lowered)

        optimized_params_cnt = 0
        for group in group_specs:
            parameters = []
            for parameter_name in group["params"]:
                translated_names = (
                    distributed_backend.translate_model_parameter_name_for_node(
                        parameter_name
                    )
                )
                for translated_name in translated_names:
                    parameter = param_name_mapping[translated_name]
                    parameters.append(parameter)
                    param_to_name[id(parameter)] = translated_name
                    if use_magma_for_param(translated_name):
                        magma_param_ids.add(id(parameter))
            group["params"] = parameters
            optimized_params_cnt += sum(
                parameter.numel() for parameter in parameters
            )
        params_cnt, nonemb_param_cnt = get_logged_parameter_counts(raw_model)
        return (
            group_specs,
            param_to_name,
            magma_param_ids,
            optimized_params_cnt,
            raw_model,
            params_cnt,
            nonemb_param_cnt,
        )

    (
        group_specs,
        param_to_name,
        magma_param_ids,
        optimized_params_cnt,
        raw_model,
        params_cnt,
        nonemb_param_cnt,
    ) = _run_synchronized_action(
        distributed_backend,
        "optimizer parameter-group assembly",
        prepare_parameter_groups,
    )
    print("number of parameters: %.2fM" % (params_cnt / 1e6,))
    print("number of optimized parameters: %.2fM" % (optimized_params_cnt / 1e6,))
    print("number of non-embedding parameters: %.2fM" % (nonemb_param_cnt / 1e6,))

    args.world_size = distributed_backend.get_world_size()

    opt = _run_synchronized_action(
        distributed_backend,
        "optimizer construction",
        lambda: build_optimizer(args, model, group_specs, magma_param_ids),
    )
    print(f"\nOptimizer:\n{opt}")
    if args.log_optimizer_groups:
        log_optimizer_groups(opt, param_to_name, "before scheduler")
    if "magma" in args.opt:
        print(
            "Magma targets: "
            f"{len(magma_param_ids)} tensors "
            f"(scope={args.magma_scope}, p={args.magma_survival_p}, "
            f"tau={args.magma_tau}, beta={args.magma_beta})"
        )

    resolved_plan = _run_synchronized_action(
        distributed_backend,
        "optimization plan resolution",
        lambda: resolve_optimization_plan(
            preflight_manifest["optimization_plan"],
            opt,
            raw_model.named_parameters(),
            overlay_param_ids={"magma_modifier": magma_param_ids}
            if "magma" in args.opt
            else None,
        ),
    )
    run_manifest = build_run_manifest(
        args,
        data_manifest,
        code=preflight_manifest["code"],
        runtime=preflight_manifest["runtime"],
        optimization_plan=resolved_plan,
    )
    _require_consistent_distributed_value(
        distributed_backend,
        "Resolved run identity",
        run_manifest["run_identity"],
    )
    _run_synchronized_action(
        distributed_backend,
        "resolved run manifest reconciliation",
        lambda: reconcile_run_manifest(
            manifest_path,
            run_manifest,
            phase="resolved",
        ),
        master_only=True,
    )
    distributed_backend.barrier()

    args.run_identity = run_manifest["run_identity"]
    args.metric_semantics = run_manifest["metric_semantics"]
    args.evaluation_protocol = run_manifest["evaluation_protocol"]
    args.training_semantics = run_manifest["training_semantics"]

    scheduler = _run_synchronized_action(
        distributed_backend,
        "scheduler construction",
        lambda: build_scheduler(args, opt, group_specs),
    )
    if args.log_optimizer_groups:
        log_optimizer_groups(opt, param_to_name, "after scheduler init")

    public_config = sanitized_config(args)
    print(f"Config:\n{public_config}\n")
    if args.wandb:
        def initialize_wandb():
            wandb.init(
                project=args.wandb_project,
                name=exp_name,
                config=public_config,
                entity=args.wandb_entity,
            )
            wandb.define_metric("iter")
            wandb.define_metric("train/*", step_metric="iter")
            wandb.define_metric("val/*", step_metric="iter")
            wandb.define_metric("lr", step_metric="iter")
            wandb.log(
                {
                    "parameters": params_cnt,
                    "optimized_parameters": optimized_params_cnt,
                    "non_embedding_parameters": nonemb_param_cnt,
                }
            )

        _run_synchronized_action(
            distributed_backend,
            "W&B initialization",
            initialize_wandb,
            master_only=True,
        )

    stats = train(
        model=model,
        opt=opt,
        datareaders=datareaders,
        scheduler=scheduler,
        exp_dir=exp_dir,
        distributed_backend=distributed_backend,
        cfg=args,
    )

    stats["args"] = public_config
    stats["run_identity"] = args.run_identity
    stats["preflight_identity"] = run_manifest["preflight_identity"]
    stats["optimization_plan"] = run_manifest["optimization_plan"]
    stats["metric_semantics"] = args.metric_semantics
    stats["evaluation_protocol"] = args.evaluation_protocol
    stats["training_semantics"] = args.training_semantics

    evaluation_summary_path = (
        exp_dir
        / "evaluations"
        / args.evaluation_protocol["identity"]
        / "summary.json"
    )

    def write_summaries():
        # Preserve each evaluation protocol independently, then update the
        # conventional latest-summary path used by existing tooling.
        write_json_atomic(evaluation_summary_path, stats)
        write_json_atomic(exp_dir / "summary.json", stats)

    _run_synchronized_action(
        distributed_backend,
        "summary write",
        write_summaries,
        master_only=True,
    )


def get_data_readers(args, verbose=True):
    data_srcs = get_dataset(args)
    data_manifest = build_data_manifest(args.dataset, data_srcs)
    train_reader = DataReader(
        data_src=data_srcs["train"],
        batch_size=args.batch_size,
        sequence_length=args.sequence_length,
        seed=args.data_seed,
        with_replacement=False,
        auto_shard=True,
        keep_in_ram=args.data_in_ram,
    )
    val_reader = DataReader(
        data_src=data_srcs["val"],
        batch_size=args.batch_size,
        sequence_length=args.sequence_length,
        seed=args.data_seed,
        with_replacement=False,
        auto_shard=False,  # NOTE Identical Per Rank
        keep_in_ram=args.data_in_ram,
    )

    if verbose:
        print(f"Num training tokens: {train_reader.num_tokens}")
        print(f"Num validation tokens: {val_reader.num_tokens}")

    return {
        "train": train_reader,
        "val": val_reader,
        "_data_manifest": data_manifest,
    }


def get_exp_name(
    args,
    parser,
    distributed_backend,
    key_args=["model", "dataset", "opt"],
    ignore_args=[
        "eval_interval",
        "eval_batches",
        "eval_seq_prefix",
        "final_eval_batches",
        "final_eval_tokens",
        "full_eval_at",
        "distributed_backend",
        "distributed_control_timeout_seconds",
        "latest_ckpt_interval",
        "permanent_ckpt_interval",
        "datasets_dir",
        "wandb",
        "wandb_project",
        "wandb_entity",
        "batch_size",
        "acc_steps",
        "results_base_folder",
        "run_prefix",
        "wandb_run_prefix",
        "seed",
        "device",
        "adema_beta3_warmup",
        "adema_alpha_warmup",
        "plot_router_logits",
        "weight_average",
        # "wa_interval",
        # "wa_horizon",
        "wa_dtype",
        "wa_use_temp_dir",
        "wa_sweep_horizon",
        # "max_num_wa_sweeps",
        "exponential_weight_average",
        # "ewa_interval",
        # "ewa_decay",
        # "ewa_after_warmup",
        "moe",
        "log_interval",
        "log_parameter_norms",
        "log_dynamics",
        "log_optimizer_groups",
        "dynamics_logger_cfg",
        "experiment_name",
        "resume_from",
        "allow_legacy_checkpoint_resume",
        "run_identity",
        "run_seed",
        "metric_semantics",
        "evaluation_protocol",
        "training_semantics",
    ],
):
    # Set the custom exp name if needed
    if args.experiment_name is not None:
        return args.experiment_name

    # Get the default values
    defaults = vars(parser.parse_args([]))

    # rank = distributed_backend.rank # decided to remove rank from the exp name

    # Generate the prefix with key arguments
    prefix_parts = []
    for key in key_args:
        if hasattr(args, key):
            value = getattr(args, key)
            if key == "model":
                if getattr(args, "moe", False):
                    value = f"moe_{value}"
                if getattr(args, "weight_average", False):
                    value = f"{value}_WA"
                if getattr(args, "exponential_weight_average", False):
                    value = f"{value}_EWA"
            prefix_parts.append(f"{key}-{value}")

    prefix = "_".join(prefix_parts)
    prefix = f"{args.batch_size}x{args.acc_steps}_" + prefix  # rank={rank}

    # Generate the rest of the string with non-default arguments
    non_default_parts = []
    for key, value in vars(args).items():
        if key in ignore_args or key.startswith("notify_"):
            continue
        if key not in defaults:
            print(f"Warning: {key} not in defaults")
            continue
        if key not in key_args and value != defaults[key]:
            non_default_parts.append(f"{key}-{value}")

    non_default_string = "_".join(non_default_parts)

    if args.run_prefix is not None:
        prefix = args.run_prefix + "_" + prefix

    # Combine prefix and non-default string
    if non_default_string:
        return f"{prefix}__{non_default_string}"
    else:
        return prefix


if __name__ == "__main__":
    args, parser = get_args()
    main(args, parser)
