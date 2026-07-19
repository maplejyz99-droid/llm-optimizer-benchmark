"""Credential-safe run and data artifact identity helpers."""

import fcntl
import hashlib
import importlib.metadata
import json
import math
import os
import platform
import subprocess
import sys
import tempfile
from datetime import datetime, timezone
from pathlib import Path


RUN_MANIFEST_SCHEMA_VERSION = 3
OPTIMIZATION_PLAN_SCHEMA_VERSION = 1
DATA_MANIFEST_SCHEMA_VERSION = 1
DEFAULT_FULL_HASH_LIMIT = 64 * 1024 * 1024
DEFAULT_SAMPLE_CHUNK_SIZE = 1024 * 1024

EPFL_BENCHMARK_TASKS = {
    "arc_easy",
    "arc_challenge",
    "hellaswag",
    "logiqa",
    "piqa",
    "sciq",
    "humaneval",
    "gsm8k",
    "kodcode",
    "mathqa",
    "medqa",
}

COMPATIBILITY_EXCLUDED_CONFIG = {
    "auto_resume",
    "allow_legacy_checkpoint_resume",
    "datasets_dir",
    "device",
    "dynamics_logger_cfg",
    "eval_batches",
    "eval_interval",
    "eval_seq_prefix",
    "final_eval_batches",
    "final_eval_tokens",
    "experiment_name",
    "full_eval_at",
    "iterations",
    "latest_ckpt_interval",
    "log_dynamics",
    "log_interval",
    "log_optimizer_groups",
    "log_parameter_norms",
    "norm_order",
    "notify_interval",
    "permanent_ckpt_interval",
    "results_base_folder",
    "resume_from",
    "resume_from_swa",
    "run_prefix",
    "run_identity",
    "save_final_model",
    "sophia_verify_rank_state",
    "distributed_control_timeout_seconds",
    "evaluation_protocol",
    "metric_semantics",
    "optimization_plan",
    "training_semantics",
    "wandb",
    "wandb_entity",
    "wandb_project",
    "wandb_run_prefix",
}

# The parser intentionally exposes every optimizer through one namespace. These
# fields stay in the redacted audit config, but compatibility is defined by the
# active optimization plan instead of defaults belonging to inactive branches.
OPTIMIZATION_CONFIG_KEYS = {
    "opt",
    "lr",
    "weight_decay",
    "beta1",
    "beta2",
    "momentum",
    "nesterov",
    "grad_clip",
    "scheduler",
    "warmup_steps",
    "final_div_factor",
    "cos_inf_steps",
    "wsd_final_lr_scale",
    "wsd_fract_decay",
    "decay_type",
    "shampoo_beta",
    "precondition_frequency",
    "max_precond_dim",
    "merge_dims",
    "precondition_1d",
    "normalize_grads",
    "soap_data_format",
    "correct_bias",
    "muon_ns_steps",
    "muon_lr_factor",
    "newton_muon_precond_every",
    "newton_muon_precond_ewma",
    "newton_muon_precond_init_diag",
    "newton_muon_precond_ridge_mult",
    "newton_muon_precond_eps",
    "cautious_xi",
    "magma_survival_p",
    "magma_tau",
    "magma_beta",
    "magma_scope",
    "adema_beta3",
    "adema_alpha",
    "adema_beta3_warmup",
    "adema_alpha_warmup",
    "schedulefree_r",
    "weight_lr_power",
    "dampening",
    "prodigy_beta3",
    "prodigy_decouple",
    "prodigy_use_bias_correction",
    "prodigy_safeguard_warmup",
    "prodigy_fsdp_in_use",
    "sophia_rho",
    "sophia_bs",
    "sophia_estimator_mode",
    "mars_type",
    "mars_vr_gamma",
    "mars_is_approx",
    "mars_lr",
    "mars_beta1",
    "mars_beta2",
    "adafactor_decay_rate",
    "lamb_use_bias_correction",
    "adopt_decouple",
    "adopt_eps",
    "scion_lmh_scale",
    "scion_emb_scale",
    "scion_tr_scale",
    "gn_inner_iters",
    "gn_inner_lr",
    "gn_inner_b1",
    "gn_inner_b2",
    "gn_inner_wd",
    "gn_linesearch",
    "gn_ls_range",
    "gn_log_inner_steps",
    "clipping_type",
    "clip_eta",
}

MUON_SPLIT_OPTIMIZERS = {
    "muon",
    "newton-muon",
    "muon-magma",
    "softeq-k2000-muon",
    "d-muon",
}

COMBINED_SCHEDULER_OPTIMIZERS = {
    "muon",
    "newton-muon",
    "muon-magma",
    "softeq-k2000-muon",
}

SENSITIVE_CONFIG_KEYS = {
    "notify_email_from",
    "notify_email_to",
    "notify_pushplus_token",
    "notify_pushplus_topic",
    "notify_smtp_host",
    "notify_smtp_pass",
    "notify_smtp_port",
    "notify_smtp_user",
    "notify_webhook",
}

SAFE_NOTIFY_CONFIG_KEYS = {
    "notify_interval",
    "notify_method",
}

RUNTIME_PACKAGES = (
    "torch",
    "numpy",
    "pyyaml",
    "tiktoken",
    "datasets",
    "huggingface-hub",
    "transformers",
    "wandb",
)


def _canonical_bytes(value):
    return json.dumps(
        _normalize(value),
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
    ).encode("utf-8")


def _normalize(value):
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, dict):
        return {str(key): _normalize(item) for key, item in sorted(value.items())}
    if isinstance(value, (list, tuple)):
        return [_normalize(item) for item in value]
    if isinstance(value, set):
        return sorted(_normalize(item) for item in value)
    if hasattr(value, "item"):
        try:
            return _normalize(value.item())
        except (TypeError, ValueError):
            pass
    return repr(value)


def _sha256_bytes(content):
    return hashlib.sha256(content).hexdigest()


def _full_file_sha256(path):
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _sampled_file_sha256(path, chunk_size):
    size = path.stat().st_size
    offsets = sorted(
        {
            0,
            max(0, size // 2 - chunk_size // 2),
            max(0, size - chunk_size),
        }
    )
    digest = hashlib.sha256()
    digest.update(f"sampled-sha256-v1:{size}:{chunk_size}".encode("ascii"))
    with path.open("rb") as stream:
        for offset in offsets:
            stream.seek(offset)
            content = stream.read(chunk_size)
            digest.update(offset.to_bytes(8, "big"))
            digest.update(len(content).to_bytes(8, "big"))
            digest.update(content)
    return {
        "method": "sampled-sha256-v1",
        "sha256": digest.hexdigest(),
        "chunk_size": chunk_size,
        "sample_count": len(offsets),
        "offsets": offsets,
    }


def fingerprint_file(
    path,
    *,
    full_hash_limit=DEFAULT_FULL_HASH_LIMIT,
    sample_chunk_size=DEFAULT_SAMPLE_CHUNK_SIZE,
):
    path = Path(path)
    size = path.stat().st_size
    if size <= full_hash_limit:
        return {"method": "sha256", "sha256": _full_file_sha256(path)}
    return _sampled_file_sha256(path, sample_chunk_size)


def _artifact_path(value):
    if isinstance(value, (str, os.PathLike)):
        return Path(value)
    filename = getattr(value, "filename", None)
    return Path(filename) if filename is not None else None


def _array_byte_view(value):
    """Expose contiguous array bytes without duplicating a large token array."""
    try:
        view = memoryview(value)
        if not view.c_contiguous:
            raise TypeError("non-contiguous buffer")
        return view.cast("B")
    except (TypeError, ValueError):
        if not hasattr(value, "tobytes"):
            raise TypeError(
                f"Unsupported in-memory data artifact: {type(value)!r}"
            )
        return memoryview(value.tobytes(order="C"))


def describe_data_artifact(
    value,
    *,
    full_hash_limit=DEFAULT_FULL_HASH_LIMIT,
    sample_chunk_size=DEFAULT_SAMPLE_CHUNK_SIZE,
):
    path = _artifact_path(value)
    dtype = str(getattr(value, "dtype", "uint16" if path and path.suffix == ".bin" else "unknown"))
    shape = list(getattr(value, "shape", ()))

    if path is not None:
        path = path.expanduser().resolve()
        stat_result = path.stat()
        artifact = {
            "path": str(path),
            "byte_size": stat_result.st_size,
            "mtime_ns": stat_result.st_mtime_ns,
            "dtype": dtype,
            "shape": shape,
            "fingerprint": fingerprint_file(
                path,
                full_hash_limit=full_hash_limit,
                sample_chunk_size=sample_chunk_size,
            ),
        }
        if dtype == "uint16" and stat_result.st_size % 2 == 0:
            artifact["token_count"] = stat_result.st_size // 2
        return artifact

    content = _array_byte_view(value)
    byte_size = len(content)
    if byte_size > full_hash_limit:
        chunk_size = min(sample_chunk_size, byte_size)
        offsets = sorted(
            {0, max(0, byte_size // 2 - chunk_size // 2), max(0, byte_size - chunk_size)}
        )
        digest = hashlib.sha256()
        digest.update(
            f"sampled-sha256-v1:{byte_size}:{chunk_size}".encode("ascii")
        )
        for offset in offsets:
            sample = content[offset : offset + chunk_size]
            digest.update(offset.to_bytes(8, "big"))
            digest.update(len(sample).to_bytes(8, "big"))
            digest.update(sample)
        fingerprint = {
            "method": "sampled-sha256-v1",
            "sha256": digest.hexdigest(),
            "chunk_size": chunk_size,
            "sample_count": len(offsets),
            "offsets": offsets,
        }
    else:
        fingerprint = {"method": "sha256", "sha256": _sha256_bytes(content)}
    artifact = {
        "path": None,
        "byte_size": byte_size,
        "dtype": dtype,
        "shape": shape,
        "fingerprint": fingerprint,
    }
    if dtype == "uint16" and byte_size % 2 == 0:
        artifact["token_count"] = byte_size // 2
    return artifact


def _artifact_identity(artifact):
    return {
        key: artifact[key]
        for key in ("byte_size", "dtype", "shape", "token_count", "fingerprint")
        if key in artifact
    }


def build_data_manifest(
    dataset,
    data_sources,
    *,
    full_hash_limit=DEFAULT_FULL_HASH_LIMIT,
    sample_chunk_size=DEFAULT_SAMPLE_CHUNK_SIZE,
):
    artifacts = {
        name: describe_data_artifact(
            value,
            full_hash_limit=full_hash_limit,
            sample_chunk_size=sample_chunk_size,
        )
        for name, value in sorted(data_sources.items())
        if name in {"train", "val", "train_len", "val_len"}
    }
    semantics_id = (
        "epfl-flat-eotpad-v1"
        if dataset in EPFL_BENCHMARK_TASKS
        else "flat-next-token-v1"
    )
    identity_payload = {
        "dataset": dataset,
        "semantics_id": semantics_id,
        "tokenizer": "gpt2",
        "artifacts": {
            name: _artifact_identity(artifact)
            for name, artifact in artifacts.items()
        },
    }
    return {
        "schema_version": DATA_MANIFEST_SCHEMA_VERSION,
        **identity_payload,
        "artifacts": artifacts,
        "identity": _sha256_bytes(_canonical_bytes(identity_payload)),
    }


def _git_command(repo_root, arguments):
    result = subprocess.run(
        ["git", *arguments],
        cwd=repo_root,
        check=True,
        capture_output=True,
    )
    return result.stdout


def collect_code_fingerprint(repo_root=None):
    repo_root = Path(repo_root or Path(__file__).resolve().parents[1])
    try:
        root = Path(
            _git_command(repo_root, ["rev-parse", "--show-toplevel"])
            .decode("utf-8")
            .strip()
        )
        head = _git_command(root, ["rev-parse", "HEAD"]).decode("ascii").strip()
        tracked_diff = _git_command(
            root,
            ["diff", "--binary", "--no-ext-diff", "HEAD", "--", "src", "scripts"],
        )
        raw_untracked = _git_command(
            root,
            ["ls-files", "--others", "--exclude-standard", "-z", "--", "src", "scripts"],
        )
    except (OSError, subprocess.CalledProcessError, UnicodeDecodeError) as exc:
        return {
            "available": False,
            "error": f"{type(exc).__name__}: {exc}",
            "source_fingerprint_sha256": None,
        }

    untracked = []
    for raw_name in raw_untracked.split(b"\0"):
        if not raw_name:
            continue
        relative = Path(os.fsdecode(raw_name))
        path = root / relative
        if not path.is_file():
            continue
        untracked.append(
            {
                "path": relative.as_posix(),
                "sha256": _full_file_sha256(path),
                "byte_size": path.stat().st_size,
            }
        )

    source_payload = {
        "head": head,
        "tracked_diff_sha256": _sha256_bytes(tracked_diff),
        "untracked_sources": untracked,
    }
    return {
        "available": True,
        "repo_root": str(root),
        **source_payload,
        "source_fingerprint_sha256": _sha256_bytes(_canonical_bytes(source_payload)),
    }


def collect_runtime_identity():
    packages = {}
    for name in RUNTIME_PACKAGES:
        try:
            packages[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            packages[name] = None
    return {
        "python": {
            "implementation": platform.python_implementation(),
            "version": platform.python_version(),
            "executable": sys.executable,
        },
        "platform": {
            "system": platform.system(),
            "release": platform.release(),
            "machine": platform.machine(),
        },
        "packages": packages,
    }


def sanitized_config(args):
    config = {}
    for key, value in sorted(vars(args).items()):
        is_notification_secret = (
            key.startswith("notify_") and key not in SAFE_NOTIFY_CONFIG_KEYS
        )
        if key in SENSITIVE_CONFIG_KEYS or is_notification_secret:
            config[key] = None if value is None else "<redacted>"
        else:
            config[key] = _normalize(value)
    return config


def _normalized_device_type(device):
    if device is None:
        return None
    return str(device).strip().lower().split(":", 1)[0]


def compatibility_config(args):
    config = {
        key: value
        for key, value in sanitized_config(args).items()
        if key not in COMPATIBILITY_EXCLUDED_CONFIG
        and key not in OPTIMIZATION_CONFIG_KEYS
        and key not in SENSITIVE_CONFIG_KEYS
        and not key.startswith("notify_")
    }
    if "run_seed" in config:
        config["seed"] = config.pop("run_seed")
    if hasattr(args, "device"):
        config["device_type"] = _normalized_device_type(args.device)
    return config


_MISSING = object()


def _arg(args, name, default=_MISSING):
    if hasattr(args, name):
        return getattr(args, name)
    return default


def _put_arg(target, key, args, name=None, *, transform=None):
    value = _arg(args, name or key)
    if value is _MISSING:
        return
    target[key] = transform(value) if transform is not None else value


def _put_betas(target, args, first="beta1", second="beta2"):
    beta1 = _arg(args, first)
    beta2 = _arg(args, second)
    if beta1 is not _MISSING and beta2 is not _MISSING:
        target["betas"] = [beta1, beta2]


def _component(component_id, role, algorithm, hyperparameters):
    return {
        "id": component_id,
        "role": role,
        "algorithm": algorithm,
        "hyperparameters": _normalize(hyperparameters),
    }


def _optimization_intent_payload(plan):
    try:
        routing = plan["routing"]
        return {
            "schema_version": plan["schema_version"],
            "strategy": plan["strategy"],
            "components": plan["components"],
            "scheduler": plan["scheduler"],
            "gradient_processing": plan["gradient_processing"],
            "routing": {
                "update_routes": routing["update_routes"],
                "overlays": routing["overlays"],
            },
        }
    except (KeyError, TypeError) as exc:
        raise ValueError("Optimization plan is missing required intent fields.") from exc


def _optimization_intent_identity(plan):
    return _sha256_bytes(_canonical_bytes(_optimization_intent_payload(plan)))


def _optimization_realization_payload(realization):
    if not isinstance(realization, dict):
        raise ValueError("Optimization realization must be an object.")
    return {
        key: value
        for key, value in realization.items()
        if key not in {"status", "identity"}
    }


def _optimization_realization_identity(realization):
    return _sha256_bytes(
        _canonical_bytes(_optimization_realization_payload(realization))
    )


def validate_optimization_plan(plan, *, expected_status=None):
    if not isinstance(plan, dict):
        raise ValueError("Optimization plan must be an object.")
    if plan.get("schema_version") != OPTIMIZATION_PLAN_SCHEMA_VERSION:
        raise ValueError("Unsupported optimization plan schema.")
    expected_intent_identity = _optimization_intent_identity(plan)
    if plan.get("identity") != expected_intent_identity:
        raise ValueError(
            "Optimization plan identity does not match its canonical intent."
        )
    try:
        realization = plan["routing"]["realization"]
    except (KeyError, TypeError) as exc:
        raise ValueError("Optimization plan has no routing realization.") from exc
    status = realization.get("status") if isinstance(realization, dict) else None
    if status not in {"pending", "resolved"}:
        raise ValueError(f"Unknown optimization realization status: {status!r}.")
    if expected_status is not None and status != expected_status:
        raise ValueError(
            f"Optimization realization must be {expected_status!r}, got {status!r}."
        )
    if status == "resolved":
        expected_realization_identity = _optimization_realization_identity(realization)
        if realization.get("identity") != expected_realization_identity:
            raise ValueError(
                "Optimization realization identity does not match its "
                "canonical content."
            )
    elif "identity" in realization:
        raise ValueError("Pending optimization realization cannot have an identity.")
    return plan


def _adaptive_hyperparameters(
    args,
    *,
    lr="lr",
    beta1="beta1",
    beta2="beta2",
    weight_decay="weight_decay",
    eps=None,
):
    values = {}
    _put_arg(values, "lr", args, lr)
    _put_betas(values, args, beta1, beta2)
    _put_arg(values, "weight_decay", args, weight_decay)
    if eps is not None:
        values["eps"] = eps
    return values


def _sgd_hyperparameters(args, *, lr="lr", weight_decay="weight_decay"):
    values = {}
    _put_arg(values, "lr", args, lr)
    _put_arg(values, "momentum", args)
    _put_arg(values, "weight_decay", args, weight_decay)
    _put_arg(values, "nesterov", args)
    return values


def _effective_muon_lr(args):
    configured = _arg(args, "muon_lr_factor")
    if configured is _MISSING:
        return _MISSING
    if _arg(args, "model", "llama") != "mup_llama":
        return configured
    width = _arg(args, "n_embd")
    base_width = _arg(args, "scale_base_model")
    if width is _MISSING or base_width in {_MISSING, 0}:
        return configured
    return configured / (width / base_width)


def _muon_components(
    args,
    *,
    algorithm="muon",
    effective_lr=None,
    matrix_weight_decay=False,
    nesterov=True,
    ns_steps=True,
):
    matrix = {}
    configured_lr = _arg(args, "muon_lr_factor")
    if configured_lr is not _MISSING:
        matrix["configured_lr"] = configured_lr
    resolved_lr = (
        _effective_muon_lr(args) if effective_lr is None else effective_lr
    )
    if resolved_lr is not _MISSING:
        matrix["effective_lr"] = resolved_lr
    _put_arg(matrix, "momentum", args)
    if nesterov:
        _put_arg(matrix, "nesterov", args)
    if ns_steps:
        _put_arg(matrix, "ns_steps", args, "muon_ns_steps")
    if matrix_weight_decay:
        _put_arg(matrix, "weight_decay", args)

    backup = _adaptive_hyperparameters(args, eps=1e-8)
    return [
        _component("matrix_update", "primary", algorithm, matrix),
        _component("fallback_update", "fallback", "adamw", backup),
    ]


def _scheduler_plan(args, opt):
    scheduler = _arg(args, "scheduler", "cos")
    if opt == "adafactor":
        plan = {
            "kind": "internal",
            "algorithm": "adafactor_relative_step",
        }
        iterations = _arg(args, "iterations")
        if iterations is not _MISSING:
            plan["horizon_steps"] = iterations
        return _normalize(plan)
    plan = {"kind": "none" if scheduler == "none" else "external"}
    if scheduler == "none":
        plan["algorithm"] = "none"
    elif scheduler in {"cos", "linear"}:
        plan.update(
            {
                "algorithm": (
                    "combined_one_cycle"
                    if opt in COMBINED_SCHEDULER_OPTIMIZERS
                    else "one_cycle"
                ),
                "anneal_strategy": scheduler,
                "initial_div_factor": 100.0,
                "final_div_factor": (
                    1.0
                    if opt in COMBINED_SCHEDULER_OPTIMIZERS
                    else _arg(args, "final_div_factor", 1.0)
                ),
            }
        )
    elif scheduler == "cos_inf":
        plan.update(
            {
                "algorithm": (
                    "combined_cos_inf"
                    if opt in COMBINED_SCHEDULER_OPTIMIZERS
                    else "cos_inf"
                ),
                "initial_div_factor": 100.0,
                "final_div_factor": 0.1,
                "infinite_tail_steps": _arg(args, "cos_inf_steps", 0),
            }
        )
    elif scheduler == "wsd":
        plan.update(
            {
                "algorithm": (
                    "combined_wsd"
                    if opt in COMBINED_SCHEDULER_OPTIMIZERS
                    else "wsd"
                ),
                "initial_div_factor": 100.0,
                "final_lr_scale": _arg(args, "wsd_final_lr_scale", 0.0),
                "decay_fraction": _arg(args, "wsd_fract_decay", 0.1),
                "decay_type": _arg(args, "decay_type", "linear"),
            }
        )
    else:
        raise ValueError(f"Unknown scheduler: {scheduler!r}")

    iterations = _arg(args, "iterations")
    if iterations is not _MISSING:
        plan["horizon_steps"] = iterations
    if scheduler != "none":
        warmup = _arg(args, "warmup_steps")
        if warmup is not _MISSING:
            plan["warmup_steps"] = warmup
    return _normalize(plan)


def _all_parameters_route(component):
    return {
        "component": component,
        "selector": {
            "kind": "all_optimizer_parameters",
            "version": "all_optimizer_parameters_v1",
        },
    }


def _muon_split_routes():
    return [
        {
            "component": "matrix_update",
            "selector": {
                "kind": "optimizer_state_flag",
                "field": "use_muon",
                "equals": True,
                "version": "use_muon_state_v1",
            },
        },
        {
            "component": "fallback_update",
            "selector": {
                "kind": "optimizer_state_flag",
                "field": "use_muon",
                "equals": False,
                "version": "use_muon_state_v1",
            },
        },
    ]


def _magma_component(args):
    values = {}
    _put_arg(values, "survival_probability", args, "magma_survival_p")
    _put_arg(values, "temperature", args, "magma_tau")
    _put_arg(values, "ema_beta", args, "magma_beta")
    _put_arg(values, "scope", args, "magma_scope")
    return _component("magma_modifier", "modifier", "magma", values)


def _sophia_estimator_component(args):
    semantics = build_training_semantics(args).get("sophia_hessian_estimator", {})
    return _component(
        "hessian_estimator",
        "auxiliary",
        "gauss_newton_bartlett_diagonal",
        semantics,
    )


def build_optimization_plan(args):
    """Build the active, composable optimization intent from the selected CLI branch."""
    opt = _arg(args, "opt", "adamw")
    strategy = {"id": f"{opt}_v1", "kind": "single"}
    components = []
    update_routes = []
    overlays = []

    if opt == "adamw":
        components = [
            _component(
                "primary_update",
                "primary",
                "adamw",
                _adaptive_hyperparameters(args, eps=1e-8),
            )
        ]
        update_routes = [_all_parameters_route("primary_update")]
    elif opt in {"gn-prox", "gn-full"}:
        strategy["kind"] = "nested"
        outer = {
            "variant": opt.removeprefix("gn-"),
        }
        _put_arg(outer, "inner_iterations", args, "gn_inner_iters")
        _put_arg(outer, "proximal_weight_decay", args, "gn_inner_wd")
        _put_arg(outer, "line_search", args, "gn_linesearch")
        if _arg(args, "gn_linesearch", False):
            _put_arg(outer, "line_search_range", args, "gn_ls_range")
        inner = _adaptive_hyperparameters(
            args,
            lr="gn_inner_lr",
            beta1="gn_inner_b1",
            beta2="gn_inner_b2",
            weight_decay="__fixed_zero_weight_decay",
            eps=1e-8,
        )
        inner["weight_decay"] = 0.0
        base_lr = _arg(args, "lr")
        if (
            base_lr is not _MISSING
            and _arg(args, "model", "llama") in {"mup_gpt", "mup_llama"}
        ):
            inner["parameter_group_base_lr"] = base_lr
        components = [
            _component("outer_method", "outer", "gauss_newton", outer),
            _component("inner_update", "inner", "adamw", inner),
        ]
        update_routes = [_all_parameters_route("inner_update")]
    elif opt == "cadamw":
        values = _adaptive_hyperparameters(args, eps=1e-8)
        _put_arg(values, "cautious_xi", args)
        components = [
            _component("primary_update", "primary", "cautious_adamw", values)
        ]
        update_routes = [_all_parameters_route("primary_update")]
    elif opt == "adamw-magma":
        strategy["kind"] = "overlay"
        components = [
            _component(
                "primary_update",
                "primary",
                "adamw",
                _adaptive_hyperparameters(args, eps=1e-8),
            ),
            _magma_component(args),
        ]
        update_routes = [_all_parameters_route("primary_update")]
        overlays = [
            {
                "component": "magma_modifier",
                "selector": {
                    "kind": "magma_scope",
                    "scope": _arg(args, "magma_scope", "all"),
                    "version": "magma_name_scope_v1",
                },
            }
        ]
    elif opt == "sgd":
        components = [
            _component("primary_update", "primary", "sgd", _sgd_hyperparameters(args))
        ]
        update_routes = [_all_parameters_route("primary_update")]
    elif opt in MUON_SPLIT_OPTIMIZERS:
        strategy["kind"] = "parameter_split"
        if opt == "muon":
            components = _muon_components(args)
        elif opt == "newton-muon":
            strategy["kind"] = "parameter_split_with_auxiliary"
            components = _muon_components(args, algorithm="newton_muon")
            preconditioner = {}
            for output, source in (
                ("refresh_frequency", "newton_muon_precond_every"),
                ("ewma", "newton_muon_precond_ewma"),
                ("initial_diagonal", "newton_muon_precond_init_diag"),
                ("ridge_multiplier", "newton_muon_precond_ridge_mult"),
                ("epsilon", "newton_muon_precond_eps"),
            ):
                _put_arg(preconditioner, output, args, source)
            components.append(
                _component(
                    "newton_preconditioner",
                    "auxiliary",
                    "activation_covariance_newton",
                    preconditioner,
                )
            )
            overlays = [
                {
                    "component": "newton_preconditioner",
                    "selector": {
                        "kind": "optimizer_preconditioner_map",
                        "version": "newton_preconditioner_map_v1",
                    },
                }
            ]
        elif opt == "muon-magma":
            strategy["kind"] = "parameter_split_with_overlay"
            # This branch intentionally does not apply muP width scaling.
            components = _muon_components(
                args,
                effective_lr=_arg(args, "muon_lr_factor"),
            )
            components.append(_magma_component(args))
            overlays = [
                {
                    "component": "magma_modifier",
                    "selector": {
                        "kind": "magma_scope",
                        "scope": _arg(args, "magma_scope", "all"),
                        "version": "magma_name_scope_v1",
                    },
                }
            ]
        elif opt == "softeq-k2000-muon":
            components = _muon_components(
                args,
                algorithm="softeq_k2000_muon",
                matrix_weight_decay=True,
                nesterov=False,
                ns_steps=False,
            )
            components[0]["hyperparameters"].update(
                {
                    "phase_steps": 2000,
                    "soft_equivalence_alpha": 0.5,
                    "orthogonalization_steps": 12,
                }
            )
        else:  # d-muon
            components = _muon_components(
                args,
                algorithm="distributed_muon",
                effective_lr=_arg(args, "lr"),
                matrix_weight_decay=True,
            )
            components[0]["hyperparameters"].pop("configured_lr", None)
            components[0]["hyperparameters"]["matched_adamw_rms"] = 0.2
            fallback = components[1]["hyperparameters"]
            fallback["lr"] = _arg(args, "lr", fallback.get("lr"))
        update_routes = _muon_split_routes()
    elif opt == "soap":
        values = _adaptive_hyperparameters(args)
        configured_shampoo_beta = _arg(args, "shampoo_beta", -1.0)
        beta2 = _arg(args, "beta2")
        if configured_shampoo_beta >= 0:
            values["shampoo_beta"] = configured_shampoo_beta
        elif beta2 is not _MISSING:
            values["shampoo_beta"] = beta2
        for output, source in (
            ("precondition_frequency", "precondition_frequency"),
            ("max_precondition_dimension", "max_precond_dim"),
            ("merge_dimensions", "merge_dims"),
            ("precondition_1d", "precondition_1d"),
            ("normalize_gradients", "normalize_grads"),
            ("correct_bias", "correct_bias"),
        ):
            _put_arg(values, output, args, source)
        components = [_component("primary_update", "primary", "soap", values)]
        update_routes = [_all_parameters_route("primary_update")]
    elif opt == "ademamix":
        values = _adaptive_hyperparameters(args)
        for output, source in (
            ("beta3", "adema_beta3"),
            ("alpha", "adema_alpha"),
            ("beta3_warmup_steps", "adema_beta3_warmup"),
            ("alpha_warmup_steps", "adema_alpha_warmup"),
        ):
            _put_arg(values, output, args, source)
        components = [_component("primary_update", "primary", "ademamix", values)]
        update_routes = [_all_parameters_route("primary_update")]
    elif opt == "lion":
        components = [
            _component(
                "primary_update",
                "primary",
                "lion",
                _adaptive_hyperparameters(args),
            )
        ]
        update_routes = [_all_parameters_route("primary_update")]
    elif opt in {"sf-adamw", "sf-sgd"}:
        values = (
            _adaptive_hyperparameters(args)
            if opt == "sf-adamw"
            else _sgd_hyperparameters(args)
        )
        if opt == "sf-sgd":
            # SGDScheduleFree does not accept or consume the shared CLI flag.
            values.pop("nesterov", None)
        _put_arg(values, "warmup_steps", args)
        _put_arg(values, "r", args, "schedulefree_r")
        _put_arg(values, "weight_lr_power", args)
        values["parameter_point_version"] = "schedule_free_parameter_point_v2"
        components = [
            _component(
                "primary_update",
                "primary",
                "schedule_free_adamw" if opt == "sf-adamw" else "schedule_free_sgd",
                values,
            )
        ]
        update_routes = [_all_parameters_route("primary_update")]
    elif opt in {"signsgd", "signum"}:
        values = {}
        _put_arg(values, "lr", args)
        momentum = 0.0 if opt == "signsgd" else _arg(args, "momentum", 0.0)
        values["momentum"] = momentum
        _put_arg(values, "weight_decay", args)
        if opt == "signum" and momentum != 0:
            _put_arg(values, "dampening", args)
            _put_arg(values, "nesterov", args)
        values["sign_update"] = True
        components = [_component("primary_update", "primary", opt, values)]
        update_routes = [_all_parameters_route("primary_update")]
    elif opt == "prodigy":
        values = _adaptive_hyperparameters(args, eps=1e-8)
        configured_beta3 = _arg(args, "prodigy_beta3", None)
        beta2 = _arg(args, "beta2")
        if configured_beta3 is not None:
            values["beta3"] = configured_beta3
        elif beta2 is not _MISSING:
            values["beta3"] = math.sqrt(beta2)
        for output, source in (
            ("decoupled_weight_decay", "prodigy_decouple"),
            ("bias_correction", "prodigy_use_bias_correction"),
            ("safeguard_warmup", "prodigy_safeguard_warmup"),
            ("fsdp_in_use", "prodigy_fsdp_in_use"),
        ):
            _put_arg(values, output, args, source)
        components = [_component("primary_update", "primary", "prodigy", values)]
        update_routes = [_all_parameters_route("primary_update")]
    elif opt == "sophiag":
        strategy["kind"] = "single_with_auxiliary"
        values = _adaptive_hyperparameters(args)
        _put_arg(values, "rho", args, "sophia_rho")
        _put_arg(values, "batch_scale_examples", args, "sophia_bs")
        components = [
            _component("primary_update", "primary", "sophiag", values),
            _sophia_estimator_component(args),
        ]
        update_routes = [_all_parameters_route("primary_update")]
        overlays = [
            {
                "component": "hessian_estimator",
                "selector": {
                    "kind": "all_optimizer_parameters",
                    "version": "sophia_gnb_auxiliary_v1",
                },
            }
        ]
    elif opt == "adopt":
        values = _adaptive_hyperparameters(args)
        _put_arg(values, "eps", args, "adopt_eps")
        _put_arg(values, "decoupled_weight_decay", args, "adopt_decouple")
        components = [_component("primary_update", "primary", "adopt", values)]
        update_routes = [_all_parameters_route("primary_update")]
    elif opt == "mars":
        strategy["kind"] = "parameter_split"
        variant = _arg(args, "mars_type", "mars-adamw")
        mars = {}
        _put_arg(mars, "lr", args, "mars_lr")
        if variant == "mars-adamw":
            _put_betas(mars, args, "mars_beta1", "mars_beta2")
            mars["eps"] = 1e-8
        else:
            _put_arg(mars, "beta1", args, "mars_beta1")
        _put_arg(mars, "weight_decay", args)
        for output, source in (
            ("variant", "mars_type"),
            ("variance_reduction_gamma", "mars_vr_gamma"),
            ("approximate", "mars_is_approx"),
        ):
            _put_arg(mars, output, args, source)
        components = [
            _component("matrix_update", "primary", "mars", mars),
            _component(
                "fallback_update",
                "fallback",
                "adamw",
                _adaptive_hyperparameters(args, eps=1e-8),
            ),
        ]
        update_routes = [
            {
                "component": "matrix_update",
                "selector": {
                    "kind": "parameter_ndim",
                    "equals": 2,
                    "version": "mars_rank2_v1",
                },
            },
            {
                "component": "fallback_update",
                "selector": {
                    "kind": "parameter_ndim",
                    "not_equals": 2,
                    "version": "mars_rank2_v1",
                },
            },
        ]
    elif opt == "adafactor":
        values = {}
        _put_arg(values, "beta1", args)
        _put_arg(values, "weight_decay", args)
        _put_arg(values, "decay_rate", args, "adafactor_decay_rate")
        values.update(
            {
                "clip_threshold": 1.0,
                "scale_parameter": True,
                "relative_step": True,
                "warmup_init": False,
            }
        )
        components = [_component("primary_update", "primary", "adafactor", values)]
        update_routes = [_all_parameters_route("primary_update")]
    elif opt == "lamb":
        values = _adaptive_hyperparameters(args, eps=1e-6)
        _put_arg(values, "bias_correction", args, "lamb_use_bias_correction")
        values["adam_mode"] = False
        components = [_component("primary_update", "primary", "lamb", values)]
        update_routes = [_all_parameters_route("primary_update")]
    elif opt in {"scion", "scion-light"}:
        strategy["kind"] = "parameter_groups"
        values = {}
        _put_arg(values, "lr", args)
        if opt == "scion":
            _put_arg(values, "momentum", args)
        scale_fields = [
            ("lm_head_scale", "scion_lmh_scale"),
            ("transformer_scale", "scion_tr_scale"),
        ]
        if _arg(args, "untied_embeds", False):
            scale_fields.append(("embedding_scale", "scion_emb_scale"))
        for output, source in scale_fields:
            _put_arg(values, output, args, source)
        components = [_component("primary_update", "primary", opt, values)]
        update_routes = [
            {
                "component": "primary_update",
                "selector": {
                    "kind": "scion_partition_policy",
                    "version": "scion_partition_policy_v1",
                },
            }
        ]
    elif opt == "muon-pytorch":
        values = {}
        _put_arg(values, "lr", args)
        _put_arg(values, "momentum", args)
        _put_arg(values, "nesterov", args)
        _put_arg(values, "ns_steps", args, "muon_ns_steps")
        values.update(
            {
                "weight_decay": 0.1,
                "ns_coefficients": [3.4445, -4.775, 2.0315],
                "eps": 1e-7,
                "adjust_lr": None,
                "parameter_constraint": "two_dimensional_only_v1",
            }
        )
        components = [
            _component("primary_update", "primary", "pytorch_muon", values)
        ]
        update_routes = [_all_parameters_route("primary_update")]
    else:
        raise ValueError(f"Unknown optimizer: {opt!r}")

    intent = {
        "strategy": strategy,
        "components": components,
        "scheduler": _scheduler_plan(args, opt),
        "gradient_processing": {
            "global_norm_clip": _arg(args, "grad_clip", None),
        },
        "routing": {
            "update_routes": update_routes,
            "overlays": overlays,
            "realization": {"status": "pending"},
        },
    }
    plan = {
        "schema_version": OPTIMIZATION_PLAN_SCHEMA_VERSION,
        **_normalize(intent),
    }
    plan["identity"] = _optimization_intent_identity(plan)
    return plan


def _parameter_numel(parameter):
    numel = getattr(parameter, "numel", None)
    if callable(numel):
        return int(numel())
    shape = tuple(getattr(parameter, "shape", ()))
    total = 1
    for dimension in shape:
        total *= int(dimension)
    return total


def _parameter_ndim(parameter):
    ndim = getattr(parameter, "ndim", None)
    if ndim is not None:
        return int(ndim)
    return len(tuple(getattr(parameter, "shape", ())))


def _parameter_names_sha256(names):
    return _sha256_bytes(_canonical_bytes(sorted(names)))


def _is_stable_group_value(value):
    if value is None or isinstance(value, (str, int, float, bool)):
        return True
    if isinstance(value, (list, tuple, set)):
        return all(_is_stable_group_value(item) for item in value)
    if isinstance(value, dict):
        return all(
            isinstance(key, (str, int, float, bool))
            and _is_stable_group_value(item)
            for key, item in value.items()
        )
    return False


def _semantic_group_hyperparameters(plan, group):
    components = plan.get("components", ())
    algorithms = {component.get("algorithm") for component in components}
    hyperparameters = {
        str(key): _normalize(value)
        for key, value in group.items()
        if key != "params" and _is_stable_group_value(value)
    }
    if "adafactor" in algorithms:
        # relative_step=True replaces the configured group lr in every update.
        hyperparameters.pop("lr", None)
    if "signsgd" in algorithms:
        # signSGD fixes momentum at zero, so these shared Signum fields do not
        # affect a successful update.
        hyperparameters.pop("dampening", None)
        hyperparameters.pop("nesterov", None)
    signum_component = next(
        (
            component
            for component in components
            if component.get("algorithm") == "signum"
        ),
        None,
    )
    if (
        signum_component is not None
        and signum_component.get("hyperparameters", {}).get("momentum") == 0
    ):
        hyperparameters.pop("dampening", None)
        hyperparameters.pop("nesterov", None)
    if "scion-light" in algorithms:
        # The current training loop clears gradients after every step, so the
        # in-gradient momentum scaling performed by ScionLight is not retained.
        hyperparameters.pop("momentum", None)
    mars_component = next(
        (
            component
            for component in components
            if component.get("algorithm") == "mars"
        ),
        None,
    )
    if mars_component is not None:
        variant = mars_component.get("hyperparameters", {}).get("variant")
        if variant in {"mars-lion", "mars-shampoo"}:
            betas = hyperparameters.pop("betas", None)
            if isinstance(betas, (list, tuple)) and betas:
                hyperparameters["beta1"] = _normalize(betas[0])
    if "soap" in algorithms:
        shampoo_beta = hyperparameters.get("shampoo_beta")
        betas = hyperparameters.get("betas")
        if (
            isinstance(shampoo_beta, (int, float))
            and shampoo_beta < 0
            and isinstance(betas, (list, tuple))
            and len(betas) > 1
        ):
            hyperparameters["shampoo_beta"] = _normalize(betas[1])
        # Current benchmark model families have no convolutional parameters.
        hyperparameters.pop("data_format", None)
    if "prodigy" in algorithms and hyperparameters.get("beta3") is None:
        betas = hyperparameters.get("betas")
        if isinstance(betas, (list, tuple)) and len(betas) > 1:
            hyperparameters["beta3"] = math.sqrt(float(betas[1]))
    return hyperparameters


def _summarize_parameters(parameters, names_by_id):
    names = [names_by_id[id(parameter)] for parameter in parameters]
    return {
        "tensor_count": len(parameters),
        "parameter_count": sum(_parameter_numel(parameter) for parameter in parameters),
        "parameter_names_sha256": _parameter_names_sha256(names),
    }


def _optimizer_parameters(optimizer):
    parameters = []
    seen = set()
    for group_index, group in enumerate(getattr(optimizer, "param_groups", ())):
        for parameter in group.get("params", ()):
            parameter_id = id(parameter)
            if parameter_id in seen:
                raise ValueError(
                    "Optimizer parameter appears in more than one parameter group: "
                    f"group {group_index}."
                )
            seen.add(parameter_id)
            parameters.append(parameter)
    return parameters


def _route_matches(parameter, optimizer, selector):
    kind = selector["kind"]
    if kind in {"all_optimizer_parameters", "scion_partition_policy"}:
        return True
    if kind == "optimizer_state_flag":
        state = getattr(optimizer, "state", {}).get(parameter, {})
        return state.get(selector["field"]) == selector["equals"]
    if kind == "parameter_ndim":
        ndim = _parameter_ndim(parameter)
        if "equals" in selector:
            return ndim == selector["equals"]
        return ndim != selector["not_equals"]
    raise ValueError(f"Unsupported update-route selector: {kind!r}")


def _overlay_parameters(
    contract,
    parameters,
    optimizer,
    names_by_id,
    explicit_param_ids,
):
    component = contract["component"]
    if component in explicit_param_ids:
        selected_ids = set(explicit_param_ids[component])
        unknown = selected_ids - {id(parameter) for parameter in parameters}
        if unknown:
            raise ValueError(
                f"Overlay {component!r} contains parameters outside the optimizer."
            )
        return [
            parameter for parameter in parameters if id(parameter) in selected_ids
        ]

    selector = contract["selector"]
    kind = selector["kind"]
    if kind == "all_optimizer_parameters":
        return list(parameters)
    if kind == "magma_scope":
        if selector["scope"] == "all":
            return list(parameters)
        return [
            parameter
            for parameter in parameters
            if any(
                token in names_by_id[id(parameter)].lower()
                for token in ("attn", "mlp")
            )
        ]
    if kind == "optimizer_preconditioner_map":
        selected_ids = {
            id(parameter)
            for parameter in getattr(optimizer, "_precond_map", {})
        }
        return [
            parameter for parameter in parameters if id(parameter) in selected_ids
        ]
    raise ValueError(f"Unsupported overlay selector: {kind!r}")


def resolve_optimization_plan(
    plan,
    optimizer,
    named_parameters,
    *,
    overlay_param_ids=None,
):
    """Attach optimizer-realized routing without mutating the intent object."""
    validate_optimization_plan(plan, expected_status="pending")

    named_parameters = (
        list(named_parameters.items())
        if isinstance(named_parameters, dict)
        else list(named_parameters)
    )
    names_by_id = {}
    for name, parameter in named_parameters:
        parameter_id = id(parameter)
        if parameter_id in names_by_id and names_by_id[parameter_id] != name:
            raise ValueError(
                "A parameter has multiple canonical names: "
                f"{names_by_id[parameter_id]!r} and {name!r}."
            )
        names_by_id[parameter_id] = str(name)

    parameters = _optimizer_parameters(optimizer)
    trainable_model_parameters = [
        parameter
        for _name, parameter in named_parameters
        if getattr(parameter, "requires_grad", True) is not False
    ]
    optimizer_parameter_ids = {id(parameter) for parameter in parameters}
    missing_from_optimizer = [
        parameter
        for parameter in trainable_model_parameters
        if id(parameter) not in optimizer_parameter_ids
    ]
    if missing_from_optimizer:
        raise ValueError(
            f"{len(missing_from_optimizer)} trainable model parameter(s) are "
            "missing from the optimizer."
        )
    missing_names = [
        parameter for parameter in parameters if id(parameter) not in names_by_id
    ]
    if missing_names:
        raise ValueError(
            f"{len(missing_names)} optimizer parameter(s) lack a canonical model name."
        )

    assignment_counts = {id(parameter): 0 for parameter in parameters}
    resolved_routes = []
    for contract in plan["routing"]["update_routes"]:
        selected = [
            parameter
            for parameter in parameters
            if _route_matches(parameter, optimizer, contract["selector"])
        ]
        for parameter in selected:
            assignment_counts[id(parameter)] += 1
        resolved_routes.append(
            {
                "component": contract["component"],
                **_summarize_parameters(selected, names_by_id),
            }
        )

    unassigned = [
        parameter
        for parameter in parameters
        if assignment_counts[id(parameter)] == 0
    ]
    multiply_assigned = [
        parameter
        for parameter in parameters
        if assignment_counts[id(parameter)] > 1
    ]
    if unassigned or multiply_assigned:
        raise ValueError(
            "Optimization updater routing must cover every optimizer parameter "
            "exactly once; "
            f"unassigned={len(unassigned)}, "
            f"multiply_assigned={len(multiply_assigned)}."
        )

    explicit_param_ids = overlay_param_ids or {}
    resolved_overlays = []
    for contract in plan["routing"]["overlays"]:
        selected = _overlay_parameters(
            contract,
            parameters,
            optimizer,
            names_by_id,
            explicit_param_ids,
        )
        resolved_overlays.append(
            {
                "component": contract["component"],
                **_summarize_parameters(selected, names_by_id),
            }
        )

    parameter_groups = []
    for group in getattr(optimizer, "param_groups", ()):
        selected = list(group.get("params", ()))
        hyperparameters = _semantic_group_hyperparameters(plan, group)
        parameter_groups.append(
            {
                **_summarize_parameters(selected, names_by_id),
                "hyperparameters": hyperparameters,
            }
        )
    parameter_groups.sort(
        key=lambda group: (
            group["parameter_names_sha256"],
            _canonical_bytes(group["hyperparameters"]),
        )
    )

    coverage = {
        "model_trainable_tensor_count": len(trainable_model_parameters),
        "optimizer_tensor_count": len(parameters),
        "optimizer_parameter_count": sum(
            _parameter_numel(parameter) for parameter in parameters
        ),
        "missing_from_optimizer_tensors": 0,
        "unassigned_tensors": 0,
        "multiply_assigned_tensors": 0,
    }
    realization_payload = {
        "update_routes": resolved_routes,
        "overlays": resolved_overlays,
        "parameter_groups": parameter_groups,
        "coverage": coverage,
        "optimizer_class": type(optimizer).__name__,
    }
    realization = {
        "status": "resolved",
        **realization_payload,
        "identity": _sha256_bytes(_canonical_bytes(realization_payload)),
    }
    resolved = _normalize(plan)
    resolved["routing"]["realization"] = realization
    validate_optimization_plan(resolved, expected_status="resolved")
    return resolved


def _evaluation_protocol_payload(protocol):
    if not isinstance(protocol, dict):
        raise ValueError("Evaluation protocol must be an object.")
    return {
        key: value
        for key, value in protocol.items()
        if key != "identity"
    }


def _evaluation_protocol_identity(protocol):
    return _sha256_bytes(
        _canonical_bytes(_evaluation_protocol_payload(protocol))
    )


def validate_evaluation_protocol(protocol):
    expected_identity = _evaluation_protocol_identity(protocol)
    if protocol.get("identity") != expected_identity:
        raise ValueError(
            "Evaluation protocol identity does not match its canonical content."
        )
    return protocol


def build_evaluation_protocol(args):
    """Describe reporting-only validation budgets without changing train identity."""
    final_eval_batches = getattr(args, "final_eval_batches", None)
    final_eval_tokens = getattr(args, "final_eval_tokens", None)
    if final_eval_batches is not None and final_eval_tokens is not None:
        raise ValueError(
            "final_eval_batches and final_eval_tokens are mutually exclusive"
        )

    if final_eval_tokens is not None:
        final_and_full = {
            "mode": "token_cap",
            "max_tokens": int(final_eval_tokens),
        }
    elif final_eval_batches is not None:
        final_and_full = {
            "mode": "batch_cap",
            "max_batches": int(final_eval_batches),
        }
    else:
        final_and_full = {"mode": "full_dataset"}

    periodic_batches = getattr(args, "eval_batches", None)
    periodic = (
        {
            "mode": "batch_cap",
            "max_batches": int(periodic_batches),
            "interval": getattr(args, "eval_interval", None),
        }
        if periodic_batches is not None
        else {"mode": "unspecified"}
    )
    protocol = {
        "periodic": periodic,
        "final_and_full": final_and_full,
        "full_eval_at": sorted(
            {int(step) for step in (getattr(args, "full_eval_at", None) or [])}
        ),
        "generation_prefix": getattr(args, "eval_seq_prefix", "none"),
        "token_counting": "targets_not_equal_to_-1",
    }
    result = {
        **protocol,
        "identity": _sha256_bytes(_canonical_bytes(protocol)),
    }
    validate_evaluation_protocol(result)
    return result


def build_training_semantics(args, optimization_plan=None):
    """Return a reader-facing view derived from identity-bearing plan semantics."""
    if optimization_plan is not None:
        semantics = {}
        for component in optimization_plan.get("components", ()):
            hyperparameters = component.get("hyperparameters", {})
            if component.get("algorithm") in {
                "schedule_free_adamw",
                "schedule_free_sgd",
            }:
                semantics["schedule_free_parameter_point"] = hyperparameters[
                    "parameter_point_version"
                ]
            if component.get("id") == "hessian_estimator":
                semantics["sophia_hessian_estimator"] = hyperparameters
        return semantics

    semantics = {}
    if getattr(args, "opt", None) in {"sf-adamw", "sf-sgd"}:
        semantics["schedule_free_parameter_point"] = (
            "schedule_free_parameter_point_v2"
        )
    if getattr(args, "opt", None) == "sophiag":
        mode = getattr(
            args, "sophia_estimator_mode", "legacy_last_microbatch"
        )
        version = {
            "legacy_last_microbatch": "legacy_last_microbatch_v1",
            "global_accum": "global_accum_gnb_v1",
        }[mode]
        world_size = int(getattr(args, "world_size", 1))
        local_batch = int(getattr(args, "batch_size"))
        local_acc_steps = int(getattr(args, "acc_steps"))
        examples_per_refresh = world_size * local_batch
        if mode == "global_accum":
            examples_per_refresh *= local_acc_steps
        sequence_length = int(getattr(args, "sequence_length"))
        semantics["sophia_hessian_estimator"] = {
            "mode": mode,
            "version": version,
            "refresh_frequency": int(getattr(args, "precondition_frequency")),
            "expected_examples_per_refresh": examples_per_refresh,
            "expected_tokens_per_refresh": examples_per_refresh * sequence_length,
            "optimizer_scale_tokens": int(getattr(args, "sophia_bs"))
            * sequence_length,
        }
    return semantics


def build_metric_semantics(data_manifest, args=None):
    """Return reporting definitions; these are recorded but not train-compatible state."""
    training_loss = (
        "gn_last_inner_base_loss_v1"
        if getattr(args, "opt", None) in {"gn-prox", "gn-full"}
        else "global_step_microbatch_mean_v2"
    )
    return {
        "training_loss": training_loss,
        "validation_loss": "next_token_cross_entropy",
        "validation_accuracy": (
            "token_accuracy_epfl_flat_eotpad_v1"
            if data_manifest["semantics_id"] == "epfl-flat-eotpad-v1"
            else "next_token_accuracy"
        ),
    }


def compatibility_runtime(runtime):
    """Keep runtime semantics in the identity, but exclude installation locations."""
    runtime = _normalize(runtime)
    python = runtime.get("python", {})
    host = runtime.get("platform", {})
    compatible = {
        "python": {
            key: python.get(key)
            for key in ("implementation", "version")
            if key in python
        },
        "platform": {
            key: host.get(key)
            for key in ("system", "machine")
            if key in host
        },
        "packages": runtime.get("packages", {}),
    }
    if "torch_runtime" in runtime:
        compatible["torch_runtime"] = runtime["torch_runtime"]
    return compatible


def _manifest_preflight_payload(
    compatibility,
    optimization_plan_identity,
    code,
    data_identity,
    runtime,
):
    return {
        "schema_version": RUN_MANIFEST_SCHEMA_VERSION,
        "config": compatibility,
        "optimization_plan_identity": optimization_plan_identity,
        "code": {
            "head": code.get("head"),
            "source_fingerprint_sha256": code.get("source_fingerprint_sha256"),
        },
        "data_identity": data_identity,
        "runtime": compatibility_runtime(runtime),
    }


def _resolved_run_identity(preflight_identity, realization_identity):
    return _sha256_bytes(
        _canonical_bytes(
            {
                "schema_version": RUN_MANIFEST_SCHEMA_VERSION,
                "preflight_identity": preflight_identity,
                "optimization_realization_identity": realization_identity,
            }
        )
    )


def validate_run_manifest(manifest):
    if not isinstance(manifest, dict):
        raise ValueError("Run manifest must be an object.")
    if manifest.get("schema_version") != RUN_MANIFEST_SCHEMA_VERSION:
        raise ValueError(
            f"Unsupported run manifest schema: {manifest.get('schema_version')!r}."
        )
    state = manifest.get("manifest_state")
    if state not in {"preflight", "resolved"}:
        raise ValueError(f"Unknown run manifest state: {state!r}.")
    try:
        plan = manifest["optimization_plan"]
        compatibility = manifest["compatibility_config"]
        code = manifest["code"]
        data_identity = manifest["data"]["identity"]
        runtime = manifest["runtime"]
    except (KeyError, TypeError) as exc:
        raise ValueError("Run manifest is missing identity-bearing fields.") from exc
    validate_optimization_plan(
        plan,
        expected_status="resolved" if state == "resolved" else "pending",
    )
    try:
        evaluation_protocol = manifest["evaluation_protocol"]
    except (KeyError, TypeError) as exc:
        raise ValueError("Run manifest has no evaluation protocol.") from exc
    validate_evaluation_protocol(evaluation_protocol)
    preflight_payload = _manifest_preflight_payload(
        compatibility,
        plan["identity"],
        code,
        data_identity,
        runtime,
    )
    expected_preflight = _sha256_bytes(_canonical_bytes(preflight_payload))
    if manifest.get("preflight_identity") != expected_preflight:
        raise ValueError(
            "Run manifest preflight identity does not match its canonical content."
        )
    expected_run_identity = (
        _resolved_run_identity(
            expected_preflight,
            plan["routing"]["realization"]["identity"],
        )
        if state == "resolved"
        else expected_preflight
    )
    if manifest.get("run_identity") != expected_run_identity:
        raise ValueError(
            "Run manifest run identity does not match its canonical content."
        )
    return manifest


def build_run_manifest(
    args,
    data_manifest,
    *,
    code=None,
    runtime=None,
    optimization_plan=None,
):
    code = collect_code_fingerprint() if code is None else _normalize(code)
    runtime = collect_runtime_identity() if runtime is None else _normalize(runtime)
    optimization_plan = (
        build_optimization_plan(args)
        if optimization_plan is None
        else _normalize(optimization_plan)
    )
    validate_optimization_plan(optimization_plan)
    training_semantics = build_training_semantics(args, optimization_plan)
    preflight_payload = _manifest_preflight_payload(
        compatibility_config(args),
        optimization_plan["identity"],
        code,
        data_manifest["identity"],
        runtime,
    )
    preflight_identity = _sha256_bytes(_canonical_bytes(preflight_payload))
    realization = optimization_plan["routing"]["realization"]
    is_resolved = realization.get("status") == "resolved"
    run_identity = (
        _resolved_run_identity(
            preflight_identity,
            realization["identity"],
        )
        if is_resolved
        else preflight_identity
    )
    manifest = {
        "schema_version": RUN_MANIFEST_SCHEMA_VERSION,
        "manifest_state": "resolved" if is_resolved else "preflight",
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "preflight_identity": preflight_identity,
        "run_identity": run_identity,
        "config": sanitized_config(args),
        "compatibility_config": preflight_payload["config"],
        "optimization_plan": optimization_plan,
        "evaluation_protocol": build_evaluation_protocol(args),
        "training_semantics": training_semantics,
        "code": code,
        "runtime": runtime,
        "data": data_manifest,
        "metric_semantics": build_metric_semantics(data_manifest, args),
    }
    validate_run_manifest(manifest)
    return manifest


def _serialized_json(payload):
    return json.dumps(_normalize(payload), indent=2, sort_keys=True) + "\n"


def _fsync_directory(path):
    try:
        descriptor = os.open(path, os.O_RDONLY)
    except OSError:
        return
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def write_json_atomic(path, payload):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    serialized = _serialized_json(payload)
    temporary_path = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="w",
            encoding="utf-8",
            dir=path.parent,
            prefix=f".{path.name}.",
            delete=False,
        ) as temporary:
            temporary.write(serialized)
            temporary.flush()
            os.fsync(temporary.fileno())
            temporary_path = Path(temporary.name)
        os.replace(temporary_path, path)
        temporary_path = None
        _fsync_directory(path.parent)
    finally:
        if temporary_path is not None and temporary_path.exists():
            temporary_path.unlink()


def write_json_atomic_if_missing(path, payload):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    serialized = _serialized_json(payload)
    temporary_path = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="w",
            encoding="utf-8",
            dir=path.parent,
            prefix=f".{path.name}.",
            delete=False,
        ) as temporary:
            temporary.write(serialized)
            temporary.flush()
            os.fsync(temporary.fileno())
            temporary_path = Path(temporary.name)
        try:
            os.link(temporary_path, path)
        except FileExistsError:
            return False
        _fsync_directory(path.parent)
        return True
    finally:
        if temporary_path is not None and temporary_path.exists():
            temporary_path.unlink()


def _read_json_object(path):
    try:
        payload = json.loads(Path(path).read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ValueError(f"Invalid run manifest {path}: {exc}") from exc
    if not isinstance(payload, dict):
        raise ValueError(f"Run manifest must contain a JSON object: {path}.")
    return payload


def require_resolved_run_manifest(path):
    path = Path(path)
    if not path.is_file():
        raise ValueError(
            "Resume requires an existing schema-v3 resolved run manifest; "
            "legacy run artifacts are read-only."
        )
    existing = _read_json_object(path)
    try:
        validate_run_manifest(existing)
    except ValueError as exc:
        raise ValueError(
            "Resume requires a valid schema-v3 resolved run manifest."
        ) from exc
    if existing.get("manifest_state") != "resolved":
        raise ValueError(
            "Resume requires a resolved run manifest; a preflight manifest "
            "cannot own a checkpoint."
        )
    return existing


def ensure_compatible_manifest(path, current_manifest, *, phase="resolved"):
    if phase not in {"preflight", "resolved"}:
        raise ValueError(f"Unknown manifest compatibility phase: {phase!r}")
    expected_state = "preflight" if phase == "preflight" else "resolved"
    validate_run_manifest(current_manifest)
    if current_manifest.get("manifest_state") != expected_state:
        raise ValueError(
            f"{phase.capitalize()} compatibility requires a "
            f"{expected_state} manifest."
        )

    path = Path(path)
    if not path.exists():
        if phase == "resolved":
            raise ValueError(
                "Cannot write a resolved manifest without an existing "
                "preflight manifest."
            )
        return "create"
    existing = _read_json_object(path)
    expected_schema = current_manifest.get("schema_version")
    actual_schema = existing.get("schema_version")
    if actual_schema != expected_schema:
        raise ValueError(
            f"Existing run manifest schema {actual_schema!r} does not match "
            f"current schema {expected_schema!r}; legacy manifests are read-only "
            "and require a different experiment directory."
        )
    validate_run_manifest(existing)

    expected_preflight = current_manifest.get("preflight_identity")
    actual_preflight = existing.get("preflight_identity")
    if actual_preflight != expected_preflight:
        raise ValueError(
            f"Existing preflight identity {actual_preflight!r} does not match "
            f"current preflight identity {expected_preflight!r}; choose a "
            "different experiment directory."
        )

    if existing.get("manifest_state") == "resolved":
        existing_protocol = existing.get("evaluation_protocol")
        current_protocol = current_manifest.get("evaluation_protocol")
        existing_protocol_identity = (
            existing_protocol.get("identity")
            if isinstance(existing_protocol, dict)
            else None
        )
        current_protocol_identity = (
            current_protocol.get("identity")
            if isinstance(current_protocol, dict)
            else None
        )
        if existing_protocol_identity != current_protocol_identity:
            raise ValueError(
                "Existing resolved evaluation protocol identity "
                f"{existing_protocol_identity!r} does not match current identity "
                f"{current_protocol_identity!r}; resume with the original "
                "evaluation settings or choose a different experiment directory."
            )

    if phase == "preflight":
        return "keep"
    if existing.get("manifest_state") == "preflight":
        return "upgrade"

    expected = current_manifest.get("run_identity")
    actual = existing.get("run_identity")
    if actual != expected:
        raise ValueError(
            f"Existing run identity {actual!r} does not match current run identity "
            f"{expected!r}; choose a different experiment directory."
        )
    return "keep"


def reconcile_run_manifest(path, current_manifest, *, phase="resolved"):
    action = ensure_compatible_manifest(path, current_manifest, phase=phase)
    if action == "keep":
        return action
    if action == "create":
        if write_json_atomic_if_missing(path, current_manifest):
            return action
        return ensure_compatible_manifest(path, current_manifest, phase=phase)
    if action == "upgrade":
        lock_path = path.with_name(f".{path.stem}.lock")
        lock_path.parent.mkdir(parents=True, exist_ok=True)
        with lock_path.open("a+", encoding="utf-8") as lock_file:
            fcntl.flock(lock_file.fileno(), fcntl.LOCK_EX)
            try:
                locked_action = ensure_compatible_manifest(
                    path,
                    current_manifest,
                    phase=phase,
                )
                if locked_action == "keep":
                    return locked_action
                if locked_action != "upgrade":
                    raise RuntimeError(
                        "Unexpected locked run manifest reconciliation action: "
                        f"{locked_action!r}"
                    )
                write_json_atomic(path, current_manifest)
                return locked_action
            finally:
                fcntl.flock(lock_file.fileno(), fcntl.LOCK_UN)
    raise RuntimeError(f"Unexpected run manifest reconciliation action: {action!r}")
