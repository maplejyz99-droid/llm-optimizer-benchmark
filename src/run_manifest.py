"""Credential-safe run and data artifact identity helpers."""

import hashlib
import importlib.metadata
import json
import os
import platform
import subprocess
import sys
import tempfile
from datetime import datetime, timezone
from pathlib import Path


RUN_MANIFEST_SCHEMA_VERSION = 1
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
    "metric_semantics",
    "wandb",
    "wandb_entity",
    "wandb_project",
    "wandb_run_prefix",
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
        and key not in SENSITIVE_CONFIG_KEYS
        and not key.startswith("notify_")
    }
    if "run_seed" in config:
        config["seed"] = config.pop("run_seed")
    if hasattr(args, "device"):
        config["device_type"] = _normalized_device_type(args.device)
    return config


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


def build_run_manifest(args, data_manifest, *, code=None, runtime=None):
    code = collect_code_fingerprint() if code is None else _normalize(code)
    runtime = collect_runtime_identity() if runtime is None else _normalize(runtime)
    identity_payload = {
        "config": compatibility_config(args),
        "code": {
            "head": code.get("head"),
            "source_fingerprint_sha256": code.get("source_fingerprint_sha256"),
        },
        "data_identity": data_manifest["identity"],
        "runtime": compatibility_runtime(runtime),
    }
    return {
        "schema_version": RUN_MANIFEST_SCHEMA_VERSION,
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "run_identity": _sha256_bytes(_canonical_bytes(identity_payload)),
        "config": sanitized_config(args),
        "compatibility_config": identity_payload["config"],
        "code": code,
        "runtime": runtime,
        "data": data_manifest,
        "metric_semantics": {
            "validation_loss": "next_token_cross_entropy",
            "validation_accuracy": (
                "token_accuracy_epfl_flat_eotpad_v1"
                if data_manifest["semantics_id"] == "epfl-flat-eotpad-v1"
                else "next_token_accuracy"
            ),
        },
    }


def write_json_atomic(path, payload):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    serialized = json.dumps(_normalize(payload), indent=2, sort_keys=True) + "\n"
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
        temporary_path.replace(path)
    finally:
        if temporary_path is not None and temporary_path.exists():
            temporary_path.unlink()


def ensure_compatible_manifest(path, current_manifest):
    path = Path(path)
    if not path.exists():
        return
    existing = json.loads(path.read_text(encoding="utf-8"))
    expected = current_manifest.get("run_identity")
    actual = existing.get("run_identity")
    if actual != expected:
        raise ValueError(
            f"Existing run identity {actual!r} does not match current run identity "
            f"{expected!r}; choose a different experiment directory."
        )
