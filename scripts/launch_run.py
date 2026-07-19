#!/usr/bin/env python3
"""Launch one immutable benchmark run with logs and lifecycle metadata."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import shlex
import signal
import subprocess
import sys
import tempfile
import threading
import time
import uuid
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


SAFE_COMPONENT = re.compile(r"[A-Za-z0-9][A-Za-z0-9._-]{0,127}\Z")
OWNED_TRAINING_OPTIONS = {"--experiment_name", "--results_base_folder"}
UNSUPPORTED_TRAINING_OPTIONS = {"--resume_from"}
PUBLIC_NOTIFY_OPTIONS = {"notify_interval", "notify_method"}
SECRET_COMPONENTS = {
    "api_key",
    "credential",
    "credentials",
    "key",
    "passwd",
    "password",
    "secret",
    "smtp_pass",
    "token",
    "webhook",
}
NVIDIA_SMI_QUERY = (
    "timestamp,index,uuid,name,memory.used,memory.total,"
    "utilization.gpu,utilization.memory,power.draw,temperature.gpu"
)
NVIDIA_SMI_HEADER = NVIDIA_SMI_QUERY + "\n"
RUN_MANIFEST_SCHEMA_VERSION = 3
OPTIMIZATION_PLAN_SCHEMA_VERSION = 1


@dataclass(frozen=True)
class RunPaths:
    runs_root: Path
    suite_dir: Path
    run_dir: Path
    launch_dir: Path
    logs_dir: Path


@dataclass(frozen=True)
class ArtifactVerification:
    ok: bool
    reason: str | None = None


class _StopLaunch(Exception):
    """Internal control flow after status has already been populated."""


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _absolute(path: Path) -> Path:
    return Path(os.path.abspath(os.fspath(path.expanduser())))


def _option_name(argument: str) -> str | None:
    if not argument.startswith("--"):
        return None
    return argument.split("=", 1)[0]


def _normalized_option_component(argument: str) -> str | None:
    name = _option_name(argument)
    if name is None:
        return None
    return name[2:].lower().replace("-", "_")


def _is_secret_option(argument: str) -> bool:
    component = _normalized_option_component(argument)
    if component is None:
        return False
    if component.startswith("notify_") and component not in PUBLIC_NOTIFY_OPTIONS:
        return True
    parts = component.split("_")
    if component in SECRET_COMPONENTS:
        return True
    if any(part in SECRET_COMPONENTS for part in parts):
        return True
    return component.endswith("_api_key") or component.endswith("_webhook")


def redact_argv(argv: list[str]) -> list[str]:
    redacted: list[str] = []
    index = 0
    while index < len(argv):
        argument = argv[index]
        if _is_secret_option(argument):
            if "=" in argument:
                redacted.append(f"{argument.split('=', 1)[0]}=<redacted>")
            else:
                redacted.append(argument)
                if index + 1 < len(argv) and not argv[index + 1].startswith("--"):
                    redacted.append("<redacted>")
                    index += 1
            index += 1
            continue
        redacted.append(argument)
        index += 1
    return redacted


def _validate_component(value: str, label: str) -> None:
    if not SAFE_COMPONENT.fullmatch(value):
        raise ValueError(
            f"{label} must match {SAFE_COMPONENT.pattern!r}; received {value!r}"
        )


def resolve_run_paths(runs_root: Path, suite: str, run_id: str) -> RunPaths:
    root = _absolute(runs_root)
    suite_path = Path(suite)
    if suite_path.is_absolute() or not suite_path.parts:
        raise ValueError("suite must be a non-empty relative path")
    for component in suite_path.parts:
        if component in {".", ".."}:
            raise ValueError("suite may not contain '.' or '..'")
        _validate_component(component, "suite component")
    _validate_component(run_id, "run-id")

    suite_dir = root.joinpath(*suite_path.parts)
    run_dir = suite_dir / run_id
    resolved_root = root.resolve(strict=False)
    resolved_run = run_dir.resolve(strict=False)
    if not resolved_run.is_relative_to(resolved_root):
        raise ValueError("resolved run path escapes runs-root")
    return RunPaths(
        runs_root=root,
        suite_dir=suite_dir,
        run_dir=run_dir,
        launch_dir=run_dir / "launch",
        logs_dir=run_dir / "logs",
    )


def validate_training_args(
    training_args: list[str], nproc_per_node: int | None
) -> None:
    if not training_args:
        raise ValueError("training arguments are required after '--'")
    option_names = {_option_name(argument) for argument in training_args}
    conflicts = sorted(OWNED_TRAINING_OPTIONS & option_names)
    if conflicts:
        raise ValueError(
            "launcher owns these training options: " + ", ".join(conflicts)
        )
    unsupported = sorted(UNSUPPORTED_TRAINING_OPTIONS & option_names)
    if unsupported:
        raise ValueError(
            "v1 launcher only supports new runs; unsupported: "
            + ", ".join(unsupported)
        )

    has_distributed_backend = "--distributed_backend" in option_names
    if has_distributed_backend and nproc_per_node is None:
        raise ValueError(
            "--distributed_backend requires launcher option --nproc-per-node"
        )
    if nproc_per_node is not None and nproc_per_node > 1 and not has_distributed_backend:
        raise ValueError(
            "--nproc-per-node greater than 1 requires --distributed_backend"
        )


def parse_gpu_ids(raw: str | None) -> list[str]:
    if raw is None:
        return []
    values = [value.strip() for value in raw.split(",")]
    if not values or any(not value for value in values):
        raise ValueError("gpu-ids must be a comma-separated non-empty list")
    allowed = re.compile(r"[A-Za-z0-9_.:/-]+\Z")
    if any(not allowed.fullmatch(value) for value in values):
        raise ValueError("gpu-ids contains an unsupported character")
    if len(set(values)) != len(values):
        raise ValueError("gpu-ids may not contain duplicates")
    return values


def build_child_command(
    python: Path,
    repo_root: Path,
    paths: RunPaths,
    run_id: str,
    training_args: list[str],
    nproc_per_node: int | None,
) -> list[str]:
    main_path = _absolute(repo_root) / "src" / "main.py"
    if not main_path.is_file():
        raise ValueError(f"training entrypoint does not exist: {main_path}")
    command = [str(_absolute(python))]
    if nproc_per_node is not None:
        command.extend(
            [
                "-m",
                "torch.distributed.run",
                "--standalone",
                f"--nproc_per_node={nproc_per_node}",
            ]
        )
    command.append(str(main_path))
    command.extend(training_args)
    command.extend(
        [
            "--results_base_folder",
            str(paths.suite_dir),
            "--experiment_name",
            run_id,
        ]
    )
    return command


def _fsync_directory(path: Path) -> None:
    try:
        descriptor = os.open(path, os.O_RDONLY)
    except OSError:
        return
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def write_text_atomic(path: Path, content: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary_path: Path | None = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="w",
            encoding="utf-8",
            dir=path.parent,
            prefix=f".{path.name}.",
            delete=False,
        ) as temporary:
            temporary.write(content)
            temporary.flush()
            os.fsync(temporary.fileno())
            temporary_path = Path(temporary.name)
        os.replace(temporary_path, path)
        temporary_path = None
        _fsync_directory(path.parent)
    finally:
        if temporary_path is not None and temporary_path.exists():
            temporary_path.unlink()


def write_json_atomic(path: Path, payload: dict[str, Any]) -> None:
    write_text_atomic(path, json.dumps(payload, indent=2, sort_keys=True) + "\n")


def _read_json_object(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"expected a JSON object: {path}")
    return payload


def _canonical_sha256(payload: Any) -> str:
    encoded = json.dumps(
        payload,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _verified_evaluation_protocol_identity(
    payload: dict[str, Any], label: str
) -> tuple[str | None, str | None]:
    protocol_label = (
        "evaluation protocol"
        if label == "evaluation"
        else f"{label} evaluation protocol"
    )
    protocol = payload.get("evaluation_protocol")
    if not isinstance(protocol, dict):
        return None, f"{label} has no evaluation protocol"
    identity = protocol.get("identity")
    if not isinstance(identity, str) or not identity:
        return None, f"{label} has no evaluation protocol identity"
    protocol_payload = {
        key: value
        for key, value in protocol.items()
        if key != "identity"
    }
    if identity != _canonical_sha256(protocol_payload):
        return (
            None,
            f"{protocol_label} identity does not match canonical content",
        )
    return identity, None


def _resolved_optimization_identities(
    payload: dict[str, Any], label: str
) -> tuple[tuple[str, str] | None, str | None]:
    plan = payload.get("optimization_plan")
    if not isinstance(plan, dict):
        return None, f"{label} has no optimization plan"
    if plan.get("schema_version") != OPTIMIZATION_PLAN_SCHEMA_VERSION:
        return None, f"{label} optimization plan schema mismatch"

    plan_identity = plan.get("identity")
    if not isinstance(plan_identity, str) or not plan_identity:
        return None, f"{label} has no optimization plan identity"

    routing = plan.get("routing")
    try:
        intent_payload = {
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
    except (KeyError, TypeError):
        return None, f"{label} optimization plan is missing canonical intent fields"
    if plan_identity != _canonical_sha256(intent_payload):
        return (
            None,
            f"{label} optimization plan identity does not match canonical intent",
        )

    realization = (
        routing.get("realization") if isinstance(routing, dict) else None
    )
    if (
        not isinstance(realization, dict)
        or realization.get("status") != "resolved"
    ):
        return None, f"{label} optimization plan is not resolved"
    realization_identity = realization.get("identity")
    if not isinstance(realization_identity, str) or not realization_identity:
        return None, f"{label} has no optimization realization identity"
    realization_payload = {
        key: value
        for key, value in realization.items()
        if key not in {"status", "identity"}
    }
    if realization_identity != _canonical_sha256(realization_payload):
        return (
            None,
            f"{label} optimization realization identity does not match "
            "canonical content",
        )
    return (plan_identity, realization_identity), None


def _verify_summary_identity_contract(
    *,
    label: str,
    payload: dict[str, Any],
    run_identity: str,
    protocol_identity: str,
    optimization_identities: tuple[str, str],
) -> str | None:
    if payload.get("run_identity") != run_identity:
        return f"{label} run_identity mismatch"

    payload_protocol_identity, reason = (
        _verified_evaluation_protocol_identity(payload, label)
    )
    if reason is not None:
        return reason
    if payload_protocol_identity != protocol_identity:
        return f"{label} evaluation protocol identity mismatch"

    payload_optimization_identities, reason = _resolved_optimization_identities(
        payload, label
    )
    if reason is not None:
        return reason
    if payload_optimization_identities is None:
        return f"{label} has invalid optimization identities"
    if payload_optimization_identities[0] != optimization_identities[0]:
        return f"{label} optimization plan identity mismatch"
    if payload_optimization_identities[1] != optimization_identities[1]:
        return f"{label} optimization realization identity mismatch"
    return None


def verify_completed_artifacts(run_dir: Path) -> ArtifactVerification:
    manifest_path = run_dir / "run_manifest.json"
    summary_path = run_dir / "summary.json"
    try:
        manifest = _read_json_object(manifest_path)
        summary = _read_json_object(summary_path)
    except (OSError, ValueError, json.JSONDecodeError) as exc:
        return ArtifactVerification(False, f"invalid required artifact: {exc}")

    if manifest.get("schema_version") != RUN_MANIFEST_SCHEMA_VERSION:
        return ArtifactVerification(False, "run manifest schema mismatch")
    if manifest.get("manifest_state") != "resolved":
        return ArtifactVerification(False, "run manifest is not resolved")

    run_identity = manifest.get("run_identity")
    if not isinstance(run_identity, str) or not run_identity:
        return ArtifactVerification(False, "manifest has no run_identity")

    optimization_identities, reason = _resolved_optimization_identities(
        manifest, "manifest"
    )
    if reason is not None:
        return ArtifactVerification(False, reason)
    if optimization_identities is None:
        return ArtifactVerification(
            False, "manifest has invalid optimization identities"
        )

    protocol_id, reason = _verified_evaluation_protocol_identity(
        manifest, "manifest"
    )
    if reason is not None:
        return ArtifactVerification(False, reason)
    if not isinstance(protocol_id, str) or not SAFE_COMPONENT.fullmatch(protocol_id):
        return ArtifactVerification(
            False, "manifest has no evaluation protocol identity"
        )

    preflight_identity = manifest.get("preflight_identity")
    if not isinstance(preflight_identity, str) or not preflight_identity:
        return ArtifactVerification(False, "manifest has no preflight_identity")
    expected_run_identity = _canonical_sha256(
        {
            "schema_version": RUN_MANIFEST_SCHEMA_VERSION,
            "preflight_identity": preflight_identity,
            "optimization_realization_identity": optimization_identities[1],
        }
    )
    if run_identity != expected_run_identity:
        return ArtifactVerification(
            False,
            "manifest run_identity does not match canonical resolved identity",
        )

    reason = _verify_summary_identity_contract(
        label="summary",
        payload=summary,
        run_identity=run_identity,
        protocol_identity=protocol_id,
        optimization_identities=optimization_identities,
    )
    if reason is not None:
        return ArtifactVerification(False, reason)

    evaluation_path = run_dir / "evaluations" / str(protocol_id) / "summary.json"
    try:
        evaluation = _read_json_object(evaluation_path)
    except (OSError, ValueError, json.JSONDecodeError) as exc:
        return ArtifactVerification(False, f"invalid evaluation summary: {exc}")

    reason = _verify_summary_identity_contract(
        label="evaluation",
        payload=evaluation,
        run_identity=run_identity,
        protocol_identity=protocol_id,
        optimization_identities=optimization_identities,
    )
    if reason is not None:
        return ArtifactVerification(False, reason)
    return ArtifactVerification(True)


def _git_head(repo_root: Path) -> str | None:
    result = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=repo_root,
        text=True,
        capture_output=True,
        check=False,
    )
    return result.stdout.strip() if result.returncode == 0 else None


class GpuMonitor:
    def __init__(self, gpu_ids: list[str], output_path: Path, interval: float):
        self.gpu_ids = gpu_ids
        self.output_path = output_path
        self.interval = interval
        self.error_path = output_path.with_suffix(".error.log")
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None
        self._sample_failures = 0
        self._state = "not_started"
        self._shutdown_timed_out = False

    def _command(self) -> list[str]:
        return [
            "nvidia-smi",
            f"--id={','.join(self.gpu_ids)}",
            f"--query-gpu={NVIDIA_SMI_QUERY}",
            "--format=csv,noheader,nounits",
        ]

    def _sample(self) -> str:
        result = subprocess.run(
            self._command(),
            text=True,
            capture_output=True,
            timeout=max(5.0, min(self.interval, 30.0)),
            check=False,
        )
        if result.returncode != 0:
            detail = result.stderr.strip() or f"exit code {result.returncode}"
            raise RuntimeError(f"nvidia-smi sampling failed: {detail}")
        if not result.stdout.strip():
            raise RuntimeError("nvidia-smi sampling returned no GPU rows")
        return result.stdout

    def preflight(self) -> str:
        return self._sample()

    def start(self, initial_sample: str) -> None:
        self.output_path.parent.mkdir(parents=True, exist_ok=True)
        self.output_path.write_text(
            NVIDIA_SMI_HEADER + initial_sample,
            encoding="utf-8",
        )
        self._state = "running"
        self._thread = threading.Thread(target=self._run, daemon=True)
        self._thread.start()

    def _record_error(self, exc: Exception) -> None:
        self._sample_failures += 1
        with self.error_path.open("a", encoding="utf-8") as stream:
            stream.write(f"{utc_now()} {type(exc).__name__}: {exc}\n")

    def _run(self) -> None:
        while not self._stop.wait(self.interval):
            try:
                sample = self._sample()
                with self.output_path.open("a", encoding="utf-8") as stream:
                    stream.write(sample)
            except Exception as exc:  # monitor failure must not kill training
                self._record_error(exc)
        if self._shutdown_timed_out:
            self._state = "shutdown_timeout"
        else:
            self._state = "degraded" if self._sample_failures else "completed"

    def stop(self) -> None:
        self._stop.set()
        if self._thread is not None:
            self._thread.join(timeout=max(5.0, min(self.interval + 2.0, 32.0)))
            if self._thread.is_alive():
                self._shutdown_timed_out = True
                self._state = "shutdown_timeout"
                return
        if self._state == "running" and not self._shutdown_timed_out:
            self._state = "degraded" if self._sample_failures else "completed"

    def snapshot(self) -> dict[str, Any]:
        return {
            "enabled": True,
            "gpu_ids": self.gpu_ids,
            "state": self._state,
            "sample_failures": self._sample_failures,
        }


def _normalize_return_code(return_code: int) -> int:
    return 128 + abs(return_code) if return_code < 0 else return_code


def _process_group_exists(process_group_id: int) -> bool:
    try:
        os.killpg(process_group_id, 0)
    except ProcessLookupError:
        return False
    except PermissionError:
        return True
    return True


def _signal_process_group(process_group_id: int, signum: int) -> None:
    try:
        os.killpg(process_group_id, signum)
    except ProcessLookupError:
        pass


def _terminate_process_group(
    process: subprocess.Popen[bytes], grace_seconds: float
) -> None:
    _signal_process_group(process.pid, signal.SIGTERM)
    deadline = time.monotonic() + grace_seconds
    while _process_group_exists(process.pid) and time.monotonic() < deadline:
        process.poll()
        time.sleep(0.05)
    if _process_group_exists(process.pid):
        _signal_process_group(process.pid, signal.SIGKILL)
    try:
        process.wait(timeout=max(1.0, grace_seconds))
    except subprocess.TimeoutExpired:
        pass


def _command_record(
    *,
    command: list[str],
    run_id: str,
    suite: str,
    repo_root: Path,
    gpu_ids: list[str],
    started_at: str,
) -> str:
    inherited_cuda_devices = os.environ.get("CUDA_VISIBLE_DEVICES")
    if gpu_ids:
        visible_devices = ",".join(gpu_ids)
    elif inherited_cuda_devices is not None:
        visible_devices = inherited_cuda_devices
    else:
        visible_devices = "<unset>"
    values = {
        "run_id": run_id,
        "suite": suite,
        "repo_root": str(repo_root),
        "git_head": _git_head(repo_root) or "unknown",
        "cuda_visible_devices": visible_devices,
        "started_at_utc": started_at,
        "command": shlex.join(redact_argv(command)),
    }
    return "".join(f"{name}={value}\n" for name, value in values.items())


def run(
    *,
    repo_root: Path,
    runs_root: Path,
    suite: str,
    run_id: str,
    python: Path,
    training_args: list[str],
    nproc_per_node: int | None = None,
    gpu_ids_raw: str | None = None,
    monitor_gpu: bool = False,
    gpu_sample_interval: float = 5.0,
    termination_grace_seconds: float = 30.0,
    dry_run: bool = False,
) -> int:
    launch_started_monotonic = time.monotonic()
    repo_root = _absolute(repo_root)
    paths = resolve_run_paths(runs_root, suite, run_id)
    gpu_ids = parse_gpu_ids(gpu_ids_raw)
    validate_training_args(training_args, nproc_per_node)
    if nproc_per_node is not None and nproc_per_node < 1:
        raise ValueError("nproc-per-node must be at least 1")
    if nproc_per_node is not None and gpu_ids and len(gpu_ids) != nproc_per_node:
        raise ValueError("gpu-ids count must equal nproc-per-node")
    if monitor_gpu and not gpu_ids:
        raise ValueError("--monitor-gpu requires --gpu-ids")
    if gpu_sample_interval <= 0:
        raise ValueError("gpu-sample-interval must be positive")
    if termination_grace_seconds < 0:
        raise ValueError("termination-grace-seconds may not be negative")

    command = build_child_command(
        python,
        repo_root,
        paths,
        run_id,
        training_args,
        nproc_per_node,
    )
    if dry_run:
        print(
            json.dumps(
                {
                    "run_dir": str(paths.run_dir),
                    "command": redact_argv(command),
                    "cuda_visible_devices": gpu_ids or None,
                    "monitor_gpu": monitor_gpu,
                },
                indent=2,
            )
        )
        return 0

    if not paths.runs_root.is_dir():
        raise ValueError(
            f"runs-root does not exist; run setup_runtime_paths.py first: {paths.runs_root}"
        )
    monitor: GpuMonitor | None = None
    initial_gpu_sample: str | None = None
    process: subprocess.Popen[bytes] | None = None
    received_signal: int | None = None
    signal_received_at: float | None = None
    sent_sigkill = False
    descendant_cleanup_started_at: float | None = None
    descendant_cleanup_needed = False
    exit_code = 2

    def handle_signal(signum: int, _frame: Any) -> None:
        nonlocal received_signal, signal_received_at
        if received_signal is None:
            received_signal = signum
            signal_received_at = time.monotonic()
        if process is not None:
            _signal_process_group(process.pid, signum)

    previous_handlers = {
        signum: signal.getsignal(signum) for signum in (signal.SIGINT, signal.SIGTERM)
    }
    for signum in previous_handlers:
        signal.signal(signum, handle_signal)

    try:
        paths.run_dir.mkdir(parents=True, exist_ok=False)
        paths.launch_dir.mkdir()
        paths.logs_dir.mkdir()

        started_at = utc_now()
        attempt_id = uuid.uuid4().hex
        status_path = paths.launch_dir / "status.json"
        status: dict[str, Any] = {
            "schema_version": 1,
            "attempt_id": attempt_id,
            "state": "starting",
            "run_id": run_id,
            "suite": suite,
            "run_dir": str(paths.run_dir),
            "repo_root": str(repo_root),
            "launcher_pid": os.getpid(),
            "child_pid": None,
            "started_at_utc": started_at,
            "ended_at_utc": None,
            "duration_seconds": None,
            "return_code": None,
            "received_signal": None,
            "artifacts_verified": False,
            "failure_reason": None,
            "observability_complete": not monitor_gpu,
            "gpu_monitor": {
                "enabled": False,
                "gpu_ids": gpu_ids,
                "state": "disabled",
                "sample_failures": 0,
            },
        }
        write_text_atomic(
            paths.launch_dir / "command.txt",
            _command_record(
                command=command,
                run_id=run_id,
                suite=suite,
                repo_root=repo_root,
                gpu_ids=gpu_ids,
                started_at=started_at,
            ),
        )
        write_json_atomic(status_path, status)
    except Exception:
        for signum, handler in previous_handlers.items():
            signal.signal(signum, handler)
        raise

    try:
        if received_signal is not None:
            status["state"] = "interrupted"
            status["received_signal"] = received_signal
            status["return_code"] = 128 + received_signal
            status["failure_reason"] = (
                f"launcher received signal {received_signal} before training start"
            )
            exit_code = 128 + received_signal
            raise _StopLaunch
        if monitor_gpu:
            monitor = GpuMonitor(
                gpu_ids,
                paths.logs_dir / "gpu_smi.csv",
                gpu_sample_interval,
            )
            initial_gpu_sample = monitor.preflight()
        if received_signal is not None:
            if monitor is not None:
                status["gpu_monitor"] = monitor.snapshot()
            status["state"] = "interrupted"
            status["received_signal"] = received_signal
            status["return_code"] = 128 + received_signal
            status["failure_reason"] = (
                f"launcher received signal {received_signal} before training start"
            )
            exit_code = 128 + received_signal
            raise _StopLaunch

        environment = dict(os.environ)
        environment["PYTHONUNBUFFERED"] = "1"
        if gpu_ids:
            environment["CUDA_VISIBLE_DEVICES"] = ",".join(gpu_ids)
        with (paths.logs_dir / "stdout.log").open("wb") as output:
            process = subprocess.Popen(
                command,
                cwd=repo_root,
                env=environment,
                stdin=subprocess.DEVNULL,
                stdout=output,
                stderr=subprocess.STDOUT,
                start_new_session=True,
            )
            if received_signal is not None:
                _signal_process_group(process.pid, received_signal)
            status["state"] = "running"
            status["child_pid"] = process.pid
            if monitor is not None and initial_gpu_sample is not None:
                monitor.start(initial_gpu_sample)
                status["gpu_monitor"] = monitor.snapshot()
            write_json_atomic(status_path, status)

            while True:
                child_return_code = process.poll()
                group_exists = _process_group_exists(process.pid)
                now = time.monotonic()
                if received_signal is not None:
                    if not group_exists and child_return_code is not None:
                        break
                    if (
                        signal_received_at is not None
                        and not sent_sigkill
                        and now - signal_received_at > termination_grace_seconds
                    ):
                        _signal_process_group(process.pid, signal.SIGKILL)
                        sent_sigkill = True
                    if sent_sigkill and child_return_code is not None:
                        break
                elif child_return_code is not None:
                    if not group_exists:
                        break
                    descendant_cleanup_needed = True
                    if descendant_cleanup_started_at is None:
                        descendant_cleanup_started_at = now
                        _signal_process_group(process.pid, signal.SIGTERM)
                    elif now - descendant_cleanup_started_at > termination_grace_seconds:
                        _signal_process_group(process.pid, signal.SIGKILL)
                        break
                time.sleep(0.1)
            child_return_code = process.wait()

        if monitor is not None:
            monitor.stop()
            status["gpu_monitor"] = monitor.snapshot()
            status["observability_complete"] = (
                status["gpu_monitor"]["state"] == "completed"
            )

        if received_signal is not None:
            exit_code = 128 + received_signal
            status["state"] = "interrupted"
            status["received_signal"] = received_signal
            status["failure_reason"] = f"launcher received signal {received_signal}"
        else:
            exit_code = _normalize_return_code(child_return_code)
            if exit_code == 0:
                verification = verify_completed_artifacts(paths.run_dir)
                status["artifacts_verified"] = verification.ok
                if descendant_cleanup_needed:
                    status["state"] = "failed"
                    status["failure_reason"] = (
                        "descendant processes outlived the training launcher"
                    )
                    exit_code = 2
                elif verification.ok:
                    status["state"] = "completed"
                else:
                    status["state"] = "failed"
                    status["failure_reason"] = verification.reason
                    exit_code = 2
            else:
                status["state"] = "failed"
                status["failure_reason"] = f"training process exited with {exit_code}"
        status["return_code"] = exit_code
    except _StopLaunch:
        pass
    except Exception as exc:
        if process is not None and _process_group_exists(process.pid):
            _terminate_process_group(process, termination_grace_seconds)
        elif process is not None:
            try:
                process.wait(timeout=1.0)
            except subprocess.TimeoutExpired:
                pass
        if monitor is not None:
            monitor.stop()
            status["gpu_monitor"] = monitor.snapshot()
        status["state"] = "failed"
        status["return_code"] = 2
        status["failure_reason"] = f"{type(exc).__name__}: {exc}"
        exit_code = 2
    finally:
        for signum, handler in previous_handlers.items():
            signal.signal(signum, handler)
        status["ended_at_utc"] = utc_now()
        status["duration_seconds"] = round(
            time.monotonic() - launch_started_monotonic, 3
        )
        try:
            write_json_atomic(status_path, status)
        except Exception as exc:
            print(f"could not write final launch status: {exc}", file=sys.stderr)
            exit_code = 2
    return exit_code


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--runs-root", type=Path, required=True)
    parser.add_argument("--suite", required=True)
    parser.add_argument("--run-id", required=True)
    parser.add_argument(
        "--repo-root",
        type=Path,
        default=Path(__file__).resolve().parents[1],
    )
    parser.add_argument("--python", type=Path, default=Path(sys.executable))
    parser.add_argument("--gpu-ids")
    parser.add_argument("--nproc-per-node", type=int)
    parser.add_argument("--monitor-gpu", action="store_true")
    parser.add_argument("--gpu-sample-interval", type=float, default=5.0)
    parser.add_argument("--termination-grace-seconds", type=float, default=30.0)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("training_args", nargs=argparse.REMAINDER)
    args = parser.parse_args(argv)
    if args.training_args and args.training_args[0] == "--":
        args.training_args = args.training_args[1:]
    return args


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    try:
        return run(
            repo_root=args.repo_root,
            runs_root=args.runs_root,
            suite=args.suite,
            run_id=args.run_id,
            python=args.python,
            training_args=args.training_args,
            nproc_per_node=args.nproc_per_node,
            gpu_ids_raw=args.gpu_ids,
            monitor_gpu=args.monitor_gpu,
            gpu_sample_interval=args.gpu_sample_interval,
            termination_grace_seconds=args.termination_grace_seconds,
            dry_run=args.dry_run,
        )
    except (OSError, ValueError) as exc:
        print(f"run launch rejected: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
