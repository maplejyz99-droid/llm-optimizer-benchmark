import json
import os
import signal
import subprocess
import sys
import tempfile
import time
import unittest
from contextlib import redirect_stderr, redirect_stdout
from io import StringIO
from pathlib import Path
from unittest.mock import Mock, patch

from scripts import launch_run


REPO_ROOT = Path(__file__).resolve().parents[2]
LAUNCHER_PATH = REPO_ROOT / "scripts" / "launch_run.py"


FAKE_MAIN = r"""#!/usr/bin/env python3
import argparse
import hashlib
import json
import os
import signal
import subprocess
import sys
import time
from pathlib import Path

parser = argparse.ArgumentParser(add_help=False)
parser.add_argument("--results_base_folder", required=True)
parser.add_argument("--experiment_name", required=True)
parser.add_argument("--fake_mode", default="success")
args, remainder = parser.parse_known_args()
run_dir = Path(args.results_base_folder) / args.experiment_name
run_dir.mkdir(parents=True, exist_ok=True)
(run_dir / "received.json").write_text(
    json.dumps({"argv": sys.argv[1:], "remainder": remainder}), encoding="utf-8"
)

if args.fake_mode == "fail":
    raise SystemExit(7)
if args.fake_mode == "missing":
    raise SystemExit(0)
if args.fake_mode in {"sleep", "stubborn"}:
    child_marker = run_dir / "child.signal"
    grandchild_marker = run_dir / "grandchild.signal"
    grandchild_ready = run_dir / "grandchild.ready"
    heartbeat = run_dir / "grandchild.heartbeat"
    grandchild_code = '''
import signal
import sys
import time
from pathlib import Path

marker = Path(sys.argv[1])
ready = Path(sys.argv[2])
heartbeat = Path(sys.argv[3])
mode = sys.argv[4]
def handle(signum, _frame):
    marker.write_text(str(signum), encoding="utf-8")
    raise SystemExit(0)
if mode == "stubborn":
    signal.signal(signal.SIGTERM, signal.SIG_IGN)
else:
    signal.signal(signal.SIGTERM, handle)
signal.signal(signal.SIGINT, handle)
ready.write_text("ready", encoding="utf-8")
while True:
    heartbeat.write_text(str(time.monotonic()), encoding="utf-8")
    time.sleep(0.02)
'''
    def handle(signum, _frame):
        child_marker.write_text(str(signum), encoding="utf-8")
        raise SystemExit(0)
    signal.signal(signal.SIGTERM, handle)
    signal.signal(signal.SIGINT, handle)
    grandchild = subprocess.Popen(
        [
            sys.executable,
            "-c",
            grandchild_code,
            str(grandchild_marker),
            str(grandchild_ready),
            str(heartbeat),
            args.fake_mode,
        ]
    )
    deadline = time.monotonic() + 5
    while time.monotonic() < deadline and not grandchild_ready.exists():
        time.sleep(0.01)
    if not grandchild_ready.exists():
        raise RuntimeError("grandchild did not become ready")
    (run_dir / "processes.json").write_text(
        json.dumps({"child": os.getpid(), "grandchild": grandchild.pid}),
        encoding="utf-8",
    )
    time.sleep(60)
    raise SystemExit(0)

def canonical_sha256(payload):
    encoded = json.dumps(
        payload,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()

evaluation_protocol = {
    "periodic": {"mode": "batch_cap", "max_batches": 64},
    "final_and_full": {"mode": "full_dataset"},
    "full_eval_at": [],
    "generation_prefix": "none",
    "token_counting": "targets_not_equal_to_-1",
}
protocol_id = canonical_sha256(evaluation_protocol)
evaluation_protocol["identity"] = protocol_id

realization = {
    "status": "resolved",
    "update_routes": [],
    "overlays": [],
    "parameter_groups": [],
    "coverage": {
        "missing_from_optimizer_tensors": 0,
        "unassigned_tensors": 0,
        "multiply_assigned_tensors": 0,
    },
    "optimizer_class": "FakeOptimizer",
}
realization["identity"] = canonical_sha256(
    {
        key: value
        for key, value in realization.items()
        if key not in {"status", "identity"}
    }
)
optimization_plan = {
    "schema_version": 1,
    "strategy": {"id": "fake-adamw-v1", "kind": "single"},
    "components": [
        {
            "id": "parameter_update",
            "role": "primary",
            "algorithm": "adamw",
            "hyperparameters": {"lr": 0.001},
        }
    ],
    "scheduler": {"kind": "none", "algorithm": "none"},
    "gradient_processing": {"gradient_clipping": {"max_norm": 0.5}},
    "routing": {
        "update_routes": [],
        "overlays": [],
        "realization": realization,
    },
}
optimization_plan["identity"] = canonical_sha256(
    {
        "schema_version": optimization_plan["schema_version"],
        "strategy": optimization_plan["strategy"],
        "components": optimization_plan["components"],
        "scheduler": optimization_plan["scheduler"],
        "gradient_processing": optimization_plan["gradient_processing"],
        "routing": {
            "update_routes": optimization_plan["routing"]["update_routes"],
            "overlays": optimization_plan["routing"]["overlays"],
        },
    }
)
preflight_identity = "fake-preflight-identity"
identity = canonical_sha256(
    {
        "schema_version": 3,
        "preflight_identity": preflight_identity,
        "optimization_realization_identity": realization["identity"],
    }
)
manifest = {
    "schema_version": 3,
    "manifest_state": "resolved",
    "preflight_identity": preflight_identity,
    "run_identity": identity,
    "evaluation_protocol": evaluation_protocol,
    "optimization_plan": optimization_plan,
}
summary = {
    "run_identity": identity,
    "evaluation_protocol": evaluation_protocol,
    "optimization_plan": optimization_plan,
}
(run_dir / "run_manifest.json").write_text(json.dumps(manifest), encoding="utf-8")
(run_dir / "summary.json").write_text(json.dumps(summary), encoding="utf-8")
evaluation_dir = run_dir / "evaluations" / protocol_id
evaluation_dir.mkdir(parents=True)
(evaluation_dir / "summary.json").write_text(json.dumps(summary), encoding="utf-8")
"""


class RunLauncherBehaviorTest(unittest.TestCase):
    def _fixture(self, root: Path):
        repo = root / "fake repo"
        (repo / "src").mkdir(parents=True)
        (repo / "src" / "main.py").write_text(FAKE_MAIN, encoding="utf-8")
        runs = root / "runs"
        runs.mkdir()
        return repo, runs

    def _run(self, repo: Path, runs: Path, run_id="run-one", **overrides):
        options = {
            "repo_root": repo,
            "runs_root": runs,
            "suite": "tests/smoke",
            "run_id": run_id,
            "python": Path(sys.executable),
            "training_args": ["--fake_mode", "success"],
            "termination_grace_seconds": 0.5,
        }
        options.update(overrides)
        return launch_run.run(**options)

    def test_success_creates_one_self_contained_run_bundle(self):
        with tempfile.TemporaryDirectory() as tmp_dir:
            root = Path(tmp_dir)
            repo, runs = self._fixture(root)

            return_code = self._run(
                repo,
                runs,
                training_args=["--fake_mode", "success", "--value", "with spaces"],
            )

            run_dir = runs / "tests" / "smoke" / "run-one"
            status = json.loads((run_dir / "launch" / "status.json").read_text())
            received = json.loads((run_dir / "received.json").read_text())
            self.assertEqual(return_code, 0)
            self.assertEqual(status["state"], "completed")
            self.assertEqual(status["return_code"], 0)
            self.assertTrue(status["artifacts_verified"])
            self.assertGreaterEqual(status["duration_seconds"], 0)
            self.assertTrue((run_dir / "logs" / "stdout.log").is_file())
            self.assertTrue((run_dir / "launch" / "command.txt").is_file())
            value_index = received["argv"].index("--value")
            self.assertEqual(received["argv"][value_index + 1], "with spaces")

    def test_child_failure_and_missing_artifacts_are_not_success(self):
        with tempfile.TemporaryDirectory() as tmp_dir:
            root = Path(tmp_dir)
            repo, runs = self._fixture(root)

            failed = self._run(
                repo,
                runs,
                run_id="failed",
                training_args=["--fake_mode", "fail"],
            )
            missing = self._run(
                repo,
                runs,
                run_id="missing",
                training_args=["--fake_mode", "missing"],
            )

            failed_status = json.loads(
                (runs / "tests/smoke/failed/launch/status.json").read_text()
            )
            missing_status = json.loads(
                (runs / "tests/smoke/missing/launch/status.json").read_text()
            )
            self.assertEqual(failed, 7)
            self.assertEqual(failed_status["state"], "failed")
            self.assertEqual(missing, 2)
            self.assertEqual(missing_status["state"], "failed")
            self.assertIn("artifact", missing_status["failure_reason"])

    def test_output_ownership_resume_and_path_traversal_are_rejected(self):
        with tempfile.TemporaryDirectory() as tmp_dir:
            root = Path(tmp_dir)
            repo, runs = self._fixture(root)
            for argument in (
                ["--results_base_folder", "/tmp/escape"],
                ["--experiment_name=escape"],
                ["--resume_from", "/tmp/checkpoint"],
            ):
                with self.subTest(argument=argument), self.assertRaises(ValueError):
                    self._run(repo, runs, training_args=argument)

            for suite, run_id in (("../escape", "safe"), ("safe", "../escape")):
                with self.subTest(suite=suite, run_id=run_id), self.assertRaises(
                    ValueError
                ):
                    self._run(repo, runs, suite=suite, run_id=run_id)

            self.assertEqual(list(runs.iterdir()), [])

    def test_existing_run_is_never_overwritten(self):
        with tempfile.TemporaryDirectory() as tmp_dir:
            root = Path(tmp_dir)
            repo, runs = self._fixture(root)
            self.assertEqual(self._run(repo, runs), 0)
            command_before = (runs / "tests/smoke/run-one/launch/command.txt").read_bytes()

            with self.assertRaises(FileExistsError):
                self._run(repo, runs)

            self.assertEqual(
                (runs / "tests/smoke/run-one/launch/command.txt").read_bytes(),
                command_before,
            )

    def test_dry_run_writes_nothing_and_redacts_credentials(self):
        secret = "launch-secret-value"
        with tempfile.TemporaryDirectory() as tmp_dir:
            root = Path(tmp_dir)
            repo, runs = self._fixture(root)
            output = StringIO()
            with redirect_stdout(output):
                return_code = self._run(
                    repo,
                    runs,
                    dry_run=True,
                    training_args=[
                        "--notify_smtp_pass",
                        secret,
                        "--notify_interval",
                        "5",
                        f"--api-key={secret}",
                        "--tokenizer",
                        "gpt2",
                    ],
                )

            self.assertEqual(return_code, 0)
            self.assertNotIn(secret, output.getvalue())
            self.assertIn("<redacted>", output.getvalue())
            self.assertIn("tokenizer", output.getvalue())
            self.assertEqual(list(runs.iterdir()), [])

    def test_recorded_command_is_redacted_but_child_receives_real_value(self):
        secret = "real-secret-for-child"
        with tempfile.TemporaryDirectory() as tmp_dir:
            root = Path(tmp_dir)
            repo, runs = self._fixture(root)
            return_code = self._run(
                repo,
                runs,
                training_args=["--fake_mode", "success", "--notify_webhook", secret],
            )

            run_dir = runs / "tests/smoke/run-one"
            command = (run_dir / "launch/command.txt").read_text()
            received = (run_dir / "received.json").read_text()
            status = (run_dir / "launch/status.json").read_text()
            self.assertEqual(return_code, 0)
            self.assertNotIn(secret, command)
            self.assertNotIn(secret, status)
            self.assertIn("<redacted>", command)
            self.assertIn(secret, received)

    def test_inherited_cuda_visible_devices_is_recorded(self):
        with tempfile.TemporaryDirectory() as tmp_dir:
            root = Path(tmp_dir)
            repo, runs = self._fixture(root)
            with patch.dict(os.environ, {"CUDA_VISIBLE_DEVICES": "7"}):
                return_code = self._run(repo, runs)

            command = (runs / "tests/smoke/run-one/launch/command.txt").read_text()
            self.assertEqual(return_code, 0)
            self.assertIn("cuda_visible_devices=7", command)

    def test_completed_verification_requires_schema_v3_resolved_manifest(self):
        with tempfile.TemporaryDirectory() as tmp_dir:
            root = Path(tmp_dir)
            repo, runs = self._fixture(root)
            self.assertEqual(self._run(repo, runs), 0)
            run_dir = runs / "tests/smoke/run-one"
            manifest_path = run_dir / "run_manifest.json"
            original = json.loads(manifest_path.read_text())

            for field, value, expected_reason in (
                ("schema_version", 2, "manifest schema mismatch"),
                ("manifest_state", "preflight", "manifest is not resolved"),
            ):
                with self.subTest(field=field, value=value):
                    manifest = json.loads(json.dumps(original))
                    manifest[field] = value
                    manifest_path.write_text(
                        json.dumps(manifest), encoding="utf-8"
                    )

                    verification = launch_run.verify_completed_artifacts(run_dir)
                    self.assertFalse(verification.ok)
                    self.assertIn(expected_reason, verification.reason)

    def test_completed_verification_requires_resolved_plan_identities(self):
        with tempfile.TemporaryDirectory() as tmp_dir:
            root = Path(tmp_dir)
            repo, runs = self._fixture(root)
            self.assertEqual(self._run(repo, runs), 0)
            run_dir = runs / "tests/smoke/run-one"
            manifest_path = run_dir / "run_manifest.json"
            original = json.loads(manifest_path.read_text())

            cases = (
                (
                    ("optimization_plan", "schema_version"),
                    2,
                    "optimization plan schema mismatch",
                ),
                (
                    ("optimization_plan", "identity"),
                    "",
                    "no optimization plan identity",
                ),
                (
                    (
                        "optimization_plan",
                        "routing",
                        "realization",
                        "status",
                    ),
                    "pending",
                    "optimization plan is not resolved",
                ),
                (
                    (
                        "optimization_plan",
                        "routing",
                        "realization",
                        "identity",
                    ),
                    None,
                    "no optimization realization identity",
                ),
            )
            for path, value, expected_reason in cases:
                with self.subTest(path=path):
                    manifest = json.loads(json.dumps(original))
                    target = manifest
                    for component in path[:-1]:
                        target = target[component]
                    target[path[-1]] = value
                    manifest_path.write_text(
                        json.dumps(manifest), encoding="utf-8"
                    )

                    verification = launch_run.verify_completed_artifacts(run_dir)
                    self.assertFalse(verification.ok)
                    self.assertIn(expected_reason, verification.reason)

    def test_identity_contract_must_match_summary_and_versioned_evaluation(self):
        with tempfile.TemporaryDirectory() as tmp_dir:
            root = Path(tmp_dir)
            repo, runs = self._fixture(root)
            self.assertEqual(self._run(repo, runs), 0)
            run_dir = runs / "tests/smoke/run-one"
            manifest_path = run_dir / "run_manifest.json"
            summary_path = run_dir / "summary.json"
            original_manifest = json.loads(manifest_path.read_text())
            evaluation_path = (
                run_dir
                / "evaluations"
                / original_manifest["evaluation_protocol"]["identity"]
                / "summary.json"
            )
            original_summary = json.loads(summary_path.read_text())
            original_evaluation = json.loads(evaluation_path.read_text())

            cases = (
                (
                    summary_path,
                    ("run_identity",),
                    "different-run",
                    "summary run_identity mismatch",
                ),
                (
                    summary_path,
                    ("evaluation_protocol", "identity"),
                    "different-protocol",
                    "summary evaluation protocol identity does not match canonical content",
                ),
                (
                    summary_path,
                    ("optimization_plan", "identity"),
                    "different-plan",
                    "summary optimization plan identity does not match canonical intent",
                ),
                (
                    summary_path,
                    (
                        "optimization_plan",
                        "routing",
                        "realization",
                        "identity",
                    ),
                    "different-realization",
                    "summary optimization realization identity does not match canonical content",
                ),
                (
                    evaluation_path,
                    ("run_identity",),
                    "different-run",
                    "evaluation run_identity mismatch",
                ),
                (
                    evaluation_path,
                    ("evaluation_protocol", "identity"),
                    "different-protocol",
                    "evaluation protocol identity does not match canonical content",
                ),
                (
                    evaluation_path,
                    ("optimization_plan", "identity"),
                    "different-plan",
                    "evaluation optimization plan identity does not match canonical intent",
                ),
                (
                    evaluation_path,
                    (
                        "optimization_plan",
                        "routing",
                        "realization",
                        "identity",
                    ),
                    "different-realization",
                    "evaluation optimization realization identity does not match canonical content",
                ),
            )
            for path, field_path, value, expected_reason in cases:
                with self.subTest(path=path.name, field_path=field_path):
                    summary_path.write_text(
                        json.dumps(original_summary), encoding="utf-8"
                    )
                    evaluation_path.write_text(
                        json.dumps(original_evaluation), encoding="utf-8"
                    )
                    payload = json.loads(path.read_text())
                    target = payload
                    for component in field_path[:-1]:
                        target = target[component]
                    target[field_path[-1]] = value
                    path.write_text(json.dumps(payload), encoding="utf-8")

                    verification = launch_run.verify_completed_artifacts(run_dir)
                    self.assertFalse(verification.ok)
                    self.assertIn(expected_reason, verification.reason)

            for path, original in (
                (manifest_path, original_manifest),
                (summary_path, original_summary),
                (evaluation_path, original_evaluation),
            ):
                payload = json.loads(json.dumps(original))
                payload["optimization_plan"]["components"][0][
                    "hyperparameters"
                ]["lr"] = 0.5
                path.write_text(json.dumps(payload), encoding="utf-8")
            verification = launch_run.verify_completed_artifacts(run_dir)
            self.assertFalse(verification.ok)
            self.assertIn(
                "manifest optimization plan identity does not match canonical intent",
                verification.reason,
            )

            for path, original in (
                (manifest_path, original_manifest),
                (summary_path, original_summary),
                (evaluation_path, original_evaluation),
            ):
                payload = json.loads(json.dumps(original))
                payload["evaluation_protocol"]["generation_prefix"] = "tampered"
                path.write_text(json.dumps(payload), encoding="utf-8")
            verification = launch_run.verify_completed_artifacts(run_dir)
            self.assertFalse(verification.ok)
            self.assertIn(
                "manifest evaluation protocol identity does not match canonical content",
                verification.reason,
            )

            for path, original in (
                (manifest_path, original_manifest),
                (summary_path, original_summary),
                (evaluation_path, original_evaluation),
            ):
                payload = json.loads(json.dumps(original))
                realization = payload["optimization_plan"]["routing"][
                    "realization"
                ]
                realization["optimizer_class"] = "ChangedOptimizer"
                realization["identity"] = launch_run._canonical_sha256(
                    {
                        key: value
                        for key, value in realization.items()
                        if key not in {"status", "identity"}
                    }
                )
                path.write_text(json.dumps(payload), encoding="utf-8")
            verification = launch_run.verify_completed_artifacts(run_dir)
            self.assertFalse(verification.ok)
            self.assertIn(
                "manifest run_identity does not match canonical resolved identity",
                verification.reason,
            )

    def test_final_status_write_failure_changes_launcher_exit_code(self):
        with tempfile.TemporaryDirectory() as tmp_dir:
            root = Path(tmp_dir)
            repo, runs = self._fixture(root)
            original_write = launch_run.write_json_atomic

            def fail_completed_status(path, payload):
                if path.name == "status.json" and payload.get("state") == "completed":
                    raise OSError("simulated final status failure")
                return original_write(path, payload)

            errors = StringIO()
            with patch.object(
                launch_run, "write_json_atomic", side_effect=fail_completed_status
            ), redirect_stderr(errors):
                return_code = self._run(repo, runs)

            self.assertEqual(return_code, 2)
            self.assertIn("could not write final launch status", errors.getvalue())

    def test_monitor_is_not_touched_unless_explicitly_enabled(self):
        with tempfile.TemporaryDirectory() as tmp_dir:
            root = Path(tmp_dir)
            repo, runs = self._fixture(root)
            with patch.object(launch_run, "GpuMonitor") as monitor:
                return_code = self._run(repo, runs)

            self.assertEqual(return_code, 0)
            monitor.assert_not_called()

    def test_monitor_preflight_failure_prevents_training(self):
        with tempfile.TemporaryDirectory() as tmp_dir:
            root = Path(tmp_dir)
            repo, runs = self._fixture(root)
            with patch.object(launch_run.GpuMonitor, "preflight", side_effect=RuntimeError("no gpu")):
                return_code = self._run(
                    repo,
                    runs,
                    gpu_ids_raw="0",
                    monitor_gpu=True,
                )

            status = json.loads(
                (runs / "tests/smoke/run-one/launch/status.json").read_text()
            )
            self.assertEqual(return_code, 2)
            self.assertEqual(status["state"], "failed")
            self.assertIn("no gpu", status["failure_reason"])
            self.assertFalse((runs / "tests/smoke/run-one/received.json").exists())

    def test_monitor_shutdown_timeout_is_not_reported_complete(self):
        with tempfile.TemporaryDirectory() as tmp_dir:
            monitor = launch_run.GpuMonitor(
                ["0"], Path(tmp_dir) / "gpu_smi.csv", interval=0.01
            )
            monitor._state = "running"
            monitor._thread = Mock()
            monitor._thread.is_alive.return_value = True

            monitor.stop()

            snapshot = monitor.snapshot()
            self.assertEqual(snapshot["state"], "shutdown_timeout")
            monitor._thread.join.assert_called_once()

    def test_monitor_rejects_successful_nvidia_smi_with_no_gpu_rows(self):
        with tempfile.TemporaryDirectory() as tmp_dir:
            monitor = launch_run.GpuMonitor(
                ["0"], Path(tmp_dir) / "gpu_smi.csv", interval=1
            )
            empty_result = Mock(returncode=0, stdout="", stderr="")
            with patch.object(launch_run.subprocess, "run", return_value=empty_result):
                with self.assertRaisesRegex(RuntimeError, "no GPU rows"):
                    monitor.preflight()

    def test_signal_during_preflight_does_not_start_training(self):
        with tempfile.TemporaryDirectory() as tmp_dir:
            root = Path(tmp_dir)
            repo, runs = self._fixture(root)

            def interrupt_preflight(_monitor):
                os.kill(os.getpid(), signal.SIGTERM)
                return "unused"

            with patch.object(
                launch_run.GpuMonitor, "preflight", interrupt_preflight
            ):
                return_code = self._run(
                    repo,
                    runs,
                    gpu_ids_raw="0",
                    monitor_gpu=True,
                )

            run_dir = runs / "tests/smoke/run-one"
            status = json.loads((run_dir / "launch/status.json").read_text())
            self.assertEqual(return_code, 143)
            self.assertEqual(status["state"], "interrupted")
            self.assertEqual(status["received_signal"], signal.SIGTERM)
            self.assertFalse((run_dir / "received.json").exists())

    def test_distributed_launcher_contract_is_explicit(self):
        with tempfile.TemporaryDirectory() as tmp_dir:
            root = Path(tmp_dir)
            repo, runs = self._fixture(root)
            paths = launch_run.resolve_run_paths(runs, "tests", "distributed")
            args = ["--distributed_backend", "nccl"]

            with self.assertRaisesRegex(ValueError, "nproc-per-node"):
                launch_run.validate_training_args(args, None)
            with self.assertRaisesRegex(ValueError, "distributed_backend"):
                launch_run.validate_training_args(["--model", "llama"], 2)

            launch_run.validate_training_args(args, 2)
            command = launch_run.build_child_command(
                Path(sys.executable), repo, paths, "distributed", args, 2
            )
            self.assertIn("torch.distributed.run", command)
            self.assertIn("--nproc_per_node=2", command)

    def test_sigterm_is_forwarded_to_the_child_process_group(self):
        with tempfile.TemporaryDirectory() as tmp_dir:
            root = Path(tmp_dir)
            repo, runs = self._fixture(root)
            command = [
                sys.executable,
                str(LAUNCHER_PATH),
                "--repo-root",
                str(repo),
                "--runs-root",
                str(runs),
                "--suite",
                "tests/signal",
                "--run-id",
                "signal-run",
                "--termination-grace-seconds",
                "0.5",
                "--",
                "--fake_mode",
                "sleep",
            ]
            launcher = subprocess.Popen(
                command,
                cwd=REPO_ROOT,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
            )
            run_dir = runs / "tests/signal/signal-run"
            processes_path = run_dir / "processes.json"
            deadline = time.monotonic() + 10
            while time.monotonic() < deadline and not processes_path.exists():
                if launcher.poll() is not None:
                    break
                time.sleep(0.05)
            self.assertTrue(processes_path.exists())
            process_ids = json.loads(processes_path.read_text())

            launcher.send_signal(signal.SIGTERM)
            stdout, stderr = launcher.communicate(timeout=10)
            status = json.loads((run_dir / "launch/status.json").read_text())
            self.assertEqual(launcher.returncode, 143, msg=stdout + stderr)
            self.assertEqual(status["state"], "interrupted")
            self.assertEqual(status["received_signal"], signal.SIGTERM)
            self.assertEqual(
                (run_dir / "child.signal").read_text(), str(int(signal.SIGTERM))
            )
            self.assertEqual(
                (run_dir / "grandchild.signal").read_text(),
                str(int(signal.SIGTERM)),
            )

    def test_stubborn_grandchild_is_killed_after_signal_grace_period(self):
        with tempfile.TemporaryDirectory() as tmp_dir:
            root = Path(tmp_dir)
            repo, runs = self._fixture(root)
            command = [
                sys.executable,
                str(LAUNCHER_PATH),
                "--repo-root",
                str(repo),
                "--runs-root",
                str(runs),
                "--suite",
                "tests/signal",
                "--run-id",
                "stubborn-run",
                "--termination-grace-seconds",
                "0.2",
                "--",
                "--fake_mode",
                "stubborn",
            ]
            launcher = subprocess.Popen(
                command,
                cwd=REPO_ROOT,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
            )
            run_dir = runs / "tests/signal/stubborn-run"
            processes_path = run_dir / "processes.json"
            deadline = time.monotonic() + 10
            while time.monotonic() < deadline and not processes_path.exists():
                if launcher.poll() is not None:
                    break
                time.sleep(0.05)
            self.assertTrue(processes_path.exists())
            process_ids = json.loads(processes_path.read_text())
            self.addCleanup(self._kill_process, process_ids["grandchild"])

            launcher.send_signal(signal.SIGTERM)
            stdout, stderr = launcher.communicate(timeout=10)
            heartbeat = run_dir / "grandchild.heartbeat"
            self.assertTrue(heartbeat.exists())
            heartbeat_after_exit = heartbeat.read_text()
            time.sleep(0.2)

            status = json.loads((run_dir / "launch/status.json").read_text())
            self.assertEqual(launcher.returncode, 143, msg=stdout + stderr)
            self.assertEqual(status["state"], "interrupted")
            self.assertEqual(heartbeat.read_text(), heartbeat_after_exit)
            self.assertEqual(
                (run_dir / "child.signal").read_text(), str(int(signal.SIGTERM))
            )
            self.assertFalse((run_dir / "grandchild.signal").exists())

    @staticmethod
    def _kill_process(process_id: int) -> None:
        try:
            os.kill(process_id, signal.SIGKILL)
        except ProcessLookupError:
            pass


if __name__ == "__main__":
    unittest.main()
