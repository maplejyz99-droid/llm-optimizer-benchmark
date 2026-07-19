import json
import tempfile
import unittest
from contextlib import redirect_stdout
from io import StringIO
from pathlib import Path

from tests._helpers.behavior_harness import (
    load_main_with_fakes,
    make_args_for_main,
)


class MainLifecycleTest(unittest.TestCase):
    def run_main(self, capture, argv=None):
        module = load_main_with_fakes(capture)
        args, parser = make_args_for_main(argv or [])
        stdout = StringIO()
        with redirect_stdout(stdout):
            module.main(args, parser)
        capture["stdout"] = stdout.getvalue()
        return capture["backend"]

    def test_backend_is_finalized_once_after_success(self):
        capture = {}
        backend = self.run_main(capture)

        self.assertTrue(backend.finalized)
        self.assertEqual(backend.finalize_count, 1)

    def test_backend_is_finalized_once_when_training_raises(self):
        capture = {"train_error": RuntimeError("synthetic train failure")}
        with self.assertRaisesRegex(RuntimeError, "synthetic train failure"):
            self.run_main(capture)

        backend = capture["backend"]
        self.assertTrue(backend.finalized)
        self.assertEqual(backend.finalize_count, 1)

    def test_training_error_remains_primary_when_finalize_also_raises(self):
        capture = {
            "train_error": ValueError("primary training failure"),
            "finalize_error": RuntimeError("secondary finalize failure"),
        }

        with self.assertRaisesRegex(ValueError, "primary training failure") as raised:
            self.run_main(capture)

        self.assertIsInstance(raised.exception.__cause__, RuntimeError)
        self.assertRegex(str(raised.exception.__cause__), "secondary finalize failure")
        self.assertEqual(capture["backend"].finalize_count, 1)

    def test_finalize_error_is_reported_after_successful_training(self):
        capture = {
            "finalize_error": RuntimeError("standalone finalize failure"),
        }

        with self.assertRaisesRegex(RuntimeError, "standalone finalize failure"):
            self.run_main(capture)

        self.assertEqual(capture["backend"].finalize_count, 1)

    def test_gn_rejects_distributed_backend_before_initialization(self):
        capture = {}
        module = load_main_with_fakes(capture)
        args, parser = make_args_for_main(
            ["--opt", "gn-prox", "--distributed_backend", "nccl"]
        )

        with self.assertRaisesRegex(ValueError, "GN.*single-device"):
            with redirect_stdout(StringIO()):
                module.main(args, parser)

        self.assertNotIn("backend", capture)
        self.assertNotIn("train_kwargs", capture)

    def test_magma_optimizers_reject_distributed_backend_before_initialization(self):
        for opt_name, label in (
            ("adamw-magma", "AdamW Magma"),
            ("muon-magma", "Muon Magma"),
        ):
            with self.subTest(opt=opt_name):
                capture = {}
                module = load_main_with_fakes(capture)
                args, parser = make_args_for_main(
                    ["--opt", opt_name, "--distributed_backend", "nccl"]
                )

                with self.assertRaisesRegex(ValueError, f"{label}.*single-device"):
                    with redirect_stdout(StringIO()):
                        module.main(args, parser)

                self.assertNotIn("backend", capture)
                self.assertNotIn("train_kwargs", capture)

    def test_magma_optimizers_reject_actual_multirank_world_size(self):
        for opt_name, label in (
            ("adamw-magma", "AdamW Magma"),
            ("muon-magma", "Muon Magma"),
        ):
            with self.subTest(opt=opt_name):
                capture = {"world_size": 2}
                module = load_main_with_fakes(capture)
                args, parser = make_args_for_main(["--opt", opt_name])

                with self.assertRaisesRegex(ValueError, f"{label}.*world_size=1"):
                    with redirect_stdout(StringIO()):
                        module.main(args, parser)

                self.assertNotIn("get_model_calls", capture)
                self.assertNotIn("train_kwargs", capture)
                self.assertEqual(capture["backend"].finalize_count, 1)

    def test_run_identity_is_available_to_training_and_written_atomically(self):
        capture = {}
        self.run_main(capture)

        cfg = capture["train_kwargs"]["cfg"]
        self.assertEqual(cfg.run_identity, "fake-run-identity")
        self.assertEqual(
            cfg.evaluation_protocol["final_and_full"],
            {"mode": "full_dataset"},
        )
        written_names = [path.name for path, _ in capture["atomic_json_writes"]]
        self.assertIn("run_manifest.json", written_names)
        self.assertIn("summary.json", written_names)
        summary = next(
            payload
            for path, payload in capture["atomic_json_writes"]
            if path.name == "summary.json"
        )
        self.assertEqual(summary["evaluation_protocol"], cfg.evaluation_protocol)
        self.assertEqual(summary["training_semantics"], cfg.training_semantics)
        self.assertEqual(
            summary["optimization_plan"],
            capture["run_manifest"]["optimization_plan"],
        )
        self.assertEqual(
            summary["optimization_plan"]["routing"]["realization"]["status"],
            "resolved",
        )
        summary_paths = [
            path.as_posix()
            for path, _payload in capture["atomic_json_writes"]
            if path.name == "summary.json"
        ]
        self.assertTrue(
            any(
                path.endswith(
                    "/evaluations/fake-evaluation-protocol-identity/summary.json"
                )
                for path in summary_paths
            ),
            summary_paths,
        )

    def test_incompatible_manifest_fails_before_model_or_training_side_effects(self):
        capture = {"manifest_error": ValueError("synthetic identity mismatch")}

        with self.assertRaisesRegex(ValueError, "identity mismatch"):
            self.run_main(capture)

        self.assertEqual(
            capture["lifecycle_events"], ["preflight_manifest_reconciled"]
        )
        self.assertNotIn("get_model_calls", capture)
        self.assertNotIn("train_kwargs", capture)
        self.assertEqual(capture["backend"].finalize_count, 1)

    def test_resolved_manifest_mismatch_fails_before_wandb_or_training(self):
        capture = {
            "resolved_manifest_error": ValueError(
                "synthetic resolved identity mismatch"
            )
        }

        with self.assertRaisesRegex(ValueError, "resolved identity mismatch"):
            self.run_main(capture, ["--wandb"])

        self.assertEqual(capture["get_model_calls"], 1)
        self.assertIn("resolved_optimizer", capture)
        self.assertNotIn("wandb_init_kwargs", capture)
        self.assertNotIn("train_kwargs", capture)
        self.assertEqual(
            capture["lifecycle_events"],
            [
                "preflight_manifest_reconciled",
                "resolved_manifest_reconciled",
            ],
        )
        self.assertEqual(capture["backend"].finalize_count, 1)

    def test_manifestless_checkpoint_is_rejected_before_data_or_writes(self):
        with tempfile.TemporaryDirectory() as directory:
            run_dir = Path(directory) / "legacy-run"
            checkpoint_dir = run_dir / "ckpts" / "latest"
            checkpoint_dir.mkdir(parents=True)
            (checkpoint_dir / "main.pt").write_bytes(b"legacy checkpoint")
            capture = {}

            with self.assertRaisesRegex(ValueError, "resolved run manifest"):
                self.run_main(
                    capture,
                    [
                        "--results_base_folder",
                        directory,
                        "--experiment_name",
                        "legacy-run",
                    ],
                )

            self.assertEqual(
                capture["lifecycle_events"],
                ["resume_manifest_checked"],
            )
            self.assertNotIn("data_reader_calls", capture)
            self.assertNotIn("get_model_calls", capture)
            self.assertNotIn("atomic_json_writes", capture)
            self.assertFalse((run_dir / "run_manifest.json").exists())

            foreign_checkpoint_dir = (
                Path(directory) / "foreign-run" / "ckpts" / "latest"
            )
            foreign_checkpoint_dir.mkdir(parents=True)
            (foreign_checkpoint_dir / "main.pt").write_bytes(b"foreign checkpoint")
            foreign_capture = {}
            with self.assertRaisesRegex(ValueError, "may only point"):
                self.run_main(
                    foreign_capture,
                    [
                        "--results_base_folder",
                        directory,
                        "--experiment_name",
                        "new-run",
                        "--resume_from",
                        str(foreign_checkpoint_dir),
                    ],
                )
            self.assertNotIn("lifecycle_events", foreign_capture)
            self.assertNotIn("data_reader_calls", foreign_capture)
            self.assertNotIn("atomic_json_writes", foreign_capture)

    def test_compatible_resolved_resume_keeps_manifest_bytes(self):
        with tempfile.TemporaryDirectory() as directory:
            run_dir = Path(directory) / "resume-run"
            checkpoint_dir = run_dir / "ckpts" / "latest"
            checkpoint_dir.mkdir(parents=True)
            (checkpoint_dir / "main.pt").write_bytes(b"versioned checkpoint")
            manifest_path = run_dir / "run_manifest.json"
            manifest_path.write_bytes(b'{"fixture":"resolved"}\n')
            original_bytes = manifest_path.read_bytes()
            capture = {
                "preflight_reconcile_action": "keep",
                "resolved_reconcile_action": "keep",
            }

            self.run_main(
                capture,
                [
                    "--results_base_folder",
                    directory,
                    "--experiment_name",
                    "resume-run",
                ],
            )

            self.assertEqual(manifest_path.read_bytes(), original_bytes)
            manifest_writes = [
                path
                for path, _payload in capture["atomic_json_writes"]
                if path.name == "run_manifest.json"
            ]
            self.assertEqual(manifest_writes, [])
            self.assertEqual(
                capture["lifecycle_events"],
                [
                    "resume_manifest_checked",
                    "preflight_manifest_reconciled",
                    "resolved_manifest_reconciled",
                ],
            )

            protocol_drift_capture = {
                "existing_resolved_manifest": {
                    "evaluation_protocol": {"identity": "different-protocol"}
                }
            }
            with self.assertRaisesRegex(
                ValueError, "evaluation protocol identity"
            ):
                self.run_main(
                    protocol_drift_capture,
                    [
                        "--results_base_folder",
                        directory,
                        "--experiment_name",
                        "resume-run",
                    ],
                )
            self.assertEqual(
                protocol_drift_capture["lifecycle_events"],
                ["resume_manifest_checked"],
            )
            self.assertNotIn("data_reader_calls", protocol_drift_capture)
            self.assertNotIn("atomic_json_writes", protocol_drift_capture)

    def test_completed_summary_prevents_resume_or_overwrite(self):
        with tempfile.TemporaryDirectory() as directory:
            run_dir = Path(directory) / "completed-run"
            run_dir.mkdir()
            summary_path = run_dir / "summary.json"
            summary_path.write_text('{"completed":true}\n', encoding="utf-8")
            original_bytes = summary_path.read_bytes()
            capture = {}

            with self.assertRaisesRegex(ValueError, "completed summary"):
                self.run_main(
                    capture,
                    [
                        "--results_base_folder",
                        directory,
                        "--experiment_name",
                        "completed-run",
                    ],
                )

            self.assertEqual(summary_path.read_bytes(), original_bytes)
            self.assertNotIn("data_reader_calls", capture)
            self.assertNotIn("get_model_calls", capture)
            self.assertNotIn("atomic_json_writes", capture)

            versioned_run_dir = Path(directory) / "versioned-completed-run"
            versioned_summary_path = (
                versioned_run_dir
                / "evaluations"
                / "protocol-one"
                / "summary.json"
            )
            versioned_summary_path.parent.mkdir(parents=True)
            versioned_summary_path.write_text(
                '{"completed":true}\n',
                encoding="utf-8",
            )
            versioned_bytes = versioned_summary_path.read_bytes()
            versioned_capture = {}
            with self.assertRaisesRegex(ValueError, "completed summary"):
                self.run_main(
                    versioned_capture,
                    [
                        "--results_base_folder",
                        directory,
                        "--experiment_name",
                        "versioned-completed-run",
                    ],
                )
            self.assertEqual(versioned_summary_path.read_bytes(), versioned_bytes)
            self.assertNotIn("data_reader_calls", versioned_capture)
            self.assertNotIn("atomic_json_writes", versioned_capture)

    def test_notification_secrets_do_not_reach_console_wandb_or_summary(self):
        capture = {}
        secrets = ("smtp-secret-value", "https://secret.invalid/webhook")
        self.run_main(
            capture,
            [
                "--wandb",
                "--notify_smtp_pass",
                secrets[0],
                "--notify_webhook",
                secrets[1],
            ],
        )

        summary = next(
            payload
            for path, payload in capture["atomic_json_writes"]
            if path.name == "summary.json"
        )
        exposed_surfaces = (
            capture["stdout"],
            json.dumps(capture["wandb_init_kwargs"]["config"]),
            json.dumps(summary["args"]),
        )
        for secret in secrets:
            for surface in exposed_surfaces:
                self.assertNotIn(secret, surface)
        self.assertEqual(
            capture["wandb_init_kwargs"]["config"]["notify_smtp_pass"],
            "<redacted>",
        )
        self.assertEqual(summary["args"]["notify_webhook"], "<redacted>")
        self.assertIs(capture["wandb_init_kwargs"]["config"], summary["args"])
        self.assertEqual(capture["sanitized_config_calls"], 1)


if __name__ == "__main__":
    unittest.main()
