import json
import unittest
from contextlib import redirect_stdout
from io import StringIO

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
        written_names = [path.name for path, _ in capture["atomic_json_writes"]]
        self.assertIn("run_manifest.json", written_names)
        self.assertIn("summary.json", written_names)

    def test_incompatible_manifest_fails_before_model_or_training_side_effects(self):
        capture = {"manifest_error": ValueError("synthetic identity mismatch")}

        with self.assertRaisesRegex(ValueError, "identity mismatch"):
            self.run_main(capture)

        self.assertEqual(capture["lifecycle_events"], ["manifest_checked"])
        self.assertNotIn("get_model_calls", capture)
        self.assertNotIn("train_kwargs", capture)
        self.assertEqual(capture["backend"].finalize_count, 1)

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
