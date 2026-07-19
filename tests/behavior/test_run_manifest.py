import argparse
import json
import tempfile
import types
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np

from tests._helpers.behavior_harness import (
    SRC_ROOT,
    load_base_config_module,
    load_module_from_path,
    parse_base_args,
)


def load_run_manifest_module():
    return load_module_from_path(SRC_ROOT / "run_manifest.py")


class RunManifestTest(unittest.TestCase):
    def test_small_artifacts_use_full_hash_and_identity_ignores_path(self):
        module = load_run_manifest_module()
        with tempfile.TemporaryDirectory() as first_dir, tempfile.TemporaryDirectory() as second_dir:
            first = Path(first_dir) / "train.bin"
            second = Path(second_dir) / "train.bin"
            first.write_bytes(b"same-token-bytes")
            second.write_bytes(b"same-token-bytes")

            left = module.build_data_manifest("fineweb", {"train": first})
            right = module.build_data_manifest("fineweb", {"train": second})

        self.assertEqual(
            left["artifacts"]["train"]["fingerprint"]["method"],
            "sha256",
        )
        self.assertNotEqual(
            left["artifacts"]["train"]["path"],
            right["artifacts"]["train"]["path"],
        )
        self.assertEqual(left["identity"], right["identity"])

    def test_large_artifacts_use_fixed_sample_fingerprint(self):
        module = load_run_manifest_module()
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "large.bin"
            path.write_bytes(bytes(range(64)) * 128)
            manifest = module.build_data_manifest(
                "fineweb",
                {"train": path},
                full_hash_limit=128,
                sample_chunk_size=64,
            )

        fingerprint = manifest["artifacts"]["train"]["fingerprint"]
        self.assertEqual(fingerprint["method"], "sampled-sha256-v1")
        self.assertEqual(fingerprint["sample_count"], 3)

    def test_array_artifacts_include_dtype_shape_and_content(self):
        module = load_run_manifest_module()
        left = module.build_data_manifest(
            "arc_easy", {"train": np.array([1, 2, 3], dtype=np.uint16)}
        )
        right = module.build_data_manifest(
            "arc_easy", {"train": np.array([1, 2, 4], dtype=np.uint16)}
        )

        artifact = left["artifacts"]["train"]
        self.assertEqual(artifact["dtype"], "uint16")
        self.assertEqual(artifact["shape"], [3])
        self.assertEqual(artifact["token_count"], 3)
        self.assertNotEqual(left["identity"], right["identity"])
        self.assertEqual(left["semantics_id"], "epfl-flat-eotpad-v1")

    def test_run_identity_excludes_locations_but_includes_seed_and_data(self):
        module = load_run_manifest_module()
        base = dict(
            model="llama",
            dataset="fineweb",
            opt="adamw",
            seed=1,
            data_seed=1337,
            results_base_folder="/tmp/first",
            datasets_dir="/data/first",
            resume_from=None,
            auto_resume=True,
            notify_smtp_pass="secret-value",
            notify_webhook="https://secret.invalid/hook",
            world_size=1,
        )
        data = {
            "schema_version": 1,
            "dataset": "fineweb",
            "semantics_id": "flat-next-token-v1",
            "artifacts": {},
            "identity": "data-one",
        }
        code = {"source_fingerprint_sha256": "code-one", "head": "abc"}
        runtime = {"python": {"version": "3.10.19"}, "packages": {}}

        first = module.build_run_manifest(
            types.SimpleNamespace(**base), data, code=code, runtime=runtime
        )
        moved = dict(base, results_base_folder="/tmp/second", datasets_dir="/data/second")
        second = module.build_run_manifest(
            types.SimpleNamespace(**moved), data, code=code, runtime=runtime
        )
        changed_seed = module.build_run_manifest(
            types.SimpleNamespace(**dict(base, seed=2)), data, code=code, runtime=runtime
        )
        changed_data = module.build_run_manifest(
            types.SimpleNamespace(**base),
            {**data, "identity": "data-two"},
            code=code,
            runtime=runtime,
        )
        changed_code = module.build_run_manifest(
            types.SimpleNamespace(**base),
            data,
            code={**code, "source_fingerprint_sha256": "code-two"},
            runtime=runtime,
        )
        rank_zero = module.build_run_manifest(
            types.SimpleNamespace(**dict(base, seed=1, run_seed=1)),
            data,
            code=code,
            runtime=runtime,
        )
        rank_one = module.build_run_manifest(
            types.SimpleNamespace(**dict(base, seed=2, run_seed=1)),
            data,
            code=code,
            runtime=runtime,
        )

        self.assertEqual(first["run_identity"], second["run_identity"])
        self.assertNotEqual(first["run_identity"], changed_seed["run_identity"])
        self.assertNotEqual(first["run_identity"], changed_data["run_identity"])
        self.assertNotEqual(first["run_identity"], changed_code["run_identity"])
        self.assertEqual(rank_zero["run_identity"], rank_one["run_identity"])
        self.assertEqual(first["config"]["notify_smtp_pass"], "<redacted>")
        self.assertEqual(first["config"]["notify_webhook"], "<redacted>")
        serialized = json.dumps(first)
        self.assertNotIn("secret-value", serialized)
        self.assertNotIn("secret.invalid", serialized)

    def test_run_identity_ignores_python_install_path_but_tracks_versions(self):
        module = load_run_manifest_module()
        args = types.SimpleNamespace(model="llama", dataset="fineweb", opt="adamw")
        data = {
            "schema_version": 1,
            "dataset": "fineweb",
            "semantics_id": "flat-next-token-v1",
            "artifacts": {},
            "identity": "data-one",
        }
        code = {"source_fingerprint_sha256": "code-one", "head": "abc"}
        runtime = {
            "python": {
                "implementation": "CPython",
                "version": "3.10.19",
                "executable": "/first/env/bin/python",
            },
            "platform": {"system": "Linux", "release": "first", "machine": "x86_64"},
            "packages": {"torch": "2.9.1"},
        }

        first = module.build_run_manifest(args, data, code=code, runtime=runtime)
        moved = module.build_run_manifest(
            args,
            data,
            code=code,
            runtime={
                **runtime,
                "python": {**runtime["python"], "executable": "/second/env/bin/python"},
                "platform": {**runtime["platform"], "release": "second"},
            },
        )
        upgraded = module.build_run_manifest(
            args,
            data,
            code=code,
            runtime={**runtime, "packages": {"torch": "2.10.0"}},
        )

        self.assertEqual(first["run_identity"], moved["run_identity"])
        self.assertNotEqual(first["run_identity"], upgraded["run_identity"])
        self.assertEqual(first["runtime"]["python"]["executable"], "/first/env/bin/python")

    def test_sanitized_config_redacts_unknown_notification_credentials(self):
        module = load_run_manifest_module()
        config = module.sanitized_config(
            types.SimpleNamespace(
                notify_interval=5,
                notify_method="webhook",
                notify_smtp_pass="smtp-secret",
                notify_future_secret="future-secret",
            )
        )

        self.assertEqual(config["notify_interval"], 5)
        self.assertEqual(config["notify_method"], "webhook")
        self.assertEqual(config["notify_smtp_pass"], "<redacted>")
        self.assertEqual(config["notify_future_secret"], "<redacted>")
        self.assertNotIn("smtp-secret", json.dumps(config))
        self.assertNotIn("future-secret", json.dumps(config))

    def test_run_identity_normalizes_device_type(self):
        module = load_run_manifest_module()
        data = {
            "schema_version": 1,
            "dataset": "fineweb",
            "semantics_id": "flat-next-token-v1",
            "artifacts": {},
            "identity": "data-one",
        }
        code = {"source_fingerprint_sha256": "code-one", "head": "abc"}
        runtime = {"python": {"version": "3.10.19"}, "packages": {}}

        cuda_zero = module.build_run_manifest(
            types.SimpleNamespace(device="cuda:0"), data, code=code, runtime=runtime
        )
        cuda_one = module.build_run_manifest(
            types.SimpleNamespace(device="cuda:1"), data, code=code, runtime=runtime
        )
        cpu = module.build_run_manifest(
            types.SimpleNamespace(device="cpu"), data, code=code, runtime=runtime
        )

        self.assertEqual(cuda_zero["run_identity"], cuda_one["run_identity"])
        self.assertNotEqual(cuda_zero["run_identity"], cpu["run_identity"])
        self.assertEqual(cuda_zero["compatibility_config"]["device_type"], "cuda")
        self.assertEqual(cpu["compatibility_config"]["device_type"], "cpu")

    def test_evaluation_controls_are_recorded_but_do_not_change_run_identity(self):
        module = load_run_manifest_module()
        data = {
            "schema_version": 1,
            "dataset": "fineweb",
            "semantics_id": "flat-next-token-v1",
            "artifacts": {},
            "identity": "data-one",
        }
        code = {"source_fingerprint_sha256": "code-one", "head": "abc"}
        runtime = {"python": {"version": "3.10.19"}, "packages": {}}
        base = dict(
            opt="adamw",
            eval_batches=64,
            eval_interval=200,
            full_eval_at=[],
            final_eval_batches=None,
            final_eval_tokens=None,
            distributed_control_timeout_seconds=86400,
        )

        full = module.build_run_manifest(
            types.SimpleNamespace(**base), data, code=code, runtime=runtime
        )
        batches = module.build_run_manifest(
            types.SimpleNamespace(**dict(base, final_eval_batches=7)),
            data,
            code=code,
            runtime=runtime,
        )
        tokens = module.build_run_manifest(
            types.SimpleNamespace(**dict(base, final_eval_tokens=4096)),
            data,
            code=code,
            runtime=runtime,
        )
        timeout = module.build_run_manifest(
            types.SimpleNamespace(
                **dict(base, distributed_control_timeout_seconds=123)
            ),
            data,
            code=code,
            runtime=runtime,
        )
        generation_prefix = module.build_run_manifest(
            types.SimpleNamespace(**dict(base, eval_seq_prefix="Hello")),
            data,
            code=code,
            runtime=runtime,
        )
        alternate_schedule = module.build_run_manifest(
            types.SimpleNamespace(
                **dict(base, eval_interval=100, full_eval_at=[300])
            ),
            data,
            code=code,
            runtime=runtime,
        )
        normalized_schedule = module.build_run_manifest(
            types.SimpleNamespace(**dict(base, full_eval_at=[300, 200, 300])),
            data,
            code=code,
            runtime=runtime,
        )
        reordered_schedule = module.build_run_manifest(
            types.SimpleNamespace(**dict(base, full_eval_at=[200, 300])),
            data,
            code=code,
            runtime=runtime,
        )

        self.assertEqual(
            {manifest["run_identity"] for manifest in (full, batches, tokens, timeout)},
            {full["run_identity"]},
        )
        self.assertEqual(full["schema_version"], 3)
        self.assertEqual(
            len(
                {
                    manifest["evaluation_protocol"]["identity"]
                    for manifest in (full, batches, tokens)
                }
            ),
            3,
        )
        self.assertEqual(
            full["evaluation_protocol"]["identity"],
            timeout["evaluation_protocol"]["identity"],
        )
        self.assertNotEqual(
            full["evaluation_protocol"]["identity"],
            alternate_schedule["evaluation_protocol"]["identity"],
        )
        self.assertEqual(full["run_identity"], alternate_schedule["run_identity"])
        self.assertEqual(full["run_identity"], generation_prefix["run_identity"])
        self.assertNotEqual(
            full["evaluation_protocol"]["identity"],
            generation_prefix["evaluation_protocol"]["identity"],
        )
        self.assertEqual(
            normalized_schedule["evaluation_protocol"]["identity"],
            reordered_schedule["evaluation_protocol"]["identity"],
        )
        self.assertEqual(
            normalized_schedule["evaluation_protocol"]["full_eval_at"],
            [200, 300],
        )
        self.assertNotIn("final_eval_batches", batches["compatibility_config"])
        self.assertNotIn("final_eval_tokens", tokens["compatibility_config"])
        self.assertNotIn(
            "distributed_control_timeout_seconds", timeout["compatibility_config"]
        )
        self.assertEqual(
            full["evaluation_protocol"]["final_and_full"], {"mode": "full_dataset"}
        )
        self.assertEqual(
            batches["evaluation_protocol"]["final_and_full"],
            {"mode": "batch_cap", "max_batches": 7},
        )
        self.assertEqual(
            tokens["evaluation_protocol"]["final_and_full"],
            {"mode": "token_cap", "max_tokens": 4096},
        )
        self.assertEqual(batches["config"]["final_eval_batches"], 7)
        self.assertEqual(tokens["config"]["final_eval_tokens"], 4096)
        self.assertEqual(timeout["config"]["distributed_control_timeout_seconds"], 123)
        tampered_protocol = json.loads(json.dumps(full))
        tampered_protocol["evaluation_protocol"]["generation_prefix"] = "tampered"
        with self.assertRaisesRegex(
            ValueError, "Evaluation protocol identity.*canonical content"
        ):
            module.validate_run_manifest(tampered_protocol)

    def test_optimization_plan_affects_identity_but_reporting_views_do_not(self):
        module = load_run_manifest_module()
        data = {
            "schema_version": 1,
            "dataset": "fineweb",
            "semantics_id": "flat-next-token-v1",
            "artifacts": {},
            "identity": "data-one",
        }
        code = {"source_fingerprint_sha256": "code-one", "head": "abc"}
        runtime = {"python": {"version": "3.10.19"}, "packages": {}}
        args = types.SimpleNamespace(opt="sf-adamw")

        current = module.build_run_manifest(args, data, code=code, runtime=runtime)
        changed_plan = json.loads(json.dumps(current["optimization_plan"]))
        changed_plan["components"][0]["hyperparameters"]["lr"] = 0.123
        changed_plan["identity"] = module._optimization_intent_identity(changed_plan)
        with patch.object(
            module,
            "build_training_semantics",
            return_value={
                "schedule_free_parameter_point": "schedule_free_parameter_point_v3"
            },
        ):
            changed_training = module.build_run_manifest(
                args, data, code=code, runtime=runtime
            )
        with patch.object(
            module,
            "build_optimization_plan",
            return_value=changed_plan,
        ):
            changed_optimization = module.build_run_manifest(
                args, data, code=code, runtime=runtime
            )
        with patch.object(
            module,
            "build_metric_semantics",
            return_value={"training_loss": "another_reporting_definition"},
        ):
            changed_metric = module.build_run_manifest(
                args, data, code=code, runtime=runtime
            )

        self.assertEqual(
            current["training_semantics"]["schedule_free_parameter_point"],
            "schedule_free_parameter_point_v2",
        )
        self.assertEqual(
            current["metric_semantics"]["training_loss"],
            "global_step_microbatch_mean_v2",
        )
        self.assertEqual(current["run_identity"], changed_training["run_identity"])
        self.assertNotEqual(
            current["run_identity"], changed_optimization["run_identity"]
        )
        self.assertEqual(current["run_identity"], changed_metric["run_identity"])
        self.assertNotEqual(
            current["metric_semantics"], changed_metric["metric_semantics"]
        )

        stale_plan = json.loads(json.dumps(current["optimization_plan"]))
        stale_plan["components"][0]["hyperparameters"]["lr"] = 0.456
        with patch.object(
            module,
            "build_optimization_plan",
            return_value=stale_plan,
        ), self.assertRaisesRegex(ValueError, "identity"):
            module.build_run_manifest(args, data, code=code, runtime=runtime)

    def test_sophiag_global_accum_semantics_record_estimator_work(self):
        module = load_run_manifest_module()
        args = types.SimpleNamespace(
            opt="sophiag",
            sophia_estimator_mode="global_accum",
            world_size=4,
            batch_size=16,
            acc_steps=4,
            sequence_length=512,
            precondition_frequency=10,
            sophia_bs=256,
        )

        semantics = module.build_training_semantics(args)

        self.assertEqual(
            semantics["sophia_hessian_estimator"],
            {
                "mode": "global_accum",
                "version": "global_accum_gnb_v1",
                "refresh_frequency": 10,
                "expected_examples_per_refresh": 256,
                "expected_tokens_per_refresh": 131072,
                "optimizer_scale_tokens": 131072,
            },
        )

    def test_reporting_only_final_model_and_rank_check_do_not_change_identity(self):
        module = load_run_manifest_module()
        data = {
            "schema_version": 1,
            "dataset": "fineweb",
            "semantics_id": "flat-next-token-v1",
            "artifacts": {},
            "identity": "data-one",
        }
        code = {"source_fingerprint_sha256": "code-one", "head": "abc"}
        runtime = {"python": {"version": "3.10.19"}, "packages": {}}
        base = dict(
            opt="adamw",
            model="llama",
            seed=0,
            data_seed=1337,
            save_final_model=False,
            sophia_verify_rank_state=False,
        )
        enabled = dict(
            base, save_final_model=True, sophia_verify_rank_state=True
        )

        first = module.build_run_manifest(
            types.SimpleNamespace(**base), data, code=code, runtime=runtime
        )
        second = module.build_run_manifest(
            types.SimpleNamespace(**enabled), data, code=code, runtime=runtime
        )

        self.assertEqual(first["run_identity"], second["run_identity"])
        self.assertIs(second["config"]["save_final_model"], True)

    def test_runtime_semantics_enrichment_does_not_change_identity_on_rebuild(self):
        module = load_run_manifest_module()
        data = {
            "schema_version": 1,
            "dataset": "fineweb",
            "semantics_id": "flat-next-token-v1",
            "artifacts": {},
            "identity": "data-one",
        }
        code = {"source_fingerprint_sha256": "code-one", "head": "abc"}
        runtime = {"python": {"version": "3.10.19"}, "packages": {}}
        args = types.SimpleNamespace(
            opt="sf-adamw",
            eval_batches=64,
            eval_interval=200,
            full_eval_at=[],
            final_eval_batches=None,
            final_eval_tokens=4096,
        )

        first = module.build_run_manifest(args, data, code=code, runtime=runtime)
        args.evaluation_protocol = first["evaluation_protocol"]
        args.training_semantics = first["training_semantics"]
        args.metric_semantics = first["metric_semantics"]
        second = module.build_run_manifest(args, data, code=code, runtime=runtime)

        self.assertEqual(first["run_identity"], second["run_identity"])
        self.assertEqual(
            first["compatibility_config"], second["compatibility_config"]
        )
        self.assertNotIn("evaluation_protocol", second["compatibility_config"])
        self.assertNotIn("training_semantics", second["compatibility_config"])

    def test_gn_reports_its_actual_last_inner_loss_semantics(self):
        module = load_run_manifest_module()
        data = {
            "schema_version": 1,
            "dataset": "fineweb",
            "semantics_id": "flat-next-token-v1",
            "artifacts": {},
            "identity": "data-one",
        }
        code = {"source_fingerprint_sha256": "code-one", "head": "abc"}
        runtime = {"python": {"version": "3.10.19"}, "packages": {}}

        manifest = module.build_run_manifest(
            types.SimpleNamespace(opt="gn-full"),
            data,
            code=code,
            runtime=runtime,
        )

        self.assertEqual(
            manifest["metric_semantics"]["training_loss"],
            "gn_last_inner_base_loss_v1",
        )

    def test_manifest_v3_emits_active_only_single_adamw_plan(self):
        module = load_run_manifest_module()
        args, _ = parse_base_args(
            [
                "--opt",
                "adamw",
                "--lr",
                "0.001",
                "--beta1",
                "0.9",
                "--beta2",
                "0.999",
                "--weight_decay",
                "0.1",
            ]
        )
        data = {
            "schema_version": 1,
            "dataset": "fineweb",
            "semantics_id": "flat-next-token-v1",
            "artifacts": {},
            "identity": "data-one",
        }
        manifest = module.build_run_manifest(
            args,
            data,
            code={"source_fingerprint_sha256": "code-one", "head": "abc"},
            runtime={"python": {"version": "3.10.19"}, "packages": {}},
        )

        self.assertEqual(manifest["schema_version"], 3)
        self.assertEqual(manifest["manifest_state"], "preflight")
        self.assertEqual(manifest["optimization_plan"]["schema_version"], 1)
        self.assertEqual(
            manifest["optimization_plan"]["strategy"],
            {"id": "adamw_v1", "kind": "single"},
        )
        components = manifest["optimization_plan"]["components"]
        self.assertEqual(len(components), 1)
        self.assertEqual(components[0]["algorithm"], "adamw")
        self.assertEqual(
            components[0]["hyperparameters"]["betas"], [0.9, 0.999]
        )
        self.assertIn("sophia_bs", manifest["config"])
        self.assertNotIn("sophia_bs", manifest["compatibility_config"])
        self.assertNotIn("opt", manifest["compatibility_config"])

    def test_adamw_inactive_optimizer_args_are_identity_neutral(self):
        module = load_run_manifest_module()
        args, _ = parse_base_args(["--opt", "adamw"])
        data = {
            "schema_version": 1,
            "dataset": "fineweb",
            "semantics_id": "flat-next-token-v1",
            "artifacts": {},
            "identity": "data-one",
        }
        code = {"source_fingerprint_sha256": "code-one", "head": "abc"}
        runtime = {"python": {"version": "3.10.19"}, "packages": {}}
        baseline = module.build_run_manifest(
            args, data, code=code, runtime=runtime
        )

        for field, value in (
            ("sophia_bs", 999),
            ("sophia_rho", 0.123),
            ("muon_ns_steps", 17),
            ("gn_inner_lr", 0.25),
            ("soap_data_format", "channels_last"),
            ("mars_lr", 0.75),
        ):
            with self.subTest(field=field):
                changed_args = types.SimpleNamespace(**vars(args))
                setattr(changed_args, field, value)
                changed = module.build_run_manifest(
                    changed_args, data, code=code, runtime=runtime
                )
                self.assertEqual(
                    baseline["preflight_identity"],
                    changed["preflight_identity"],
                )
                self.assertEqual(
                    baseline["optimization_plan"]["identity"],
                    changed["optimization_plan"]["identity"],
                )
                self.assertNotEqual(
                    baseline["config"][field], changed["config"][field]
                )

    def test_active_adamw_args_and_scheduler_horizon_change_identity(self):
        module = load_run_manifest_module()
        args, _ = parse_base_args(["--opt", "adamw"])
        data = {
            "schema_version": 1,
            "dataset": "fineweb",
            "semantics_id": "flat-next-token-v1",
            "artifacts": {},
            "identity": "data-one",
        }
        code = {"source_fingerprint_sha256": "code-one", "head": "abc"}
        runtime = {"python": {"version": "3.10.19"}, "packages": {}}
        baseline = module.build_run_manifest(
            args, data, code=code, runtime=runtime
        )

        for field, value in (
            ("lr", 0.002),
            ("beta1", 0.8),
            ("beta2", 0.999),
            ("weight_decay", 0.03),
            ("iterations", args.iterations + 1),
            ("warmup_steps", args.warmup_steps + 1),
            ("final_div_factor", 10.0),
        ):
            with self.subTest(field=field):
                changed_args = types.SimpleNamespace(**vars(args))
                setattr(changed_args, field, value)
                changed = module.build_run_manifest(
                    changed_args, data, code=code, runtime=runtime
                )
                self.assertNotEqual(
                    baseline["preflight_identity"],
                    changed["preflight_identity"],
                )

    def test_inactive_scheduler_and_conditional_fields_are_identity_neutral(self):
        module = load_run_manifest_module()
        args, _ = parse_base_args(["--opt", "adamw", "--scheduler", "cos"])
        data = {
            "schema_version": 1,
            "dataset": "fineweb",
            "semantics_id": "flat-next-token-v1",
            "artifacts": {},
            "identity": "data-one",
        }
        code = {"source_fingerprint_sha256": "code-one", "head": "abc"}
        runtime = {"python": {"version": "3.10.19"}, "packages": {}}
        baseline = module.build_run_manifest(
            args, data, code=code, runtime=runtime
        )

        for field, value in (
            ("cos_inf_steps", 1234),
            ("wsd_final_lr_scale", 0.25),
            ("wsd_fract_decay", 0.4),
            ("decay_type", "sqrt"),
            ("gn_ls_range", 19),
            ("gn_log_inner_steps", True),
        ):
            with self.subTest(field=field):
                changed_args = types.SimpleNamespace(**vars(args))
                setattr(changed_args, field, value)
                changed = module.build_run_manifest(
                    changed_args, data, code=code, runtime=runtime
                )
                self.assertEqual(
                    baseline["preflight_identity"],
                    changed["preflight_identity"],
                )

        muon_args, _ = parse_base_args(["--opt", "muon", "--scheduler", "cos"])
        changed_muon_args = types.SimpleNamespace(**vars(muon_args))
        changed_muon_args.final_div_factor = 999.0
        muon = module.build_run_manifest(
            muon_args, data, code=code, runtime=runtime
        )
        changed_muon = module.build_run_manifest(
            changed_muon_args, data, code=code, runtime=runtime
        )
        self.assertEqual(muon["preflight_identity"], changed_muon["preflight_identity"])
        self.assertEqual(
            muon["optimization_plan"]["scheduler"]["final_div_factor"], 1.0
        )

    def test_optimizer_variant_fields_are_canonicalized_by_actual_semantics(self):
        module = load_run_manifest_module()
        data = {
            "schema_version": 1,
            "dataset": "fineweb",
            "semantics_id": "flat-next-token-v1",
            "artifacts": {},
            "identity": "data-one",
        }
        kwargs = {
            "code": {"source_fingerprint_sha256": "code-one", "head": "abc"},
            "runtime": {"python": {"version": "3.10.19"}, "packages": {}},
        }

        def identity(argv, **changes):
            args, _ = parse_base_args(argv)
            for key, value in changes.items():
                setattr(args, key, value)
            return module.build_run_manifest(args, data, **kwargs)[
                "preflight_identity"
            ]

        self.assertEqual(
            identity(["--opt", "gn-full"], lr=0.001),
            identity(["--opt", "gn-full"], lr=0.9),
        )
        self.assertNotEqual(
            identity(["--opt", "gn-full", "--model", "mup_llama"], lr=0.001),
            identity(["--opt", "gn-full", "--model", "mup_llama"], lr=0.9),
        )
        for optimizer in ("gn-prox", "gn-full"):
            with self.subTest(optimizer=optimizer, field="gn_inner_wd"):
                self.assertNotEqual(
                    identity(["--opt", optimizer], gn_inner_wd=0.0),
                    identity(["--opt", optimizer], gn_inner_wd=0.2),
                )

        self.assertEqual(
            identity(["--opt", "sf-sgd"], nesterov=False),
            identity(["--opt", "sf-sgd"], nesterov=True),
        )
        for variant in ("mars-lion", "mars-shampoo"):
            with self.subTest(variant=variant, field="mars_beta2"):
                self.assertEqual(
                    identity(["--opt", "mars", "--mars_type", variant], mars_beta2=0.8),
                    identity(["--opt", "mars", "--mars_type", variant], mars_beta2=0.99),
                )
        self.assertNotEqual(
            identity(["--opt", "mars", "--mars_type", "mars-adamw"], mars_beta2=0.8),
            identity(["--opt", "mars", "--mars_type", "mars-adamw"], mars_beta2=0.99),
        )

        self.assertEqual(
            identity(["--opt", "scion"], scion_emb_scale=1.0),
            identity(["--opt", "scion"], scion_emb_scale=7.0),
        )
        self.assertNotEqual(
            identity(["--opt", "scion", "--untied_embeds"], scion_emb_scale=1.0),
            identity(["--opt", "scion", "--untied_embeds"], scion_emb_scale=7.0),
        )
        self.assertEqual(
            identity(["--opt", "scion-light"], momentum=0.1),
            identity(["--opt", "scion-light"], momentum=0.9),
        )

        self.assertEqual(
            identity(["--opt", "adafactor"], lr=0.001),
            identity(["--opt", "adafactor"], lr=0.5),
        )
        self.assertEqual(
            identity(["--opt", "adafactor", "--scheduler", "none"]),
            identity(
                [
                    "--opt",
                    "adafactor",
                    "--scheduler",
                    "cos",
                    "--warmup_steps",
                    "1",
                ]
            ),
        )
        self.assertNotEqual(
            identity(["--opt", "adafactor"], iterations=100),
            identity(["--opt", "adafactor"], iterations=101),
        )

        self.assertEqual(
            identity(["--opt", "signsgd"], dampening=0.0, nesterov=False),
            identity(["--opt", "signsgd"], dampening=0.7, nesterov=True),
        )
        self.assertEqual(
            identity(["--opt", "signum"], momentum=0.0, dampening=0.0),
            identity(["--opt", "signum"], momentum=0.0, dampening=0.7),
        )
        self.assertNotEqual(
            identity(["--opt", "signum"], momentum=0.9, dampening=0.0),
            identity(["--opt", "signum"], momentum=0.9, dampening=0.2),
        )

        self.assertEqual(
            identity(["--opt", "soap"], shampoo_beta=-1.0),
            identity(["--opt", "soap"], shampoo_beta=0.95),
        )
        self.assertEqual(
            identity(["--opt", "soap"], soap_data_format="channels_first"),
            identity(["--opt", "soap"], soap_data_format="channels_last"),
        )
        self.assertEqual(
            identity(["--opt", "prodigy"], prodigy_beta3=None),
            identity(["--opt", "prodigy"], prodigy_beta3=0.95**0.5),
        )
        self.assertEqual(
            identity(["--opt", "adamw"], clipping_type="no", clip_eta=1.0),
            identity(["--opt", "adamw"], clipping_type="local", clip_eta=0.5),
        )

    def test_realization_identity_omits_inactive_raw_group_defaults(self):
        module = load_run_manifest_module()

        class Param:
            shape = (2, 2)
            ndim = 2
            requires_grad = True

            def numel(self):
                return 4

        parameter = Param()

        def resolve(argv, group):
            args, _ = parse_base_args(argv)
            optimizer = types.SimpleNamespace(
                param_groups=[{"params": [parameter], **group}],
                state={},
            )
            return module.resolve_optimization_plan(
                module.build_optimization_plan(args),
                optimizer,
                [("weight", parameter)],
            )["routing"]["realization"]["identity"]

        self.assertEqual(
            resolve(
                ["--opt", "adafactor"],
                {"lr": 0.001, "relative_step": True},
            ),
            resolve(
                ["--opt", "adafactor"],
                {"lr": 0.9, "relative_step": True},
            ),
        )
        self.assertEqual(
            resolve(
                ["--opt", "mars", "--mars_type", "mars-lion"],
                {"betas": (0.95, 0.8), "lr": 0.003},
            ),
            resolve(
                ["--opt", "mars", "--mars_type", "mars-lion"],
                {"betas": (0.95, 0.99), "lr": 0.003},
            ),
        )
        self.assertEqual(
            resolve(
                ["--opt", "signsgd"],
                {"momentum": 0.0, "dampening": 0.0, "nesterov": False},
            ),
            resolve(
                ["--opt", "signsgd"],
                {"momentum": 0.0, "dampening": 0.8, "nesterov": True},
            ),
        )
        self.assertEqual(
            resolve(
                ["--opt", "soap", "--shampoo_beta", "-1"],
                {
                    "betas": (0.9, 0.95),
                    "shampoo_beta": -1.0,
                    "data_format": "channels_first",
                },
            ),
            resolve(
                ["--opt", "soap", "--shampoo_beta", "0.95"],
                {
                    "betas": (0.9, 0.95),
                    "shampoo_beta": 0.95,
                    "data_format": "channels_last",
                },
            ),
        )
        self.assertEqual(
            resolve(
                ["--opt", "prodigy"],
                {"betas": (0.9, 0.95), "beta3": None},
            ),
            resolve(
                ["--opt", "prodigy", "--prodigy_beta3", str(0.95**0.5)],
                {"betas": (0.9, 0.95), "beta3": 0.95**0.5},
            ),
        )
        self.assertEqual(
            resolve(
                ["--opt", "scion-light", "--momentum", "0.1"],
                {"momentum": 0.1, "scale": 3.0},
            ),
            resolve(
                ["--opt", "scion-light", "--momentum", "0.9"],
                {"momentum": 0.9, "scale": 3.0},
            ),
        )

    def test_resolution_rejects_trainable_model_parameters_missing_from_optimizer(self):
        module = load_run_manifest_module()

        class Param:
            shape = (2,)
            ndim = 1
            requires_grad = True

            def numel(self):
                return 2

        kept = Param()
        dropped = Param()
        optimizer = types.SimpleNamespace(
            param_groups=[{"params": [kept], "lr": 0.001}],
            state={},
        )
        args, _ = parse_base_args(["--opt", "adamw"])
        with self.assertRaisesRegex(ValueError, "missing from the optimizer"):
            module.resolve_optimization_plan(
                module.build_optimization_plan(args),
                optimizer,
                [("kept", kept), ("dropped", dropped)],
            )

    def test_all_cli_optimizers_have_an_explicit_plan_descriptor(self):
        module = load_run_manifest_module()
        args, parser = parse_base_args([])
        opt_action = next(
            action for action in parser._actions if action.dest == "opt"
        )
        self.assertEqual(len(opt_action.choices), 27)

        for opt in opt_action.choices:
            with self.subTest(opt=opt):
                selected = types.SimpleNamespace(**vars(args))
                selected.opt = opt
                if opt in {"sf-adamw", "sf-sgd"}:
                    selected.scheduler = "none"
                plan = module.build_optimization_plan(selected)
                self.assertEqual(plan["schema_version"], 1)
                self.assertTrue(plan["components"])
                self.assertTrue(plan["routing"]["update_routes"])
                self.assertEqual(
                    plan["routing"]["realization"]["status"], "pending"
                )

        args.opt = "not-an-optimizer"
        with self.assertRaisesRegex(ValueError, "Unknown optimizer"):
            module.build_optimization_plan(args)

    def test_all_optimizer_and_scheduler_parser_fields_are_identity_routed(self):
        module = load_run_manifest_module()
        base_config = load_base_config_module()
        parser = argparse.ArgumentParser(allow_abbrev=False)
        base_config.register_scheduler_args(parser)
        base_config.register_optimizer_args(parser)
        parser_fields = {
            action.dest
            for action in parser._actions
            if action.dest not in {"help", "batch_size", "acc_steps"}
        }
        self.assertEqual(
            parser_fields
            - module.OPTIMIZATION_CONFIG_KEYS
            - module.COMPATIBILITY_EXCLUDED_CONFIG,
            set(),
        )

    def test_nested_gn_and_sophiag_auxiliary_are_structured_components(self):
        module = load_run_manifest_module()
        gn_args, _ = parse_base_args(
            [
                "--opt",
                "gn-prox",
                "--gn_inner_lr",
                "0.004",
                "--gn_inner_wd",
                "0.3",
                "--gn_linesearch",
                "--gn_ls_range",
                "7",
            ]
        )
        sophia_args, _ = parse_base_args(
            [
                "--opt",
                "sophiag",
                "--sophia_estimator_mode",
                "global_accum",
                "--batch_size",
                "16",
                "--acc_steps",
                "4",
                "--sequence_length",
                "512",
                "--sophia_bs",
                "64",
            ]
        )
        sophia_args.world_size = 1

        gn_plan = module.build_optimization_plan(gn_args)
        sophia_plan = module.build_optimization_plan(sophia_args)

        self.assertEqual(gn_plan["strategy"]["kind"], "nested")
        self.assertEqual(
            [(item["role"], item["algorithm"]) for item in gn_plan["components"]],
            [("outer", "gauss_newton"), ("inner", "adamw")],
        )
        self.assertEqual(
            gn_plan["components"][1]["hyperparameters"]["weight_decay"], 0.0
        )
        self.assertEqual(
            gn_plan["components"][0]["hyperparameters"]["line_search_range"], 7
        )
        self.assertEqual(
            sophia_plan["strategy"]["kind"], "single_with_auxiliary"
        )
        self.assertEqual(
            [(item["role"], item["algorithm"]) for item in sophia_plan["components"]],
            [
                ("primary", "sophiag"),
                ("auxiliary", "gauss_newton_bartlett_diagonal"),
            ],
        )
        estimator = sophia_plan["components"][1]["hyperparameters"]
        self.assertEqual(estimator["expected_examples_per_refresh"], 64)
        self.assertEqual(estimator["expected_tokens_per_refresh"], 32768)

    def test_muon_resolution_records_actual_partition_and_final_identity(self):
        module = load_run_manifest_module()

        class Param:
            def __init__(self, shape):
                self.shape = shape

            @property
            def ndim(self):
                return len(self.shape)

            def size(self, dimension):
                return self.shape[dimension]

            def numel(self):
                total = 1
                for dimension in self.shape:
                    total *= dimension
                return total

        matrix = Param((4, 4))
        fallback = Param((4,))
        optimizer = types.SimpleNamespace(
            param_groups=[
                {
                    "params": [matrix, fallback],
                    "lr": 0.02,
                    "adamw_lr": 0.001,
                    "weight_decay": 0.1,
                }
            ],
            state={
                matrix: {"use_muon": True},
                fallback: {"use_muon": False},
            },
        )
        args, _ = parse_base_args(
            ["--opt", "muon", "--muon_lr_factor", "0.02", "--lr", "0.001"]
        )
        pending = module.build_optimization_plan(args)
        resolved = module.resolve_optimization_plan(
            pending,
            optimizer,
            [("norm.weight", fallback), ("block.weight", matrix)],
        )
        reordered = module.resolve_optimization_plan(
            pending,
            optimizer,
            [("block.weight", matrix), ("norm.weight", fallback)],
        )

        realization = resolved["routing"]["realization"]
        self.assertEqual(realization["status"], "resolved")
        self.assertEqual(
            [
                (route["component"], route["tensor_count"])
                for route in realization["update_routes"]
            ],
            [("matrix_update", 1), ("fallback_update", 1)],
        )
        self.assertEqual(realization["coverage"]["unassigned_tensors"], 0)
        self.assertEqual(realization["coverage"]["multiply_assigned_tensors"], 0)
        self.assertEqual(realization["identity"], reordered["routing"]["realization"]["identity"])

        data = {
            "schema_version": 1,
            "dataset": "fineweb",
            "semantics_id": "flat-next-token-v1",
            "artifacts": {},
            "identity": "data-one",
        }
        kwargs = {
            "code": {"source_fingerprint_sha256": "code-one", "head": "abc"},
            "runtime": {"python": {"version": "3.10.19"}, "packages": {}},
        }
        preflight = module.build_run_manifest(args, data, **kwargs)
        final = module.build_run_manifest(
            args, data, optimization_plan=resolved, **kwargs
        )
        self.assertEqual(final["manifest_state"], "resolved")
        self.assertEqual(final["preflight_identity"], preflight["preflight_identity"])
        self.assertNotEqual(final["run_identity"], preflight["run_identity"])

        optimizer.state[matrix]["use_muon"] = False
        optimizer.state[fallback]["use_muon"] = True
        rerouted = module.resolve_optimization_plan(
            pending,
            optimizer,
            [("block.weight", matrix), ("norm.weight", fallback)],
        )
        rerouted_manifest = module.build_run_manifest(
            args, data, optimization_plan=rerouted, **kwargs
        )
        self.assertNotEqual(final["run_identity"], rerouted_manifest["run_identity"])

    def test_magma_overlay_does_not_count_as_duplicate_updater_assignment(self):
        module = load_run_manifest_module()

        class Param:
            shape = (2, 2)
            ndim = 2

            def size(self, dimension):
                return self.shape[dimension]

            def numel(self):
                return 4

        first = Param()
        second = Param()
        optimizer = types.SimpleNamespace(
            param_groups=[{"params": [first, second], "lr": 0.001}],
            state={},
        )
        args, _ = parse_base_args(
            ["--opt", "adamw-magma", "--magma_scope", "attn-mlp"]
        )
        plan = module.resolve_optimization_plan(
            module.build_optimization_plan(args),
            optimizer,
            [("block.attn.weight", first), ("norm.weight", second)],
            overlay_param_ids={"magma_modifier": {id(first)}},
        )

        realization = plan["routing"]["realization"]
        self.assertEqual(realization["update_routes"][0]["tensor_count"], 2)
        self.assertEqual(realization["overlays"][0]["tensor_count"], 1)
        self.assertEqual(realization["coverage"]["multiply_assigned_tensors"], 0)

    def test_atomic_manifest_write_and_schema_compatibility_check(self):
        module = load_run_manifest_module()

        class Param:
            shape = (2,)
            ndim = 1
            requires_grad = True

            def numel(self):
                return 2

        class FirstOptimizer:
            def __init__(self, parameter):
                self.param_groups = [{"params": [parameter], "lr": 0.001}]
                self.state = {}

        class SecondOptimizer(FirstOptimizer):
            pass

        args, _ = parse_base_args(["--opt", "adamw"])
        data = {
            "schema_version": 1,
            "dataset": "fineweb",
            "semantics_id": "flat-next-token-v1",
            "artifacts": {},
            "identity": "data-one",
        }
        kwargs = {
            "code": {"source_fingerprint_sha256": "code-one", "head": "abc"},
            "runtime": {"python": {"version": "3.10.19"}, "packages": {}},
        }
        preflight = module.build_run_manifest(args, data, **kwargs)
        parameter = Param()
        first_plan = module.resolve_optimization_plan(
            preflight["optimization_plan"],
            FirstOptimizer(parameter),
            [("weight", parameter)],
        )
        second_plan = module.resolve_optimization_plan(
            preflight["optimization_plan"],
            SecondOptimizer(parameter),
            [("weight", parameter)],
        )
        current = module.build_run_manifest(
            args,
            data,
            optimization_plan=first_plan,
            **kwargs,
        )
        different = module.build_run_manifest(
            args,
            data,
            optimization_plan=second_plan,
            **kwargs,
        )
        changed_protocol_args = types.SimpleNamespace(**vars(args))
        changed_protocol_args.final_eval_batches = 7
        changed_protocol_preflight = module.build_run_manifest(
            changed_protocol_args,
            data,
            **kwargs,
        )
        changed_protocol = module.build_run_manifest(
            changed_protocol_args,
            data,
            optimization_plan=first_plan,
            **kwargs,
        )

        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "run_manifest.json"
            module.write_json_atomic(path, {**current, "value": 1})
            original_bytes = path.read_bytes()
            self.assertEqual(
                module.ensure_compatible_manifest(path, current),
                "keep",
            )
            self.assertEqual(
                module.reconcile_run_manifest(path, current),
                "keep",
            )
            self.assertEqual(path.read_bytes(), original_bytes)
            with self.assertRaisesRegex(ValueError, "run identity"):
                module.ensure_compatible_manifest(path, different)
            for candidate, phase in (
                (changed_protocol_preflight, "preflight"),
                (changed_protocol, "resolved"),
            ):
                with self.subTest(protocol_phase=phase):
                    with self.assertRaisesRegex(
                        ValueError, "evaluation protocol identity"
                    ):
                        module.ensure_compatible_manifest(
                            path,
                            candidate,
                            phase=phase,
                        )
                    self.assertEqual(path.read_bytes(), original_bytes)

            self.assertEqual(
                module.ensure_compatible_manifest(
                    path,
                    preflight,
                    phase="preflight",
                ),
                "keep",
            )
            self.assertFalse(
                module.write_json_atomic_if_missing(path, {"should": "not overwrite"})
            )
            self.assertEqual(json.loads(path.read_text())["value"], 1)

            upgrade_path = Path(directory) / "upgrade.json"
            with self.assertRaisesRegex(ValueError, "existing schema-v3 resolved"):
                module.require_resolved_run_manifest(upgrade_path)
            self.assertEqual(
                module.reconcile_run_manifest(
                    upgrade_path,
                    preflight,
                    phase="preflight",
                ),
                "create",
            )
            with self.assertRaisesRegex(ValueError, "preflight manifest"):
                module.require_resolved_run_manifest(upgrade_path)
            self.assertEqual(
                module.reconcile_run_manifest(upgrade_path, current),
                "upgrade",
            )
            self.assertEqual(
                module.require_resolved_run_manifest(upgrade_path)["run_identity"],
                current["run_identity"],
            )
            upgraded_bytes = upgrade_path.read_bytes()
            self.assertEqual(
                module.reconcile_run_manifest(upgrade_path, current),
                "keep",
            )
            self.assertEqual(upgrade_path.read_bytes(), upgraded_bytes)

            module.write_json_atomic(upgrade_path, preflight)
            original_ensure = module.ensure_compatible_manifest
            ensure_calls = 0

            def simulate_competing_upgrade(*call_args, **call_kwargs):
                nonlocal ensure_calls
                ensure_calls += 1
                if ensure_calls == 2:
                    module.write_json_atomic(upgrade_path, different)
                return original_ensure(*call_args, **call_kwargs)

            with patch.object(
                module,
                "ensure_compatible_manifest",
                side_effect=simulate_competing_upgrade,
            ):
                with self.assertRaisesRegex(ValueError, "run identity"):
                    module.reconcile_run_manifest(upgrade_path, current)
            self.assertEqual(
                json.loads(upgrade_path.read_text())["run_identity"],
                different["run_identity"],
            )

            invalid = {**current, "manifest_state": "corrupt"}
            module.write_json_atomic(path, invalid)
            invalid_bytes = path.read_bytes()
            with self.assertRaisesRegex(ValueError, "state"):
                module.ensure_compatible_manifest(
                    path,
                    preflight,
                    phase="preflight",
                )
            self.assertEqual(path.read_bytes(), invalid_bytes)

            legacy_bytes = b'{"schema_version": 2, "run_identity": "same-run"}\n'
            path.write_bytes(legacy_bytes)
            with self.assertRaisesRegex(ValueError, "legacy manifests are read-only"):
                module.ensure_compatible_manifest(
                    path,
                    preflight,
                    phase="preflight",
                )
            self.assertEqual(path.read_bytes(), legacy_bytes)

            self.assertFalse(any(Path(directory).glob(".run_manifest.json.*")))


if __name__ == "__main__":
    unittest.main()
