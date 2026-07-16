import json
import tempfile
import types
import unittest
from pathlib import Path

import numpy as np

from tests._helpers.behavior_harness import SRC_ROOT, load_module_from_path


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

    def test_atomic_manifest_write_and_compatibility_check(self):
        module = load_run_manifest_module()
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "run_manifest.json"
            module.write_json_atomic(path, {"run_identity": "same", "value": 1})
            module.ensure_compatible_manifest(path, {"run_identity": "same"})
            with self.assertRaisesRegex(ValueError, "run identity"):
                module.ensure_compatible_manifest(path, {"run_identity": "different"})

        self.assertFalse(any(Path(directory).glob(".run_manifest.json.*")))


if __name__ == "__main__":
    unittest.main()
