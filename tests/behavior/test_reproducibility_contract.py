import hashlib
import json
import os
import re
import sys
import tempfile
import types
import unittest
from pathlib import Path
from unittest.mock import patch

from tests._helpers.behavior_harness import REPO_ROOT


class ReproducibilityContractTest(unittest.TestCase):
    def test_project_module_isolation_restores_the_exact_cache(self):
        from tests._helpers.behavior_harness import isolated_modules

        original = types.ModuleType("repro_contract_package")
        replacement = types.ModuleType("repro_contract_package.child")
        sys.modules["repro_contract_package"] = original
        try:
            with isolated_modules("repro_contract_package"):
                self.assertNotIn("repro_contract_package", sys.modules)
                sys.modules["repro_contract_package.child"] = replacement
            self.assertIs(sys.modules["repro_contract_package"], original)
            self.assertNotIn("repro_contract_package.child", sys.modules)
        finally:
            sys.modules.pop("repro_contract_package", None)
            sys.modules.pop("repro_contract_package.child", None)

    def test_training_harness_restores_the_optim_module_cache(self):
        from tests._helpers.training_harness import loaded_training_base_module

        before = {
            name: module
            for name, module in sys.modules.items()
            if name == "optim" or name.startswith("optim.")
        }
        with loaded_training_base_module({"events": []}):
            pass
        after = {
            name: module
            for name, module in sys.modules.items()
            if name == "optim" or name.startswith("optim.")
        }
        self.assertEqual(after, before)

    def test_repo_fingerprint_detects_same_status_untracked_content_changes(self):
        from repro import run_behavior_tests

        def fake_git_output(arguments, text=False):
            del text
            if arguments[0] == "diff":
                return b"unchanged tracked diff"
            if arguments[0] == "ls-files":
                return b"artifact.txt\0"
            raise AssertionError(arguments)

        with tempfile.TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            artifact = root / "artifact.txt"
            with patch.object(run_behavior_tests, "REPO_ROOT", root), patch.object(
                run_behavior_tests, "git_output", side_effect=fake_git_output
            ):
                artifact.write_text("first", encoding="utf-8")
                before = run_behavior_tests.repo_fingerprint()
                artifact.write_text("other", encoding="utf-8")
                after = run_behavior_tests.repo_fingerprint()
        self.assertNotEqual(after, before)

    def test_environment_snapshot_does_not_capture_environment_values(self):
        from repro.collect_environment import collect_snapshot

        sentinel = "REPRO_SECRET_SENTINEL_DO_NOT_CAPTURE"
        with patch.dict(os.environ, {"REPRO_SECRET_SENTINEL": sentinel}, clear=False):
            snapshot = collect_snapshot(label="unit-test")

        serialized = json.dumps(snapshot, sort_keys=True)
        self.assertNotIn(sentinel, serialized)
        self.assertEqual(snapshot["schema_version"], 1)
        self.assertEqual(snapshot["label"], "unit-test")
        self.assertIn("python", snapshot)
        self.assertIn("platform", snapshot)
        self.assertIn("packages", snapshot)
        self.assertIn("pip_check", snapshot)

    def test_test_runner_discovers_every_behavior_module(self):
        from repro.run_behavior_tests import discover_test_modules

        expected = {
            ".".join(path.relative_to(REPO_ROOT).with_suffix("").parts)
            for path in sorted((REPO_ROOT / "tests" / "behavior").glob("test_*.py"))
            if path.name != "test_reproducibility_contract.py"
        }
        expected.add("tests.behavior.test_reproducibility_contract")

        self.assertEqual(set(discover_test_modules()), expected)

    def test_reverse_order_keeps_modules_and_their_tests_reversed(self):
        from repro import run_behavior_tests

        def make_suite(module):
            cases = []
            for position in (1, 2):
                case = unittest.FunctionTestCase(lambda: None)
                case.module_label = module
                case.position = position
                cases.append(case)
            return unittest.TestSuite(cases)

        loader = types.SimpleNamespace(loadTestsFromName=make_suite)
        with patch.object(run_behavior_tests.unittest, "TestLoader", return_value=loader):
            tests = run_behavior_tests.tests_loaded_in_module_order(
                ["module_b", "module_a"], reverse_within_modules=True
            )

        self.assertEqual(
            [(test.module_label, test.position) for test in tests],
            [("module_b", 2), ("module_b", 1), ("module_a", 2), ("module_a", 1)],
        )

    def test_cpu_lock_is_pinned_hashed_and_kept_separate_from_runtime_requirements(self):
        direct = (REPO_ROOT / "requirements-ci.in").read_text()
        lock = (REPO_ROOT / "requirements-ci.lock").read_text()

        for requirement in (
            "numpy==2.2.6",
            "pyyaml==6.0.3",
            "tiktoken==0.12.0",
            "torch==2.9.1+cpu",
        ):
            self.assertIn(requirement, direct.lower())
        self.assertIn("--hash=sha256:", lock)
        self.assertIn("--only-binary=:all:", lock)
        self.assertIn("torch==2.9.1+cpu", lock)
        self.assertNotRegex(lock.lower(), r"(?m)^(?:triton|nvidia-)")
        requirement_lines = [
            index
            for index, line in enumerate(lock.splitlines())
            if re.fullmatch(r"[a-z0-9-]+==\S+ \\$", line)
        ]
        hash_lines = [
            line
            for line in lock.splitlines()
            if line.strip().startswith("--hash=sha256:")
        ]
        self.assertEqual(len(requirement_lines), len(hash_lines))
        self.assertGreater(len(requirement_lines), 4)
        lock_lines = lock.splitlines()
        for index in requirement_lines:
            self.assertRegex(lock_lines[index + 1], r"^    --hash=sha256:[0-9a-f]{64}$")
        input_hash = hashlib.sha256((REPO_ROOT / "requirements-ci.in").read_bytes()).hexdigest()
        self.assertIn(f"# Input SHA256: {input_hash}", lock)
        self.assertNotEqual(direct, (REPO_ROOT / "requirements.txt").read_text())

    def test_ci_runs_full_isolated_and_reverse_modes_from_the_lock(self):
        workflow = (
            REPO_ROOT / ".github" / "workflows" / "behavior-tests.yml"
        ).read_text()

        self.assertIn("requirements-ci.lock", workflow)
        self.assertIn("--require-hashes", workflow)
        for mode in ("full", "isolated", "reverse"):
            self.assertIn(mode, workflow)
        self.assertIn("3.10.19", workflow)
        self.assertIn("git status --porcelain", workflow)


if __name__ == "__main__":
    unittest.main()
