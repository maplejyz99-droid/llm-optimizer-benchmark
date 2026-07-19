import os
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from scripts import setup_runtime_paths


class RuntimePathsBehaviorTest(unittest.TestCase):
    def test_dry_run_has_no_side_effects(self):
        with tempfile.TemporaryDirectory() as tmp_dir:
            root = Path(tmp_dir)
            repo = root / "repo"
            repo.mkdir()
            runs = root / "runtime" / "runs"
            results = root / "curated" / "results"

            actions = setup_runtime_paths.setup_runtime_paths(
                repo, runs, results, dry_run=True
            )

            self.assertFalse(runs.exists())
            self.assertFalse(results.exists())
            self.assertFalse(os.path.lexists(repo / "runs"))
            self.assertFalse(os.path.lexists(repo / "results"))
            self.assertTrue(all(action["create_link"] for action in actions))

    def test_setup_is_successful_and_idempotent(self):
        with tempfile.TemporaryDirectory() as tmp_dir:
            root = Path(tmp_dir)
            repo = root / "repo"
            repo.mkdir()
            runs = root / "runtime" / "runs"
            results = root / "curated" / "results"

            first = setup_runtime_paths.setup_runtime_paths(repo, runs, results)
            second = setup_runtime_paths.setup_runtime_paths(repo, runs, results)

            self.assertTrue(runs.is_dir())
            self.assertTrue(results.is_dir())
            self.assertTrue((repo / "runs").is_symlink())
            self.assertTrue((repo / "results").is_symlink())
            self.assertEqual((repo / "runs").resolve(), runs.resolve())
            self.assertEqual((repo / "results").resolve(), results.resolve())
            self.assertTrue(all(action["create_link"] for action in first))
            self.assertTrue(all(not action["create_link"] for action in second))

    def test_relative_paths_and_spaces_resolve_from_current_directory(self):
        with tempfile.TemporaryDirectory() as tmp_dir:
            root = Path(tmp_dir)
            repo = root / "repo with spaces"
            repo.mkdir()
            previous_directory = Path.cwd()
            try:
                os.chdir(root)
                setup_runtime_paths.setup_runtime_paths(
                    Path("repo with spaces"),
                    Path("runtime roots/runs"),
                    Path("runtime roots/results"),
                )
            finally:
                os.chdir(previous_directory)

            self.assertEqual(
                (repo / "runs").resolve(), (root / "runtime roots" / "runs").resolve()
            )
            self.assertEqual(
                (repo / "results").resolve(),
                (root / "runtime roots" / "results").resolve(),
            )

    def test_conflict_preflight_prevents_partial_creation(self):
        with tempfile.TemporaryDirectory() as tmp_dir:
            root = Path(tmp_dir)
            repo = root / "repo"
            repo.mkdir()
            (repo / "results").write_text("owned", encoding="utf-8")
            runs = root / "runtime" / "runs"
            results = root / "runtime" / "results"

            with self.assertRaisesRegex(ValueError, "not a symlink"):
                setup_runtime_paths.setup_runtime_paths(repo, runs, results)

            self.assertFalse(runs.exists())
            self.assertFalse(results.exists())
            self.assertFalse(os.path.lexists(repo / "runs"))

    def test_wrong_and_broken_symlinks_are_rejected(self):
        for broken in (False, True):
            with self.subTest(broken=broken), tempfile.TemporaryDirectory() as tmp_dir:
                root = Path(tmp_dir)
                repo = root / "repo"
                repo.mkdir()
                wrong = root / ("missing" if broken else "wrong")
                if not broken:
                    wrong.mkdir()
                (repo / "runs").symlink_to(wrong, target_is_directory=True)

                with self.assertRaisesRegex(ValueError, "points elsewhere"):
                    setup_runtime_paths.setup_runtime_paths(
                        repo,
                        root / "expected-runs",
                        root / "expected-results",
                    )

    def test_target_file_is_rejected_before_links_are_created(self):
        with tempfile.TemporaryDirectory() as tmp_dir:
            root = Path(tmp_dir)
            repo = root / "repo"
            repo.mkdir()
            runs = root / "runs-file"
            runs.write_text("not a directory", encoding="utf-8")

            with self.assertRaisesRegex(ValueError, "not a directory"):
                setup_runtime_paths.setup_runtime_paths(
                    repo, runs, root / "results"
                )

            self.assertFalse(os.path.lexists(repo / "runs"))
            self.assertFalse(os.path.lexists(repo / "results"))

    def test_targets_may_not_create_repository_cycles(self):
        with tempfile.TemporaryDirectory() as tmp_dir:
            root = Path(tmp_dir)
            repo = root / "repo"
            repo.mkdir()
            external = root / "external-results"
            cases = (repo, repo / "artifacts", root)
            for runs in cases:
                with self.subTest(runs=runs), self.assertRaisesRegex(
                    ValueError, "repository"
                ):
                    setup_runtime_paths.setup_runtime_paths(repo, runs, external)

            self.assertFalse(os.path.lexists(repo / "runs"))
            self.assertFalse(os.path.lexists(repo / "results"))

    def test_runs_and_results_roots_may_not_be_equal_or_nested(self):
        with tempfile.TemporaryDirectory() as tmp_dir:
            root = Path(tmp_dir)
            repo = root / "repo"
            repo.mkdir()
            runtime = root / "runtime"
            cases = (
                (runtime, runtime),
                (runtime, runtime / "results"),
                (runtime / "runs", runtime),
            )
            for runs, results in cases:
                with self.subTest(runs=runs, results=results), self.assertRaisesRegex(
                    ValueError, "non-nested"
                ):
                    setup_runtime_paths.setup_runtime_paths(repo, runs, results)

    def test_failed_link_creation_rolls_back_created_parent_directories(self):
        with tempfile.TemporaryDirectory() as tmp_dir:
            root = Path(tmp_dir)
            repo = root / "repo"
            repo.mkdir()
            runtime_parent = root / "new-parent"
            plans = setup_runtime_paths.build_plan(
                repo,
                runtime_parent / "runs",
                root / "other-parent" / "results",
            )

            with patch.object(Path, "symlink_to", side_effect=OSError("simulated link failure")):
                with self.assertRaisesRegex(OSError, "simulated"):
                    setup_runtime_paths.apply_plan(plans)

            self.assertFalse(runtime_parent.exists())
            self.assertFalse((root / "other-parent").exists())


if __name__ == "__main__":
    unittest.main()
