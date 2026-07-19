import json
import os
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

from tests._helpers.behavior_harness import REPO_ROOT, SRC_ROOT


WORKER = REPO_ROOT / "tests" / "_helpers" / "distributed_behavior_worker.py"


class DistributedControlPlaneIntegrationTest(unittest.TestCase):
    def test_long_master_actions_use_control_group_and_reduce_uses_training_group(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            env = dict(os.environ)
            env["PYTHONPATH"] = str(SRC_ROOT)
            result = subprocess.run(
                [
                    sys.executable,
                    str(WORKER),
                    "--mode",
                    "control",
                    "--result-dir",
                    tmpdir,
                ],
                cwd=REPO_ROOT,
                env=env,
                check=False,
                capture_output=True,
                text=True,
                timeout=45,
            )
            self.assertEqual(
                result.returncode,
                0,
                msg=f"stdout:\n{result.stdout}\nstderr:\n{result.stderr}",
            )
            results = [
                json.loads((Path(tmpdir) / f"rank-{rank}.json").read_text())
                for rank in range(2)
            ]

        self.assertEqual([item["status"] for item in results], ["ok", "ok"])
        self.assertEqual(results[0]["success_result"], "rank-zero-result")
        self.assertIsNone(results[1]["success_result"])
        self.assertEqual([item["reduced_mean"] for item in results], [1.5, 1.5])
        self.assertEqual(
            [item["error"]["type"] for item in results],
            ["RuntimeError", "RuntimeError"],
        )
        self.assertTrue(
            all(
                "synthetic slow-action failure" in item["error"]["message"]
                for item in results
            )
        )


if __name__ == "__main__":
    unittest.main()
