import json
import os
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

from tests._helpers.behavior_harness import REPO_ROOT, SRC_ROOT


WORKER = REPO_ROOT / "tests" / "_helpers" / "distributed_behavior_worker.py"


class ScheduleFreeDistributedModeIntegrationTest(unittest.TestCase):
    def test_all_ranks_apply_the_same_noninvertible_eval_train_round_trip(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            env = dict(os.environ)
            env["PYTHONPATH"] = str(SRC_ROOT)
            result = subprocess.run(
                [
                    sys.executable,
                    str(WORKER),
                    "--mode",
                    "schedulefree",
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

        self.assertEqual([result["status"] for result in results], ["ok", "ok"])
        for result in results:
            self.assertEqual(result["rank0_weight"], result["rank1_weight"])
            self.assertNotEqual(result["rank0_weight"], result["original_weight"])
            self.assertIs(result["train_mode"], True)


if __name__ == "__main__":
    unittest.main()
