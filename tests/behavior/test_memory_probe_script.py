import os
import subprocess
import tempfile
import unittest
from pathlib import Path

from tests._helpers.behavior_harness import REPO_ROOT


SCRIPT_PATH = REPO_ROOT / "scripts" / "124m" / "memory-probe-500step.sh"


class MemoryProbeScriptBehaviorTest(unittest.TestCase):
    def run_probe_with_stubs(self, *, repo_dir=None):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            bin_dir = root / "bin"
            bin_dir.mkdir()
            capture_path = root / "python-cwd.txt"
            log_dir = root / "logs"
            conda_sh = root / "conda.sh"
            conda_sh.write_text("conda() { return 0; }\n")

            python_stub = bin_dir / "python"
            python_stub.write_text(
                "#!/bin/bash\n"
                'printf "%s\\n" "$PWD" > "$FAKE_PYTHON_CAPTURE"\n'
            )
            python_stub.chmod(0o755)

            nvidia_smi_stub = bin_dir / "nvidia-smi"
            nvidia_smi_stub.write_text("#!/bin/bash\nexit 0\n")
            nvidia_smi_stub.chmod(0o755)

            env = os.environ.copy()
            env.update(
                {
                    "CONDA_SH": str(conda_sh),
                    "FAKE_PYTHON_CAPTURE": str(capture_path),
                    "LOG_DIR": str(log_dir),
                    "PATH": f"{bin_dir}{os.pathsep}{env['PATH']}",
                    "SAMPLE_INTERVAL_SEC": "1",
                }
            )
            if repo_dir is None:
                env.pop("REPO_DIR", None)
            else:
                env["REPO_DIR"] = str(repo_dir)

            subprocess.run(
                ["bash", str(SCRIPT_PATH), "muon"],
                cwd=root,
                env=env,
                check=True,
                capture_output=True,
                text=True,
                timeout=10,
            )
            return Path(capture_path.read_text().strip()).resolve()

    def test_default_repo_dir_follows_the_script_checkout(self):
        self.assertEqual(self.run_probe_with_stubs(), REPO_ROOT.resolve())

    def test_explicit_repo_dir_override_is_preserved(self):
        with tempfile.TemporaryDirectory() as tmp:
            repo_dir = Path(tmp).resolve()
            self.assertEqual(
                self.run_probe_with_stubs(repo_dir=repo_dir),
                repo_dir,
            )

    def test_retired_remote_checkout_is_not_hardcoded(self):
        self.assertNotIn(
            "/root/work/llm-optimizer-benchmark",
            SCRIPT_PATH.read_text(),
        )


if __name__ == "__main__":
    unittest.main()
