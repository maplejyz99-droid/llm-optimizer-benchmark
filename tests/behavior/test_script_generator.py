import json
import os
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

from tests._helpers.behavior_harness import (
    REPO_ROOT,
    extract_main_commands_from_script,
    parse_base_args,
)


GENERATOR_PATH = REPO_ROOT / "scripts" / "generate_scripts.py"
SCRIPT_MANIFEST_PATH = REPO_ROOT / "scripts" / "script_manifest.json"


def load_script_manifest():
    return json.loads(SCRIPT_MANIFEST_PATH.read_text())


class ScriptGeneratorBehaviorTest(unittest.TestCase):
    def test_generator_renders_preview_scripts_matching_existing_commands(self):
        manifest = load_script_manifest()
        env = dict(os.environ, PYTHONDONTWRITEBYTECODE="1")

        with tempfile.TemporaryDirectory() as temp_dir:
            output_dir = Path(temp_dir) / "generated-scripts"
            result = subprocess.run(
                [
                    sys.executable,
                    str(GENERATOR_PATH),
                    "--manifest",
                    str(SCRIPT_MANIFEST_PATH),
                    "--output-dir",
                    str(output_dir),
                ],
                cwd=REPO_ROOT,
                env=env,
                check=True,
                text=True,
                capture_output=True,
            )

            self.assertIn("Generated 74 preview scripts", result.stdout)
            self.assertFalse((output_dir / "scripts/124m/memory-probe-500step.sh").exists())

            rendered_count = 0
            for entry in manifest["entries"]:
                if entry.get("metadata_only"):
                    continue

                source_script = REPO_ROOT / entry["path"]
                generated_script = output_dir / entry["path"]
                self.assertTrue(generated_script.exists(), f"missing generated script {entry['path']}")

                source_commands = extract_main_commands_from_script(source_script)
                generated_commands = extract_main_commands_from_script(generated_script)
                self.assertEqual(generated_commands, source_commands, entry["path"])

                for command in generated_commands:
                    parse_base_args(command["argv"])
                    rendered_count += 1

            self.assertEqual(rendered_count, 77)

    def test_generator_refuses_to_overwrite_authoritative_scripts_dir(self):
        env = dict(os.environ, PYTHONDONTWRITEBYTECODE="1")
        result = subprocess.run(
            [
                sys.executable,
                str(GENERATOR_PATH),
                "--manifest",
                str(SCRIPT_MANIFEST_PATH),
                "--output-dir",
                str(REPO_ROOT / "scripts"),
            ],
            cwd=REPO_ROOT,
            env=env,
            check=False,
            text=True,
            capture_output=True,
        )

        self.assertNotEqual(result.returncode, 0)
        self.assertIn("Refusing to write generated previews over scripts/", result.stderr)


if __name__ == "__main__":
    unittest.main()
