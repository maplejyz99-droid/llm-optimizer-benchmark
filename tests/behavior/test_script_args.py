import json
import unittest
from pathlib import Path

from tests._helpers.behavior_harness import (
    REPO_ROOT,
    extract_main_argv_from_script,
    extract_main_commands_from_script,
    parse_base_args,
)


SCRIPT_MANIFEST_PATH = REPO_ROOT / "scripts" / "script_manifest.json"


def load_script_manifest():
    return json.loads(SCRIPT_MANIFEST_PATH.read_text())


class ScriptArgumentBehaviorTest(unittest.TestCase):
    def test_representative_training_scripts_still_parse(self):
        script_paths = [
            REPO_ROOT / "scripts" / "124m" / "adamw.sh",
            REPO_ROOT / "scripts" / "124m" / "muon.sh",
            REPO_ROOT / "scripts" / "124m" / "newton-muon.sh",
            REPO_ROOT / "scripts" / "124m" / "gn-prox.sh",
            REPO_ROOT / "scripts" / "210m" / "mars.sh",
            REPO_ROOT / "scripts" / "720m" / "adamw.sh",
            REPO_ROOT / "scripts" / "moe-520m" / "adamw.sh",
        ]

        parsed = []
        for script_path in script_paths:
            argvs = extract_main_argv_from_script(script_path)
            self.assertGreater(len(argvs), 0, f"no torchrun command found in {script_path}")
            for argv in argvs:
                args, _ = parse_base_args(argv)
                parsed.append((script_path, args.opt, args.model, args.dataset))

        self.assertIn((REPO_ROOT / "scripts" / "124m" / "newton-muon.sh", "newton-muon", "llama", "fineweb"), parsed)
        self.assertIn((REPO_ROOT / "scripts" / "moe-520m" / "adamw.sh", "adamw", "llama", "fineweb"), parsed)

    def test_all_simple_torchrun_scripts_still_parse(self):
        failures = []
        parsed_count = 0
        for script_path in sorted((REPO_ROOT / "scripts").glob("*/*.sh")):
            if script_path.name == "memory-probe-500step.sh":
                continue
            for argv in extract_main_argv_from_script(script_path):
                parsed_count += 1
                try:
                    parse_base_args(argv)
                except SystemExit as exc:
                    failures.append((Path(script_path).as_posix(), argv, exc.code))

        self.assertEqual(failures, [])
        self.assertGreater(parsed_count, 70)

    def test_script_manifest_covers_current_script_inventory(self):
        manifest = load_script_manifest()
        manifest_paths = [entry["path"] for entry in manifest["entries"]]
        script_paths = [
            script_path.relative_to(REPO_ROOT).as_posix()
            for script_path in sorted((REPO_ROOT / "scripts").glob("*/*.sh"))
        ]

        self.assertEqual(manifest["version"], 1)
        self.assertEqual(manifest_paths, script_paths)
        self.assertEqual(len(manifest_paths), 75)
        self.assertEqual(
            [entry["path"] for entry in manifest["entries"] if entry.get("metadata_only")],
            ["scripts/124m/memory-probe-500step.sh"],
        )

    def test_script_manifest_matches_existing_static_commands(self):
        manifest = load_script_manifest()
        total_commands = 0
        for entry in manifest["entries"]:
            script_path = REPO_ROOT / entry["path"]
            if entry.get("metadata_only"):
                self.assertEqual(entry["commands"], [])
                self.assertGreater(entry.get("dynamic_main_command_count", 0), 0)
                continue

            manifest_commands = [
                {
                    "launcher": command["launcher"],
                    "main": command["main"],
                    "argv": command["argv"],
                }
                for command in entry["commands"]
            ]
            total_commands += len(manifest_commands)
            self.assertEqual(
                manifest_commands,
                extract_main_commands_from_script(script_path),
                f"manifest drifted from {entry['path']}",
            )

        self.assertEqual(total_commands, 77)


if __name__ == "__main__":
    unittest.main()
