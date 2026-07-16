import unittest
from pathlib import Path

from tests._helpers.behavior_harness import (
    REPO_ROOT,
    extract_main_argv_from_script,
    parse_base_args,
)


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


if __name__ == "__main__":
    unittest.main()
