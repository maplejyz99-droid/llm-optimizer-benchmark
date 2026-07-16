#!/usr/bin/env python3
"""Run behavior tests with explicit process and order-isolation contracts."""

import argparse
import hashlib
import os
import random
import stat
import subprocess
import sys
import unittest
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
TEST_ROOT = REPO_ROOT / "tests"
BEHAVIOR_ROOT = TEST_ROOT / "behavior"
PROCESS_ENVIRONMENT = {
    "PYTHONDONTWRITEBYTECODE": "1",
    "PYTHONHASHSEED": "0",
    "OMP_NUM_THREADS": "1",
    "CUDA_VISIBLE_DEVICES": "",
    "WANDB_MODE": "disabled",
}
REEXEC_MARKER = "LLMOPT_BEHAVIOR_RUNNER_REEXECUTED"
MAX_UNTRACKED_HASH_BYTES = 16 * 1024 * 1024

if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))


def discover_test_modules():
    return [
        ".".join(path.relative_to(REPO_ROOT).with_suffix("").parts)
        for path in sorted(BEHAVIOR_ROOT.glob("test_*.py"))
    ]


def flatten_suite(suite):
    for item in suite:
        if isinstance(item, unittest.TestSuite):
            yield from flatten_suite(item)
        else:
            yield item


def discovered_tests():
    loader = unittest.TestLoader()
    suite = loader.discover(
        start_dir=str(TEST_ROOT),
        top_level_dir=str(REPO_ROOT),
    )
    return list(flatten_suite(suite))


def git_output(arguments, text=False):
    result = subprocess.run(
        ["git", *arguments],
        cwd=REPO_ROOT,
        check=True,
        text=text,
        capture_output=True,
    )
    return result.stdout


def file_sha256(path):
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def repo_fingerprint():
    tracked_diff = git_output(["diff", "--binary", "--no-ext-diff", "HEAD", "--"])
    untracked_paths = git_output(
        ["ls-files", "--others", "--exclude-standard", "-z"]
    ).split(b"\0")
    untracked = []
    for raw_path in untracked_paths:
        if not raw_path:
            continue
        relative_path = Path(os.fsdecode(raw_path))
        path = REPO_ROOT / relative_path
        metadata = path.lstat()
        entry = {
            "path": relative_path.as_posix(),
            "mode": stat.S_IMODE(metadata.st_mode),
            "size": metadata.st_size,
            "mtime_ns": metadata.st_mtime_ns,
        }
        if stat.S_ISREG(metadata.st_mode) and metadata.st_size <= MAX_UNTRACKED_HASH_BYTES:
            entry["sha256"] = file_sha256(path)
        elif stat.S_ISLNK(metadata.st_mode):
            entry["target"] = os.readlink(path)
        untracked.append(entry)
    return {
        "tracked_diff_sha256": hashlib.sha256(tracked_diff).hexdigest(),
        "untracked": untracked,
    }


def repo_status():
    return git_output(
        ["status", "--porcelain=v1", "--untracked-files=all"], text=True
    )


def execute_suite(suite, label):
    print(f"[{label}]", flush=True)
    result = unittest.TextTestRunner(verbosity=1).run(suite)
    if result.skipped:
        print(f"{label}: unexpected skipped tests:", file=sys.stderr)
        for test, reason in result.skipped:
            print(f"  {test.id()}: {reason}", file=sys.stderr)
    return result.wasSuccessful() and not result.skipped


def run_full():
    return execute_suite(unittest.TestSuite(discovered_tests()), "full")


def tests_loaded_in_module_order(modules, reverse_within_modules=False):
    loader = unittest.TestLoader()
    tests = []
    for module in modules:
        module_tests = list(flatten_suite(loader.loadTestsFromName(module)))
        if reverse_within_modules:
            module_tests.reverse()
        tests.extend(module_tests)
    return tests


def run_ordered(mode, seed):
    modules = discover_test_modules()
    randomizer = random.Random(seed)
    if mode == "reverse":
        modules.reverse()
        tests = tests_loaded_in_module_order(modules, reverse_within_modules=True)
        label = "reverse"
    else:
        randomizer.shuffle(modules)
        tests = tests_loaded_in_module_order(modules)
        randomizer.shuffle(tests)
        label = f"shuffled seed={seed}"
    return execute_suite(unittest.TestSuite(tests), label)


def run_module(module):
    suite = unittest.defaultTestLoader.loadTestsFromName(module)
    return execute_suite(suite, module)


def child_environment():
    environment = dict(os.environ)
    environment.update(PROCESS_ENVIRONMENT)
    return environment


def ensure_process_environment():
    needs_restart = any(
        os.environ.get(name) != value for name, value in PROCESS_ENVIRONMENT.items()
    ) or not sys.dont_write_bytecode
    if not needs_restart:
        return
    if os.environ.get(REEXEC_MARKER) == "1":
        raise RuntimeError("Could not establish the deterministic test environment")
    environment = child_environment()
    environment[REEXEC_MARKER] = "1"
    os.execve(
        sys.executable,
        [sys.executable, str(Path(__file__).resolve()), *sys.argv[1:]],
        environment,
    )


def run_isolated():
    modules = discover_test_modules()
    if not modules:
        print("No behavior test modules discovered.", file=sys.stderr)
        return False
    passed = True
    for module in modules:
        result = subprocess.run(
            [sys.executable, str(Path(__file__).resolve()), "module", "--module", module],
            cwd=REPO_ROOT,
            env=child_environment(),
            check=False,
        )
        passed = result.returncode == 0 and passed
    return passed


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "mode",
        choices=("full", "isolated", "reverse", "shuffled", "module"),
    )
    parser.add_argument("--seed", type=int, default=20260716)
    parser.add_argument("--module")
    args = parser.parse_args(argv)

    ensure_process_environment()
    before = repo_fingerprint()
    if args.mode == "full":
        passed = run_full()
    elif args.mode == "isolated":
        passed = run_isolated()
    elif args.mode in {"reverse", "shuffled"}:
        passed = run_ordered(args.mode, args.seed)
    else:
        if not args.module:
            parser.error("module mode requires --module")
        passed = run_module(args.module)

    after = repo_fingerprint()
    if after != before:
        print("Tests changed the repository worktree.", file=sys.stderr)
        print(f"Before fingerprint: {before!r}", file=sys.stderr)
        print(f"After fingerprint: {after!r}", file=sys.stderr)
        print("Current Git status:", file=sys.stderr)
        print(repo_status() or "<clean>", file=sys.stderr)
        passed = False
    return 0 if passed else 1


if __name__ == "__main__":
    raise SystemExit(main())
