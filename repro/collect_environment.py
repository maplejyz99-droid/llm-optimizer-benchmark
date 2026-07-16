#!/usr/bin/env python3
"""Collect a credential-safe environment snapshot without importing torch."""

import argparse
import importlib.metadata
import json
import platform
import shutil
import subprocess
import sys
import sysconfig
import tempfile
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path


KEY_PACKAGES = (
    "torch",
    "triton",
    "numpy",
    "pyyaml",
    "tiktoken",
    "transformers",
    "huggingface-hub",
    "wandb",
    "datasets",
)


def _normalize_package_name(name):
    return name.lower().replace("_", "-")


def installed_distributions():
    packages = []
    for distribution in importlib.metadata.distributions():
        name = distribution.metadata.get("Name")
        if not name:
            continue
        packages.append(
            {
                "name": name,
                "normalized_name": _normalize_package_name(name),
                "version": distribution.version,
            }
        )
    return sorted(
        packages,
        key=lambda item: (item["normalized_name"], item["version"], item["name"]),
    )


def duplicate_distributions(packages):
    versions = defaultdict(set)
    for package in packages:
        versions[package["normalized_name"]].add(package["version"])
    return {
        name: sorted(found_versions)
        for name, found_versions in sorted(versions.items())
        if len(found_versions) > 1
    }


def command_result(command):
    try:
        result = subprocess.run(
            command,
            check=False,
            text=True,
            capture_output=True,
        )
    except OSError as exc:
        return {
            "command": list(command),
            "returncode": None,
            "stdout_lines": [],
            "stderr_lines": [str(exc)],
        }
    return {
        "command": list(command),
        "returncode": result.returncode,
        "stdout_lines": result.stdout.splitlines(),
        "stderr_lines": result.stderr.splitlines(),
    }


def collect_snapshot(label):
    packages = installed_distributions()
    versions_by_name = defaultdict(set)
    for package in packages:
        versions_by_name[package["normalized_name"]].add(package["version"])

    nvidia_smi = shutil.which("nvidia-smi")
    if nvidia_smi:
        gpu = command_result(
            [
                nvidia_smi,
                "--query-gpu=name,driver_version",
                "--format=csv,noheader",
            ]
        )
    else:
        gpu = {
            "command": ["nvidia-smi"],
            "returncode": None,
            "stdout_lines": [],
            "stderr_lines": ["nvidia-smi not found"],
        }

    return {
        "schema_version": 1,
        "label": label,
        "collected_at_utc": datetime.now(timezone.utc).isoformat(),
        "python": {
            "implementation": platform.python_implementation(),
            "version": platform.python_version(),
            "executable": sys.executable,
            "soabi": sysconfig.get_config_var("SOABI"),
            "pip_version": importlib.metadata.version("pip"),
        },
        "platform": {
            "system": platform.system(),
            "release": platform.release(),
            "version": platform.version(),
            "machine": platform.machine(),
            "libc": list(platform.libc_ver()),
        },
        "key_packages": {
            name: sorted(versions_by_name.get(name, set())) for name in KEY_PACKAGES
        },
        "duplicate_distributions": duplicate_distributions(packages),
        "packages": packages,
        "pip_check": command_result([sys.executable, "-m", "pip", "check"]),
        "gpu": gpu,
    }


def write_snapshot(snapshot, output_path):
    output_path.parent.mkdir(parents=True, exist_ok=True)
    serialized = json.dumps(snapshot, indent=2, sort_keys=True) + "\n"
    with tempfile.NamedTemporaryFile(
        mode="w",
        encoding="utf-8",
        dir=output_path.parent,
        prefix=f".{output_path.name}.",
        delete=False,
    ) as temporary:
        temporary.write(serialized)
        temporary_path = Path(temporary.name)
    temporary_path.replace(output_path)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--label", required=True)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args(argv)

    snapshot = collect_snapshot(args.label)
    if args.output:
        write_snapshot(snapshot, args.output)
    else:
        json.dump(snapshot, sys.stdout, indent=2, sort_keys=True)
        sys.stdout.write("\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
