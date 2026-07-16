#!/usr/bin/env python3
"""Build a platform-specific, hash-locked requirements file from wheels."""

import argparse
import hashlib
import importlib.metadata
import platform
import re
import sys
import tempfile
import zipfile
from pathlib import Path


TARGET_SYSTEM = "Linux"
TARGET_MACHINE = "x86_64"
TARGET_PYTHON = (3, 10)
FORBIDDEN_PACKAGES = ("triton", "nvidia-")


def normalize_package_name(name):
    return re.sub(r"[-_.]+", "-", name).lower()


def parse_input(path):
    options = []
    requirements = {}
    for line_number, raw_line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
        line = raw_line.strip()
        if not line or line.startswith("#"):
            continue
        if line.startswith(("--index-url ", "--extra-index-url ")):
            options.append(line)
            continue
        match = re.fullmatch(r"([A-Za-z0-9][A-Za-z0-9._-]*)==([^\s;]+)", line)
        if not match:
            raise ValueError(f"Unsupported input at {path}:{line_number}: {line!r}")
        name = normalize_package_name(match.group(1))
        version = match.group(2)
        if name in requirements:
            raise ValueError(f"Duplicate direct requirement: {name}")
        requirements[name] = version
    if not options:
        raise ValueError("The input must declare its package index URLs")
    if not requirements:
        raise ValueError("The input contains no direct requirements")
    return options, requirements


def wheel_metadata(path):
    with zipfile.ZipFile(path) as archive:
        candidates = [
            name for name in archive.namelist() if name.endswith(".dist-info/METADATA")
        ]
        if len(candidates) != 1:
            raise ValueError(f"Expected one METADATA file in {path}, found {candidates!r}")
        metadata = archive.read(candidates[0]).decode("utf-8")

    fields = {}
    for line in metadata.splitlines():
        if line.startswith("Name: "):
            fields["name"] = line.removeprefix("Name: ").strip()
        elif line.startswith("Version: "):
            fields["version"] = line.removeprefix("Version: ").strip()
        if len(fields) == 2:
            break
    if set(fields) != {"name", "version"}:
        raise ValueError(f"Missing Name or Version metadata in {path}")
    return normalize_package_name(fields["name"]), fields["version"]


def sha256(path):
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def collect_wheels(wheel_dir):
    packages = {}
    wheel_paths = sorted(wheel_dir.glob("*.whl"))
    if not wheel_paths:
        raise ValueError(f"No wheels found in {wheel_dir}")
    for wheel_path in wheel_paths:
        name, version = wheel_metadata(wheel_path)
        if name == FORBIDDEN_PACKAGES[0] or name.startswith(FORBIDDEN_PACKAGES[1]):
            raise ValueError(f"CUDA package leaked into the CPU wheel set: {name}")
        if name in packages:
            raise ValueError(f"Multiple wheels found for {name}")
        packages[name] = {
            "version": version,
            "sha256": sha256(wheel_path),
        }
    return packages


def build_lock(input_path, wheel_dir, resolver_version=None):
    options, direct_requirements = parse_input(input_path)
    packages = collect_wheels(wheel_dir)
    missing = sorted(set(direct_requirements) - set(packages))
    if missing:
        raise ValueError(f"Direct requirements missing from wheel set: {missing!r}")
    mismatches = {
        name: (expected, packages[name]["version"])
        for name, expected in direct_requirements.items()
        if packages[name]["version"] != expected
    }
    if mismatches:
        raise ValueError(f"Direct requirement versions do not match wheels: {mismatches!r}")

    input_hash = sha256(input_path)
    resolver_version = resolver_version or importlib.metadata.version("pip")
    lines = [
        "# Generated from requirements-ci.in and the resolved wheel set.",
        "# Target: CPython 3.10 / Linux x86_64",
        f"# Resolver: pip {resolver_version}",
        f"# Input SHA256: {input_hash}",
        "# Regenerate with the documented workflow; do not edit by hand.",
        *options,
        "--only-binary=:all:",
        "",
    ]
    for name in sorted(packages):
        package = packages[name]
        lines.extend(
            [
                f"{name}=={package['version']} \\",
                f"    --hash=sha256:{package['sha256']}",
            ]
        )
    return "\n".join(lines) + "\n"


def validate_target():
    actual = (platform.system(), platform.machine(), sys.version_info[:2])
    expected = (TARGET_SYSTEM, TARGET_MACHINE, TARGET_PYTHON)
    if actual != expected:
        raise RuntimeError(f"Lock target mismatch: expected {expected!r}, found {actual!r}")


def write_atomic(content, output_path):
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(
        mode="w",
        encoding="utf-8",
        dir=output_path.parent,
        prefix=f".{output_path.name}.",
        delete=False,
    ) as temporary:
        temporary.write(content)
        temporary_path = Path(temporary.name)
    temporary_path.replace(output_path)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, default=Path("requirements-ci.in"))
    parser.add_argument("--wheel-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, default=Path("requirements-ci.lock"))
    args = parser.parse_args(argv)

    validate_target()
    write_atomic(build_lock(args.input, args.wheel_dir), args.output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
