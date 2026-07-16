#!/usr/bin/env python3
"""Fail unless the active environment is the locked CPU test runtime."""

import importlib.metadata
import json
import platform
import re
import sys

import numpy
import tiktoken
import torch
import yaml


EXPECTED = {
    "numpy": "2.2.6",
    "pyyaml": "6.0.3",
    "tiktoken": "0.12.0",
    "torch": "2.9.1+cpu",
}
EXPECTED_PLATFORM = {
    "implementation": "CPython",
    "python": "3.10.19",
    "system": "Linux",
    "machine": "x86_64",
}


def normalize_package_name(name):
    return re.sub(r"[-_.]+", "-", name).lower()


def installed_versions():
    versions = {}
    for distribution in importlib.metadata.distributions():
        name = distribution.metadata.get("Name")
        if not name:
            continue
        versions.setdefault(normalize_package_name(name), set()).add(
            distribution.version
        )
    return versions


def main():
    platform_actual = {
        "implementation": platform.python_implementation(),
        "python": platform.python_version(),
        "system": platform.system(),
        "machine": platform.machine(),
    }
    if platform_actual != EXPECTED_PLATFORM:
        raise RuntimeError(f"Unexpected CPU platform: {platform_actual!r}")

    versions = installed_versions()
    metadata_actual = {
        name: sorted(versions.get(name, set())) for name in EXPECTED
    }
    metadata_expected = {name: [version] for name, version in EXPECTED.items()}
    if metadata_actual != metadata_expected:
        raise RuntimeError(f"Unexpected CPU dependency metadata: {metadata_actual!r}")

    runtime_actual = {
        "numpy": str(numpy.__version__),
        "pyyaml": str(yaml.__version__),
        "tiktoken": str(tiktoken.__version__),
        "torch": str(torch.__version__),
    }
    if runtime_actual != EXPECTED:
        raise RuntimeError(f"Unexpected CPU runtime versions: {runtime_actual!r}")
    if torch.version.cuda is not None:
        raise RuntimeError(f"Expected CPU-only torch, found CUDA {torch.version.cuda}")

    forbidden = sorted(
        name for name in versions if name == "triton" or name.startswith("nvidia-")
    )
    if forbidden:
        raise RuntimeError(f"CUDA packages leaked into CPU environment: {forbidden!r}")

    print(
        json.dumps(
            {
                "packages": runtime_actual,
                "platform": platform_actual,
                "torch_cuda": torch.version.cuda,
            },
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
