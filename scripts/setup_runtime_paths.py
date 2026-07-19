#!/usr/bin/env python3
"""Create machine-local run/result roots and repository-visible symlinks."""

from __future__ import annotations

import argparse
import json
import os
import sys
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class LinkPlan:
    name: str
    link: Path
    target: Path
    create_target: bool
    create_link: bool


def _absolute(path: Path) -> Path:
    return Path(os.path.abspath(os.fspath(path.expanduser())))


def _same_link_target(link: Path, target: Path) -> bool:
    raw_target = Path(os.readlink(link))
    if not raw_target.is_absolute():
        raw_target = link.parent / raw_target
    return raw_target.resolve(strict=False) == target.resolve(strict=False)


def _is_within(path: Path, parent: Path) -> bool:
    return path == parent or path.is_relative_to(parent)


def _create_directory_tree(path: Path) -> list[Path]:
    missing = []
    current = path
    while not os.path.lexists(current):
        missing.append(current)
        parent = current.parent
        if parent == current:
            break
        current = parent
    created = []
    try:
        for directory in reversed(missing):
            try:
                directory.mkdir()
            except FileExistsError:
                if not directory.is_dir():
                    raise
            else:
                created.append(directory)
    except Exception:
        for directory in reversed(created):
            try:
                directory.rmdir()
            except OSError:
                pass
        raise
    return created


def build_plan(repo_root: Path, runs_root: Path, results_root: Path) -> list[LinkPlan]:
    repo_root = _absolute(repo_root)
    roots = {
        "runs": _absolute(runs_root),
        "results": _absolute(results_root),
    }
    if not repo_root.is_dir():
        raise ValueError(f"Repository root is not a directory: {repo_root}")

    resolved_repo = repo_root.resolve()
    resolved_roots = {
        name: target.resolve(strict=False) for name, target in roots.items()
    }
    for name, target in resolved_roots.items():
        if _is_within(target, resolved_repo) or _is_within(resolved_repo, target):
            raise ValueError(
                f"{name} target must be outside and non-ancestral to the repository: "
                f"{roots[name]}"
            )
    runs_resolved = resolved_roots["runs"]
    results_resolved = resolved_roots["results"]
    if _is_within(runs_resolved, results_resolved) or _is_within(
        results_resolved, runs_resolved
    ):
        raise ValueError("runs and results targets must be separate, non-nested roots")

    plans = []
    errors = []
    for name, target in roots.items():
        link = repo_root / name
        target_exists = os.path.lexists(target)
        if target_exists and not target.is_dir():
            errors.append(f"Runtime target is not a directory: {target}")

        link_exists = os.path.lexists(link)
        create_link = True
        if link_exists:
            if link.is_symlink() and _same_link_target(link, target):
                create_link = False
            elif link.is_symlink():
                errors.append(
                    f"Repository link points elsewhere: {link} -> {os.readlink(link)}"
                )
            else:
                errors.append(f"Repository path already exists and is not a symlink: {link}")

        plans.append(
            LinkPlan(
                name=name,
                link=link,
                target=target,
                create_target=not target_exists,
                create_link=create_link,
            )
        )

    if errors:
        raise ValueError("\n".join(errors))
    return plans


def apply_plan(plans: list[LinkPlan], *, dry_run: bool = False) -> list[dict[str, object]]:
    actions = [
        {
            "name": plan.name,
            "link": str(plan.link),
            "target": str(plan.target),
            "create_target": plan.create_target,
            "create_link": plan.create_link,
        }
        for plan in plans
    ]
    if dry_run:
        return actions

    created_target_chains: list[list[Path]] = []
    created_links: list[Path] = []
    try:
        for plan in plans:
            if plan.create_target:
                created_target_chains.append(_create_directory_tree(plan.target))
        for plan in plans:
            if plan.create_link:
                plan.link.symlink_to(plan.target, target_is_directory=True)
                created_links.append(plan.link)
    except Exception:
        for link in reversed(created_links):
            try:
                link.unlink()
            except OSError:
                pass
        for chain in reversed(created_target_chains):
            for target in reversed(chain):
                try:
                    target.rmdir()
                except OSError:
                    pass
        raise
    return actions


def setup_runtime_paths(
    repo_root: Path,
    runs_root: Path,
    results_root: Path,
    *,
    dry_run: bool = False,
) -> list[dict[str, object]]:
    return apply_plan(
        build_plan(repo_root, runs_root, results_root),
        dry_run=dry_run,
    )


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--repo-root",
        type=Path,
        default=Path(__file__).resolve().parents[1],
    )
    parser.add_argument("--runs-root", type=Path, required=True)
    parser.add_argument("--results-root", type=Path, required=True)
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    try:
        actions = setup_runtime_paths(
            args.repo_root,
            args.runs_root,
            args.results_root,
            dry_run=args.dry_run,
        )
    except (OSError, ValueError) as exc:
        print(f"runtime path setup failed: {exc}", file=sys.stderr)
        return 2
    print(json.dumps({"dry_run": args.dry_run, "paths": actions}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
