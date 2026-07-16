#!/usr/bin/env python3
"""Run the remote analogue of microsoft/mup README's LR sweep figure.

The script is intentionally an external experiment launcher: it does not import
or mutate the training stack, it only calls ``torchrun ./src/main.py``.
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import signal
import shlex
import subprocess
import sys
import time
from dataclasses import dataclass
from pathlib import Path


SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parents[2]


@dataclass(frozen=True)
class Job:
    phase: str
    alias: str
    model: str
    n_embd: int
    n_head: int
    lr: float
    iterations: int
    warmup_steps: int
    seed: int
    experiment_name: str
    log_path: Path
    meta_path: Path
    gpu_path: Path
    extra_args: dict[str, object]


def load_manifest(path: Path) -> dict:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def lr_label(lr: float) -> str:
    # Keep labels collision-free for close LR values such as 0.0015 and 0.002.
    return f"{lr:.8g}".replace(".", "p").replace("-", "m")


def width_by_embd(manifest: dict, n_embd: int) -> dict:
    for width in manifest["widths"]:
        if int(width["n_embd"]) == n_embd:
            return width
    raise ValueError(f"n_embd={n_embd} is not present in manifest widths")


def model_by_alias(manifest: dict, alias: str) -> dict:
    for model in manifest["models"]:
        if model["alias"] == alias:
            return model
    raise ValueError(f"model alias {alias!r} is not present in manifest models")


def make_job(
    manifest: dict,
    *,
    phase: str,
    model_entry: dict,
    width: dict,
    lr: float,
    iterations: int,
    warmup_steps: int,
    extra_args: dict[str, object] | None = None,
) -> Job:
    seed = int(manifest["seed"])
    exp = (
        f"{manifest['experiment_prefix']}_{phase}_{model_entry['alias']}"
        f"_w{width['n_embd']}_lr{lr_label(lr)}_s{seed}_steps{iterations}"
    )
    log_root = REPO_ROOT / manifest["log_root"] / phase
    log_path = log_root / f"{exp}.log"
    return Job(
        phase=phase,
        alias=model_entry["alias"],
        model=model_entry["model"],
        n_embd=int(width["n_embd"]),
        n_head=int(width["n_head"]),
        lr=float(lr),
        iterations=int(iterations),
        warmup_steps=int(warmup_steps),
        seed=seed,
        experiment_name=exp,
        log_path=log_path,
        meta_path=log_path.with_suffix(".meta.json"),
        gpu_path=log_path.with_suffix(".gpu.csv"),
        extra_args=extra_args or {},
    )


def build_jobs(manifest: dict, args: argparse.Namespace) -> list[Job]:
    phase = args.phase
    jobs: list[Job] = []
    if phase == "calibrate":
        spec = manifest["calibrate"]
        model_entry = model_by_alias(manifest, spec["model_alias"])
        width = width_by_embd(manifest, int(spec["n_embd"]))
        jobs.append(
            make_job(
                manifest,
                phase=phase,
                model_entry=model_entry,
                width=width,
                lr=float(spec["lr"]),
                iterations=int(spec["iterations"]),
                warmup_steps=int(spec["warmup_steps"]),
            )
        )
        return jobs

    if phase in {"probe_v2", "probe_v2_b32"}:
        spec = manifest["v2"]["probe"]
        extra_args = dict(manifest["v2"]["base_args"])
        if phase == "probe_v2_b32":
            extra_args["batch_size"] = 32
        for model_entry in manifest["models"]:
            for width in manifest["v2"]["probe_widths"]:
                jobs.append(
                    make_job(
                        manifest,
                        phase=phase,
                        model_entry=model_entry,
                        width=width,
                        lr=float(spec["lr"]),
                        iterations=int(spec["iterations"]),
                        warmup_steps=int(spec["warmup_steps"]),
                        extra_args=extra_args,
                    )
                )
        return jobs

    if phase == "probe_v2_l4":
        spec = manifest["v2_l4"]["probe"]
        extra_args = dict(manifest["v2_l4"]["base_args"])
        for model_entry in manifest["models"]:
            for width in manifest["v2_l4"]["probe_widths"]:
                jobs.append(
                    make_job(
                        manifest,
                        phase=phase,
                        model_entry=model_entry,
                        width=width,
                        lr=float(spec["lr"]),
                        iterations=int(spec["iterations"]),
                        warmup_steps=int(spec["warmup_steps"]),
                        extra_args=extra_args,
                    )
                )
        return jobs

    if phase in {
        "sweep_v2_l4",
        "long_v2_l4_global",
        "long_v2_l4_anchor128",
        "long_v2_l4_anchor768",
        "long_v2_l4_extra",
    }:
        if not args.widths:
            raise SystemExit(f"--phase {phase} requires --widths, for example --widths 128,256,512")
        is_long = phase.startswith("long_v2_l4")
        extra_args = dict(manifest["v2_l4"]["base_args"])
        width_map = {int(width["n_embd"]): width for width in manifest["v2_l4"]["probe_widths"]}
        selected_widths = []
        for value in args.widths.split(","):
            n_embd = int(value)
            if n_embd not in width_map:
                raise SystemExit(f"unknown v2_l4 width {n_embd}; run probe_v2_l4 or update manifest")
            selected_widths.append(width_map[n_embd])
        spec = manifest["v2_l4"]["long"] if is_long else manifest["v2_l4"]["sweep"]
        if not is_long:
            lr_by_alias = None
            lrs = [float(lr) for lr in spec["lrs"]]
        else:
            if args.sp_lr is None or args.mup_lr is None:
                raise SystemExit(f"--phase {phase} requires --sp-lr and --mup-lr")
            lr_by_alias = {"sp": float(args.sp_lr), "mup": float(args.mup_lr)}
            lrs = []
        for model_entry in manifest["models"]:
            phase_lrs = [lr_by_alias[model_entry["alias"]]] if lr_by_alias else lrs
            for width in selected_widths:
                for lr in phase_lrs:
                    jobs.append(
                        make_job(
                            manifest,
                            phase=phase,
                            model_entry=model_entry,
                            width=width,
                            lr=lr,
                            iterations=int(spec["iterations"]),
                            warmup_steps=int(spec["warmup_steps"]),
                            extra_args=extra_args,
                        )
                    )
        return jobs

    if phase in {"sweep_v2", "long_v2", "sweep_v2_b32", "long_v2_b32"}:
        if not args.widths:
            raise SystemExit(f"--phase {phase} requires --widths, for example --widths 768,1536,2048")
        is_long = phase in {"long_v2", "long_v2_b32"}
        extra_args = dict(manifest["v2"]["base_args"])
        if phase.endswith("_b32"):
            extra_args["batch_size"] = 32
        width_map = {int(width["n_embd"]): width for width in manifest["v2"]["probe_widths"]}
        selected_widths = []
        for value in args.widths.split(","):
            n_embd = int(value)
            if n_embd not in width_map:
                raise SystemExit(f"unknown v2 width {n_embd}; run probe_v2 or update manifest")
            selected_widths.append(width_map[n_embd])
        spec = manifest["v2"]["long"] if is_long else manifest["v2"]["sweep"]
        if not is_long:
            lr_by_alias = None
            lrs = [float(lr) for lr in spec["lrs"]]
        else:
            if args.sp_lr is None or args.mup_lr is None:
                raise SystemExit("--phase long_v2 requires --sp-lr and --mup-lr")
            lr_by_alias = {"sp": float(args.sp_lr), "mup": float(args.mup_lr)}
            lrs = []
        for model_entry in manifest["models"]:
            phase_lrs = [lr_by_alias[model_entry["alias"]]] if lr_by_alias else lrs
            for width in selected_widths:
                for lr in phase_lrs:
                    jobs.append(
                        make_job(
                            manifest,
                            phase=phase,
                            model_entry=model_entry,
                            width=width,
                            lr=lr,
                            iterations=int(spec["iterations"]),
                            warmup_steps=int(spec["warmup_steps"]),
                            extra_args=extra_args,
                        )
                    )
        return jobs

    if phase == "coarse":
        spec = manifest["coarse"]
        lrs = [float(lr) for lr in spec["lrs"]]
    elif phase == "fine":
        if not args.lrs:
            raise SystemExit("--phase fine requires --lrs, for example --lrs 5e-4,2e-3")
        spec = manifest["fine"]
        lrs = [float(lr) for lr in args.lrs.split(",")]
    elif phase == "long":
        spec = manifest["long"]
        if args.sp_lr is None or args.mup_lr is None:
            raise SystemExit("--phase long requires --sp-lr and --mup-lr")
        lrs_by_alias = {"sp": float(args.sp_lr), "mup": float(args.mup_lr)}
        for model_entry in manifest["models"]:
            for width in manifest["widths"]:
                jobs.append(
                    make_job(
                        manifest,
                        phase=phase,
                        model_entry=model_entry,
                        width=width,
                        lr=lrs_by_alias[model_entry["alias"]],
                        iterations=int(spec["iterations"]),
                        warmup_steps=int(spec["warmup_steps"]),
                    )
                )
        return jobs
    else:
        raise ValueError(f"unsupported phase: {phase}")

    for model_entry in manifest["models"]:
        for width in manifest["widths"]:
            for lr in lrs:
                jobs.append(
                    make_job(
                        manifest,
                        phase=phase,
                        model_entry=model_entry,
                        width=width,
                        lr=lr,
                        iterations=int(spec["iterations"]),
                        warmup_steps=int(spec["warmup_steps"]),
                    )
                )
    return jobs


def cli_args(manifest: dict, job: Job) -> list[str]:
    base = {**manifest["base_args"], **job.extra_args}
    values: dict[str, object] = {
        **base,
        "model": job.model,
        "n_embd": job.n_embd,
        "n_head": job.n_head,
        "lr": job.lr,
        "iterations": job.iterations,
        "warmup_steps": job.warmup_steps,
        "seed": job.seed,
        "experiment_name": job.experiment_name,
        "results_base_folder": manifest["results_base_folder"],
    }

    cmd = ["torchrun", "--standalone", "--nproc_per_node=1", "./src/main.py"]
    for key, value in values.items():
        cmd.append(f"--{key}")
        cmd.append(str(value))
    return cmd


def write_meta(job: Job, command: list[str]) -> None:
    job.log_path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "phase": job.phase,
        "alias": job.alias,
        "model": job.model,
        "n_embd": job.n_embd,
        "n_head": job.n_head,
        "lr": job.lr,
        "iterations": job.iterations,
        "warmup_steps": job.warmup_steps,
        "seed": job.seed,
        "experiment_name": job.experiment_name,
        "log_path": str(job.log_path.relative_to(REPO_ROOT)),
        "gpu_path": str(job.gpu_path.relative_to(REPO_ROOT)),
        "command": command,
        "extra_args": job.extra_args,
    }
    with job.meta_path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, sort_keys=True)
        handle.write("\n")


def start_gpu_monitor(job: Job) -> tuple[subprocess.Popen, object]:
    job.gpu_path.parent.mkdir(parents=True, exist_ok=True)
    handle = job.gpu_path.open("w", encoding="utf-8")
    handle.write("timestamp,index,memory_used_mib,utilization_gpu_pct\n")
    handle.flush()
    command = [
        "nvidia-smi",
        "--query-gpu=timestamp,index,memory.used,utilization.gpu",
        "--format=csv,noheader,nounits",
        "-lms",
        "500",
    ]
    process = subprocess.Popen(
        command,
        cwd=REPO_ROOT,
        stdout=handle,
        stderr=subprocess.DEVNULL,
        text=True,
    )
    return process, handle


def stop_gpu_monitor(process: subprocess.Popen, handle: object) -> None:
    if process.poll() is None:
        process.send_signal(signal.SIGINT)
        try:
            process.wait(timeout=3)
        except subprocess.TimeoutExpired:
            process.kill()
            process.wait(timeout=3)
    handle.close()


def run_job(manifest: dict, job: Job, dry_run: bool, skip_existing: bool) -> int:
    command = cli_args(manifest, job)
    write_meta(job, command)
    quoted = " ".join(shlex.quote(part) for part in command)
    print(f"\n[{job.phase}] {job.experiment_name}")
    print(quoted)
    print(f"log: {job.log_path.relative_to(REPO_ROOT)}")

    if dry_run:
        return 0
    if skip_existing and job.log_path.exists():
        text = job.log_path.read_text(encoding="utf-8", errors="ignore")
        if "# returncode: 0" in text:
            print("skip: existing completed log")
            return 0

    started = time.time()
    env = os.environ.copy()
    with job.log_path.open("w", encoding="utf-8") as log:
        log.write(f"# command: {quoted}\n")
        log.flush()
        monitor, monitor_handle = start_gpu_monitor(job)
        try:
            process = subprocess.run(
                command,
                cwd=REPO_ROOT,
                stdout=log,
                stderr=subprocess.STDOUT,
                env=env,
                check=False,
            )
        finally:
            stop_gpu_monitor(monitor, monitor_handle)
        elapsed = time.time() - started
        log.write(f"\n# returncode: {process.returncode}\n")
        log.write(f"# launcher_elapsed_seconds: {elapsed:.3f}\n")
    print(f"returncode={process.returncode} elapsed={elapsed:.1f}s")
    return int(process.returncode)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", default=str(SCRIPT_DIR / "manifest.json"))
    parser.add_argument(
        "--phase",
        choices=[
            "calibrate",
            "coarse",
            "fine",
            "long",
            "probe_v2",
            "sweep_v2",
            "long_v2",
            "probe_v2_b32",
            "sweep_v2_b32",
            "long_v2_b32",
            "probe_v2_l4",
            "sweep_v2_l4",
            "long_v2_l4_global",
            "long_v2_l4_anchor128",
            "long_v2_l4_anchor768",
            "long_v2_l4_extra",
        ],
        required=True,
    )
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--skip-existing", action="store_true")
    parser.add_argument("--max-jobs", type=int, default=None)
    parser.add_argument("--lrs", default=None, help="Comma-separated LR list for --phase fine")
    parser.add_argument("--sp-lr", type=float, default=None, help="Selected SP LR for --phase long")
    parser.add_argument("--mup-lr", type=float, default=None, help="Selected muP LR for --phase long")
    parser.add_argument("--widths", default=None, help="Comma-separated n_embd values for v2 phases")
    args = parser.parse_args(argv)

    manifest = load_manifest(Path(args.manifest))
    jobs = build_jobs(manifest, args)
    if args.max_jobs is not None:
        jobs = jobs[: args.max_jobs]

    manifest_path = REPO_ROOT / manifest["log_root"] / f"{args.phase}_jobs.jsonl"
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    with manifest_path.open("w", encoding="utf-8") as handle:
        for job in jobs:
            handle.write(json.dumps(job.__dict__ | {
                "log_path": str(job.log_path.relative_to(REPO_ROOT)),
                "meta_path": str(job.meta_path.relative_to(REPO_ROOT)),
                "gpu_path": str(job.gpu_path.relative_to(REPO_ROOT)),
            }, sort_keys=True, default=str))
            handle.write("\n")
    print(f"jobs={len(jobs)} manifest={manifest_path.relative_to(REPO_ROOT)}")

    failures = 0
    for job in jobs:
        failures += int(run_job(manifest, job, args.dry_run, args.skip_existing) != 0)
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
