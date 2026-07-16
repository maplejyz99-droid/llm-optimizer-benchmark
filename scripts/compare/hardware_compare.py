#!/usr/bin/env python3
"""Normalize and compare A800 and RTX 5090 benchmark result CSVs."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import re
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd


NUMERIC_COLUMNS = [
    "sequence_length",
    "effective_batch_reported",
    "microbs_per_gpu_actual",
    "acc_steps_actual",
    "world_size",
    "steps",
    "seed",
    "avg_iter_dt",
    "median_iter_dt",
    "p95_iter_dt",
    "tokens_per_second",
    "gpu_peak_per_gpu_mib",
    "gpu_avg_utilization_pct",
    "final_train_loss",
    "final_val_loss",
]

EXACT_KEYS = [
    "phase",
    "case",
    "model",
    "sequence_length",
    "optimizer",
    "effective_batch_global",
    "microbs_per_gpu_actual",
    "acc_steps_actual",
    "world_size",
    "steps",
    "seed",
    "dtype",
    "scheduler",
    "warmup_steps",
    "final_eval_batches",
    "eval_batches",
    "checkpoint_state",
    "wandb_state",
    "optimizer_hparam_hash",
    "torch_cuda_runtime",
]

SAME_GLOBAL_KEYS = [
    "phase",
    "case",
    "model",
    "sequence_length",
    "optimizer",
    "effective_batch_global",
    "world_size",
    "steps",
    "seed",
]

DISPLAY_OPT = {
    "softeq-k2000-muon": "MuonEq/Muoneq",
}

BASE_COLUMNS = [
    "hardware",
    "run_name",
    "phase",
    "case",
    "model",
    "sequence_length",
    "optimizer",
    "optimizer_display",
    "effective_batch_reported",
    "microbs_cli_requested",
    "acc_steps_cli_requested",
    "microbs_per_gpu_actual",
    "acc_steps_actual",
    "world_size",
    "steps",
    "seed",
    "dtype",
    "scheduler",
    "warmup_steps",
    "final_eval_batches",
    "eval_batches",
    "checkpoint_state",
    "wandb_state",
    "optimizer_hparam_hash",
    "torch_cuda_runtime",
    "avg_iter_dt",
    "median_iter_dt",
    "p95_iter_dt",
    "tokens_per_second",
    "tokens_per_second_from_global",
    "gpu_peak_per_gpu_mib",
    "gpu_avg_utilization_pct",
    "final_train_loss",
    "final_val_loss",
    "returncode",
    "log_path",
    "failed",
    "effective_batch_global_from_actual",
    "effective_batch_global",
    "tokens_per_update_global",
    "gpu_total_memory_mib",
    "memory_usage_ratio",
    "batch_semantics_status",
    "run_class",
]


def empty_normalized() -> pd.DataFrame:
    return pd.DataFrame(columns=BASE_COLUMNS)


def optimizer_hparam_hash(args: dict) -> str:
    keys = [
        "opt",
        "lr",
        "muon_lr_factor",
        "weight_decay",
        "beta1",
        "beta2",
        "momentum",
        "nesterov",
        "grad_clip",
        "precondition_frequency",
        "sophia_bs",
        "sophia_rho",
        "newton_muon_precond_every",
        "newton_muon_precond_ewma",
        "newton_muon_precond_init_diag",
        "newton_muon_precond_ridge_mult",
        "newton_muon_precond_eps",
    ]
    payload = {key: args.get(key) for key in keys if key in args}
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()
    return hashlib.sha1(encoded).hexdigest()[:12]


def parse_run_name(run_name: str) -> dict[str, str]:
    meta: dict[str, str] = {}
    if "_capacity_" in run_name:
        meta["phase"] = "capacity"
    elif "_benchmark-probe_" in run_name:
        meta["phase"] = "benchmark-probe"
    elif "_ddp-smoke_" in run_name:
        meta["phase"] = "ddp-smoke"
    elif "_track3-probe_" in run_name:
        meta["phase"] = "track3-probe"
    for part in str(run_name).split("_"):
        if "-" not in part:
            continue
        key, value = part.split("-", 1)
        meta[key] = value
    return meta


def to_number(value) -> float:
    if value is None:
        return math.nan
    if isinstance(value, str) and value.strip() == "":
        return math.nan
    try:
        return float(value)
    except (TypeError, ValueError):
        return math.nan


def boolish(value) -> bool:
    if isinstance(value, bool):
        return value
    return str(value).strip().lower() in {"1", "true", "yes", "y"}


def pick_first(row: pd.Series, *names: str, meta: dict[str, str] | None = None):
    for name in names:
        if name in row and str(row[name]).strip() not in {"", "nan", "None"}:
            return row[name]
        if meta is not None and name in meta and str(meta[name]).strip():
            return meta[name]
    return ""


def infer_case(row: pd.Series, meta: dict[str, str]) -> str:
    explicit = pick_first(row, "case", meta=meta)
    if explicit:
        return str(explicit)
    model = str(pick_first(row, "model", meta=meta)).lower()
    effbs = to_number(pick_first(row, "effbs", "effective_batch_global", meta=meta))
    seq = to_number(pick_first(row, "seq", "sequence_length", meta=meta))
    phase = str(pick_first(row, "phase", meta=meta))
    if phase == "track3-probe":
        return "track3"
    if model == "124m" and seq == 512 and effbs == 32:
        return "124m-small"
    if model == "124m" and seq == 512 and effbs == 256:
        return "124m-large"
    if model == "210m" and seq == 512 and effbs == 256:
        return "210m-main"
    if model == "720m" and seq == 512 and effbs in {1984, 1960}:
        return "720m-main"
    return ""


def infer_steps(row: pd.Series, meta: dict[str, str]) -> float:
    value = pick_first(
        row,
        "steps",
        "iterations",
        "train_points",
        meta=meta,
    )
    parsed = to_number(value)
    if not math.isnan(parsed):
        return parsed
    run_name = str(row.get("run_name", ""))
    match = re.search(r"(?:steps|iterations)-(\d+)", run_name)
    return float(match.group(1)) if match else math.nan


def finalize_normalized(df: pd.DataFrame, gpu_total_memory_mib: float) -> pd.DataFrame:
    if df.empty:
        df = empty_normalized()
    for col in BASE_COLUMNS:
        if col not in df:
            df[col] = math.nan if col in NUMERIC_COLUMNS else ""
    for col in NUMERIC_COLUMNS:
        if col not in df:
            df[col] = math.nan
        df[col] = pd.to_numeric(df[col], errors="coerce")

    df["effective_batch_global_from_actual"] = (
        df["microbs_per_gpu_actual"] * df["acc_steps_actual"] * df["world_size"]
    )
    df["effective_batch_global"] = df["effective_batch_global_from_actual"]
    missing_actual = df["effective_batch_global"].isna()
    df.loc[missing_actual, "effective_batch_global"] = df.loc[
        missing_actual, "effective_batch_reported"
    ]
    df["tokens_per_update_global"] = (
        df["effective_batch_global"] * df["sequence_length"]
    )
    df["tokens_per_second_from_global"] = (
        df["tokens_per_update_global"] / df["avg_iter_dt"]
    )
    df["gpu_total_memory_mib"] = gpu_total_memory_mib
    df["memory_usage_ratio"] = df["gpu_peak_per_gpu_mib"] / gpu_total_memory_mib

    reported = df["effective_batch_reported"]
    actual = df["effective_batch_global_from_actual"]
    df["batch_semantics_status"] = "missing"
    df.loc[reported.isna() & actual.notna(), "batch_semantics_status"] = "actual_only"
    df.loc[reported.notna() & actual.isna(), "batch_semantics_status"] = "reported_only"
    df.loc[
        reported.notna() & actual.notna() & (reported == actual),
        "batch_semantics_status",
    ] = "reported_equals_actual"
    df.loc[
        reported.notna() & actual.notna() & (reported != actual),
        "batch_semantics_status",
    ] = "reported_actual_mismatch"

    df["run_class"] = "candidate"
    df.loc[df["failed"], "run_class"] = "oom_or_failed"
    df.loc[df["optimizer"].eq("newton-muon") & df["world_size"].gt(1), "run_class"] = (
        "single_only_invalid"
    )
    return df


def normalize_results(
    input_csv: Path,
    hardware: str,
    gpu_total_memory_mib: float,
) -> pd.DataFrame:
    if not input_csv.exists():
        return empty_normalized()
    raw = pd.read_csv(input_csv)
    rows: list[dict] = []
    for _, row in raw.iterrows():
        meta = parse_run_name(str(row.get("run_name", "")))
        out: dict[str, object] = {}
        out["hardware"] = hardware
        out["run_name"] = row.get("run_name", "")
        out["phase"] = pick_first(row, "phase", meta=meta)
        out["case"] = infer_case(row, meta)
        out["model"] = pick_first(row, "model", meta=meta)
        out["sequence_length"] = pick_first(row, "seq", "sequence_length", meta=meta)
        out["optimizer"] = pick_first(row, "opt", "optimizer", meta=meta)
        out["optimizer_display"] = DISPLAY_OPT.get(str(out["optimizer"]), out["optimizer"])
        out["effective_batch_reported"] = pick_first(
            row, "effbs", "effective_batch", "effective_batch_global", meta=meta
        )
        out["microbs_cli_requested"] = pick_first(
            row, "microbs_cli_requested", "batch_size_cli", "cli_microbs", meta=meta
        )
        out["acc_steps_cli_requested"] = pick_first(
            row, "acc_steps_cli_requested", "acc_cli", "cli_acc", meta=meta
        )
        out["microbs_per_gpu_actual"] = pick_first(
            row, "microbs_per_gpu_actual", "microbs", "batch_size", meta=meta
        )
        out["acc_steps_actual"] = pick_first(
            row, "acc_steps_actual", "acc", "acc_steps", meta=meta
        )
        out["world_size"] = pick_first(row, "world", "world_size", meta=meta)
        out["steps"] = infer_steps(row, meta)
        out["seed"] = pick_first(row, "seed", meta=meta)
        out["dtype"] = pick_first(row, "dtype", meta=meta)
        out["scheduler"] = pick_first(row, "scheduler", meta=meta)
        out["warmup_steps"] = pick_first(row, "warmup_steps", meta=meta)
        out["final_eval_batches"] = pick_first(row, "final_eval_batches", meta=meta)
        out["eval_batches"] = pick_first(row, "eval_batches", meta=meta)
        out["checkpoint_state"] = pick_first(row, "checkpoint_state", meta=meta)
        out["wandb_state"] = pick_first(row, "wandb_state", meta=meta)
        out["optimizer_hparam_hash"] = pick_first(row, "optimizer_hparam_hash", meta=meta)
        out["torch_cuda_runtime"] = pick_first(row, "torch_cuda_runtime", meta=meta)
        for metric in [
            "avg_iter_dt",
            "median_iter_dt",
            "p95_iter_dt",
            "tokens_per_second",
            "gpu_peak_per_gpu_mib",
            "gpu_avg_utilization_pct",
            "final_train_loss",
            "final_val_loss",
            "returncode",
            "log_path",
        ]:
            out[metric] = row.get(metric, "")
        out["failed"] = boolish(row.get("failed", False))
        rows.append(out)

    return finalize_normalized(pd.DataFrame(rows), gpu_total_memory_mib)


def normalize_summary_root(
    summary_root: Path,
    hardware: str,
    gpu_total_memory_mib: float,
) -> pd.DataFrame:
    if not summary_root.exists():
        return empty_normalized()
    rows: list[dict] = []
    for summary_path in sorted(summary_root.glob("*/summary.json")):
        run_name = summary_path.parent.name
        meta = parse_run_name(run_name)
        try:
            summary = json.loads(summary_path.read_text(encoding="utf-8"))
        except json.JSONDecodeError:
            continue
        args = summary.get("args") or {}
        completed = summary.get("completed_iterations", "")
        iterations = args.get("iterations", completed)
        opt = args.get("opt", meta.get("opt", ""))
        rows.append(
            {
                "hardware": hardware,
                "run_name": run_name,
                "phase": meta.get("phase", ""),
                "case": meta.get("case", "track3" if meta.get("phase") == "track3-probe" else ""),
                "model": meta.get("model", ""),
                "sequence_length": args.get("sequence_length", meta.get("seq", "")),
                "optimizer": opt,
                "optimizer_display": DISPLAY_OPT.get(str(opt), opt),
                "effective_batch_reported": meta.get("effbs", ""),
                "microbs_cli_requested": meta.get("microbs", ""),
                "acc_steps_cli_requested": meta.get("acc", ""),
                "microbs_per_gpu_actual": args.get("batch_size", meta.get("microbs", "")),
                "acc_steps_actual": args.get("acc_steps", meta.get("acc", "")),
                "world_size": args.get("world_size", meta.get("world", "")),
                "steps": meta.get("steps", iterations),
                "seed": args.get("seed", meta.get("seed", "")),
                "dtype": args.get("dtype", ""),
                "scheduler": args.get("scheduler", ""),
                "warmup_steps": args.get("warmup_steps", ""),
                "final_eval_batches": args.get("final_eval_batches", ""),
                "eval_batches": args.get("eval_batches", ""),
                "checkpoint_state": (
                    "off"
                    if args.get("latest_ckpt_interval", 0) == 0
                    and args.get("permanent_ckpt_interval", 0) == 0
                    else "on"
                ),
                "wandb_state": "on" if args.get("wandb") else "off",
                "optimizer_hparam_hash": optimizer_hparam_hash(args),
                "torch_cuda_runtime": "",
                "avg_iter_dt": summary.get("avg_iter_dt", ""),
                "median_iter_dt": summary.get("avg_iter_dt", ""),
                "p95_iter_dt": summary.get("avg_iter_dt", ""),
                "tokens_per_second": "",
                "gpu_peak_per_gpu_mib": "",
                "gpu_avg_utilization_pct": "",
                "final_train_loss": "",
                "final_val_loss": "",
                "returncode": 0 if completed else "",
                "log_path": str(summary_path),
                "failed": not bool(completed),
            }
        )
    return finalize_normalized(pd.DataFrame(rows), gpu_total_memory_mib)


def ensure_compare_columns(df: pd.DataFrame) -> pd.DataFrame:
    for col in set(EXACT_KEYS + SAME_GLOBAL_KEYS):
        if col not in df:
            df[col] = ""
    return df


def suffix_metrics(df: pd.DataFrame, suffix: str) -> pd.DataFrame:
    keep = set(EXACT_KEYS + SAME_GLOBAL_KEYS)
    renamed = {}
    for col in df.columns:
        if col not in keep:
            renamed[col] = f"{col}_{suffix}"
    return df.rename(columns=renamed)


def compare_pairs(
    a800: pd.DataFrame,
    gpu5090: pd.DataFrame,
    keys: list[str],
    compare_class: str,
) -> pd.DataFrame:
    a = ensure_compare_columns(a800.copy())
    b = ensure_compare_columns(gpu5090.copy())
    if "failed" not in a or "failed" not in b:
        return pd.DataFrame()
    a = a[~a["failed"]]
    b = b[~b["failed"]]
    if a.empty or b.empty:
        return pd.DataFrame()
    merged = a.merge(b, on=keys, suffixes=("_a800", "_5090"))
    if merged.empty:
        return merged
    merged["compare_class"] = compare_class
    merged["speedup_a800_over_5090"] = (
        merged["avg_iter_dt_5090"] / merged["avg_iter_dt_a800"]
    )
    merged["throughput_ratio_a800_over_5090"] = (
        merged["tokens_per_second_from_global_a800"]
        / merged["tokens_per_second_from_global_5090"]
    )
    merged["memory_ratio_a800_over_5090"] = (
        merged["gpu_peak_per_gpu_mib_a800"] / merged["gpu_peak_per_gpu_mib_5090"]
    )
    return merged


def internal_scaling(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty or "failed" not in df:
        return pd.DataFrame()
    keys = [
        "hardware",
        "case",
        "model",
        "sequence_length",
        "optimizer",
        "effective_batch_global",
        "steps",
        "seed",
    ]
    rows: list[dict] = []
    clean = df[~df["failed"]].copy()
    for _, group in clean.groupby(keys, dropna=False):
        singles = group[group["world_size"] == 1]
        ddps = group[group["world_size"] > 1]
        if singles.empty or ddps.empty:
            continue
        single = singles.sort_values("avg_iter_dt").iloc[0]
        for _, ddp in ddps.iterrows():
            world = ddp["world_size"]
            rows.append(
                {
                    "hardware": ddp["hardware"],
                    "single_phase": single["phase"],
                    "ddp_phase": ddp["phase"],
                    "case": ddp["case"],
                    "model": ddp["model"],
                    "sequence_length": ddp["sequence_length"],
                    "optimizer": ddp["optimizer"],
                    "effective_batch_global": ddp["effective_batch_global"],
                    "world_size": world,
                    "steps": ddp["steps"],
                    "single_avg_iter_dt": single["avg_iter_dt"],
                    "ddp_avg_iter_dt": ddp["avg_iter_dt"],
                    "observed_ddp_overhead_s": ddp["avg_iter_dt"]
                    - single["avg_iter_dt"] / world,
                    "observed_parallel_efficiency": single["avg_iter_dt"]
                    / (world * ddp["avg_iter_dt"]),
                    "single_tokens_per_second": single["tokens_per_second_from_global"],
                    "ddp_tokens_per_second": ddp["tokens_per_second_from_global"],
                    "single_peak_mib": single["gpu_peak_per_gpu_mib"],
                    "ddp_peak_mib_per_gpu": ddp["gpu_peak_per_gpu_mib"],
                }
            )
    return pd.DataFrame(rows)


def hardware_best(df: pd.DataFrame) -> pd.DataFrame:
    group_cols = [
        "hardware",
        "phase",
        "case",
        "model",
        "sequence_length",
        "optimizer",
        "world_size",
    ]
    if df.empty or "failed" not in df:
        return pd.DataFrame()
    clean = df[~df["failed"]].copy()
    if clean.empty:
        return pd.DataFrame()
    rows: list[dict] = []
    for key, group in clean.groupby(group_cols, dropna=False):
        max_micro = group.sort_values("microbs_per_gpu_actual", ascending=False).iloc[0]
        best_tps = group.sort_values("tokens_per_second_from_global", ascending=False).iloc[0]
        best_dt = group.sort_values("avg_iter_dt", ascending=True).iloc[0]
        base = dict(zip(group_cols, key))
        rows.append(
            {
                **base,
                "max_stable_microbs": max_micro["microbs_per_gpu_actual"],
                "max_stable_microbs_run": max_micro["run_name"],
                "best_tokens_per_second": best_tps["tokens_per_second_from_global"],
                "best_tokens_per_second_run": best_tps["run_name"],
                "best_wall_clock_iter_dt": best_dt["avg_iter_dt"],
                "best_wall_clock_run": best_dt["run_name"],
            }
        )
    return pd.DataFrame(rows)


def write_csv(df: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, index=False)


def save_placeholder(path: Path, title: str, message: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.axis("off")
    ax.text(0.5, 0.62, title, ha="center", va="center", fontsize=15)
    ax.text(0.5, 0.42, message, ha="center", va="center", fontsize=10)
    fig.tight_layout()
    fig.savefig(path, dpi=160)
    plt.close(fig)


def bar_plot(
    df: pd.DataFrame,
    path: Path,
    value_col: str,
    title: str,
    ylabel: str,
    label_col: str = "case",
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if df.empty or value_col not in df:
        save_placeholder(path, title, "No matched data available")
        return
    plot_df = df.copy()
    if label_col not in plot_df:
        plot_df[label_col] = ""
    if "optimizer" not in plot_df:
        plot_df["optimizer"] = ""
    plot_df["label"] = (
        plot_df[label_col].fillna("").astype(str)
        + "\n"
        + plot_df["optimizer"].fillna("").astype(str)
    )
    pivot = plot_df.pivot_table(
        index="label",
        columns="hardware",
        values=value_col,
        aggfunc="max",
    ).sort_index()
    if pivot.empty:
        save_placeholder(path, title, "No matched data available")
        return
    ax = pivot.plot(kind="bar", figsize=(max(8, len(pivot) * 0.45), 5))
    ax.set_title(title)
    ax.set_ylabel(ylabel)
    ax.set_xlabel("")
    ax.tick_params(axis="x", labelrotation=70)
    ax.grid(axis="y", alpha=0.25)
    ax.figure.tight_layout()
    ax.figure.savefig(path, dpi=160)
    plt.close(ax.figure)


def heatmap(df: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if df.empty or "speedup_a800_over_5090" not in df:
        save_placeholder(path, "A800 over 5090 speedup heatmap", "No exact-match data")
        return
    work = df.copy()
    work["row"] = work["case"].fillna("").astype(str) + "/" + work["model"].fillna("").astype(str)
    pivot = work.pivot_table(
        index="row",
        columns="optimizer",
        values="speedup_a800_over_5090",
        aggfunc="mean",
    )
    if pivot.empty:
        save_placeholder(path, "A800 over 5090 speedup heatmap", "No exact-match data")
        return
    fig, ax = plt.subplots(figsize=(max(7, len(pivot.columns) * 1.2), max(4, len(pivot) * 0.45)))
    image = ax.imshow(pivot.fillna(0), aspect="auto", cmap="viridis")
    ax.set_xticks(range(len(pivot.columns)))
    ax.set_xticklabels(pivot.columns, rotation=45, ha="right")
    ax.set_yticks(range(len(pivot.index)))
    ax.set_yticklabels(pivot.index)
    ax.set_title("A800 over 5090 speedup")
    for i, row in enumerate(pivot.index):
        for j, col in enumerate(pivot.columns):
            value = pivot.loc[row, col]
            if pd.notna(value):
                ax.text(j, i, f"{value:.2f}x", ha="center", va="center", color="white")
    fig.colorbar(image, ax=ax, label="speedup")
    fig.tight_layout()
    fig.savefig(path, dpi=160)
    plt.close(fig)


def line_or_bar_internal(df: pd.DataFrame, path: Path, hardware: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if df.empty or "hardware" not in df:
        save_placeholder(path, f"{hardware} internal single vs DDP efficiency", "No internal scaling pairs")
        return
    subset = df[df["hardware"] == hardware]
    title = f"{hardware} internal single vs DDP efficiency"
    if subset.empty:
        save_placeholder(path, title, "No internal scaling pairs")
        return
    work = subset.copy()
    work["label"] = (
        work["case"].fillna("").astype(str)
        + "\n"
        + work["optimizer"].fillna("").astype(str)
    )
    ax = work.plot(
        kind="bar",
        x="label",
        y="observed_parallel_efficiency",
        legend=False,
        figsize=(max(8, len(work) * 0.45), 5),
    )
    ax.axhline(1.0, color="black", linestyle="--", linewidth=1)
    ax.set_ylim(0, max(1.05, work["observed_parallel_efficiency"].max() * 1.1))
    ax.set_ylabel("parallel efficiency")
    ax.set_xlabel("")
    ax.set_title(title)
    ax.tick_params(axis="x", labelrotation=70)
    ax.grid(axis="y", alpha=0.25)
    ax.figure.tight_layout()
    ax.figure.savefig(path, dpi=160)
    plt.close(ax.figure)


def make_figures(
    normalized: pd.DataFrame,
    exact: pd.DataFrame,
    same_global: pd.DataFrame,
    scaling: pd.DataFrame,
    figure_dir: Path,
) -> None:
    figure_dir.mkdir(parents=True, exist_ok=True)
    bar_plot(
        normalized,
        figure_dir / "throughput_by_case.png",
        "tokens_per_second_from_global",
        "Throughput by case",
        "tokens/s",
    )
    bar_plot(
        normalized,
        figure_dir / "peak_memory_by_case.png",
        "gpu_peak_per_gpu_mib",
        "Peak memory by case",
        "MiB/GPU",
    )
    heatmap(exact, figure_dir / "iter_dt_speedup_heatmap.png")
    bar_plot(
        normalized,
        figure_dir / "capacity_max_batch.png",
        "microbs_per_gpu_actual",
        "Stable micro batch by case",
        "micro batch/GPU",
    )
    compare_counts = pd.DataFrame(
        [
            {"hardware": "exact", "case": "matched rows", "count": len(exact)},
            {"hardware": "same-global", "case": "matched rows", "count": len(same_global)},
        ]
    )
    bar_plot(
        compare_counts,
        figure_dir / "exact_vs_practical_comparison.png",
        "count",
        "Exact-match vs same-global-batch row counts",
        "rows",
        label_col="case",
    )
    line_or_bar_internal(scaling, figure_dir.parent / "a800" / "internal_single_vs_ddp.png", "a800")
    line_or_bar_internal(scaling, figure_dir.parent / "5090" / "internal_single_vs_ddp.png", "5090")
    bar_plot(
        scaling,
        figure_dir / "ddp_overhead_seconds.png",
        "observed_ddp_overhead_s",
        "Observed DDP overhead",
        "seconds/update",
    )
    bar_plot(
        scaling,
        figure_dir / "parallel_efficiency.png",
        "observed_parallel_efficiency",
        "Observed parallel efficiency",
        "efficiency",
    )
    save_placeholder(
        figure_dir / "nccl_time_fraction.png",
        "NCCL time fraction",
        "Run profiler probes to populate profiler_nccl_summary.csv",
    )
    save_placeholder(
        figure_dir / "allreduce_bandwidth.png",
        "All-reduce bandwidth",
        "Run all-reduce microbenchmark to populate allreduce_microbenchmark.csv",
    )
    save_placeholder(
        figure_dir / "topology_summary.png",
        "Topology summary",
        "Add nvidia-smi topo -m captures for both machines",
    )


def markdown_table(df: pd.DataFrame, columns: list[str], limit: int = 20) -> str:
    if df.empty:
        return "_No rows._"
    cols = [col for col in columns if col in df.columns]
    if not cols:
        return "_No columns._"
    rows = df[cols].head(limit).fillna("").astype(str).values.tolist()
    widths = [
        max(len(str(col)), *(len(row[idx]) for row in rows))
        for idx, col in enumerate(cols)
    ]

    def fmt(values: list[str]) -> str:
        return "| " + " | ".join(
            str(value).ljust(widths[idx]) for idx, value in enumerate(values)
        ) + " |"

    header = fmt([str(col) for col in cols])
    sep = "| " + " | ".join("-" * width for width in widths) + " |"
    body = "\n".join(fmt(row) for row in rows)
    return "\n".join([header, sep, body])


def write_reports(
    docs_dir: Path,
    normalized: pd.DataFrame,
    exact: pd.DataFrame,
    same_global: pd.DataFrame,
    best: pd.DataFrame,
    scaling: pd.DataFrame,
    issues: pd.DataFrame,
) -> None:
    docs_dir.mkdir(parents=True, exist_ok=True)
    exact_count = len(exact)
    same_count = len(same_global)
    issue_count = len(issues)
    summary = f"""# A800 vs RTX 5090 Hardware Compare Report

Generated by `scripts/compare/hardware_compare.py`.

## Summary

- Normalized rows: {len(normalized)}
- Exact-match rows: {exact_count}
- Same-global-batch practical rows: {same_count}
- OOM/failed/mismatched rows needing review: {issue_count}

## Interpretation Rules

- Exact-match rows support the cleanest hardware comparison.
- Same-global-batch rows answer the practical question: same training target, different runnable micro batch and accumulation.
- Observed DDP overhead is not pure communication time; use profiler/NCCL rows when available.

## Exact-Match Preview

{markdown_table(exact, ["phase", "case", "model", "sequence_length", "optimizer", "effective_batch_global", "world_size", "speedup_a800_over_5090", "throughput_ratio_a800_over_5090"])}

## Same-Global-Batch Preview

{markdown_table(same_global, ["phase", "case", "model", "sequence_length", "optimizer", "effective_batch_global", "world_size", "microbs_per_gpu_actual_a800", "acc_steps_actual_a800", "microbs_per_gpu_actual_5090", "acc_steps_actual_5090", "speedup_a800_over_5090"])}
"""
    (docs_dir / "hardware-compare-a800-vs-5090-report.md").write_text(
        summary, encoding="utf-8"
    )

    for hardware in ["a800", "5090"]:
        subset = normalized[normalized["hardware"] == hardware]
        internal = (
            scaling[scaling["hardware"] == hardware]
            if "hardware" in scaling
            else pd.DataFrame()
        )
        text = f"""# {hardware.upper()} Internal Comparison Report

## Summary

- Rows: {len(subset)}
- Failed/OOM rows: {int(subset["failed"].sum()) if not subset.empty else 0}
- Internal single-vs-DDP pairs: {len(internal)}

## Internal Scaling Preview

{markdown_table(internal, ["phase", "case", "model", "sequence_length", "optimizer", "effective_batch_global", "world_size", "single_avg_iter_dt", "ddp_avg_iter_dt", "observed_parallel_efficiency"])}
"""
        (docs_dir / f"{hardware}-internal-comparison-report.md").write_text(
            text, encoding="utf-8"
        )

    communication = f"""# Hardware Communication Analysis Report

## Observed DDP Overhead

These rows are estimated from single-card and DDP step time. They include communication, synchronization, launch overhead, load imbalance, memory pressure, and accumulation effects.

{markdown_table(scaling, ["hardware", "phase", "case", "model", "optimizer", "world_size", "observed_ddp_overhead_s", "observed_parallel_efficiency"])}

## Evidence Still Needed

- Profiler rows: `nccl_time_s`, `nccl_time_pct`, `all_reduce_time_s`.
- All-reduce microbenchmark rows for matched tensor sizes.
- A800 `nvidia-smi topo -m`; 5090 topology should be recorded as PCIe Gen5 x16, NODE, non-NVLink when confirmed from the target host.
"""
    (docs_dir / "hardware-communication-analysis-report.md").write_text(
        communication, encoding="utf-8"
    )

    methodology = """# Hardware Compare Methodology

## Batch Semantics

Every row must distinguish requested CLI batch fields from actual per-GPU fields. The global effective batch used for comparison is:

```text
effective_batch_global = microbs_per_gpu_actual * acc_steps_actual * world_size
tokens_per_update_global = effective_batch_global * sequence_length
```

Rows where reported `effbs` disagrees with the formula are retained but flagged for review.

## Comparison Classes

- `exact_success`: exact same model, sequence length, optimizer, effective batch, per-GPU micro batch, accumulation, world size, steps, seed, dtype, scheduler, warmup, eval/checkpoint/W&B state, optimizer hyperparams, and runtime.
- `same_effbs_fallback`: same global batch and training target, but different micro batch or accumulation because one hardware cannot run the other hardware's exact config.
- `hardware_best`: best observed stable config for max micro batch, best tokens/s, or best wall-clock for the target effective batch.

Short probes are system-performance evidence only. They do not prove final convergence quality.
"""
    (docs_dir / "hardware-compare-methodology.md").write_text(
        methodology, encoding="utf-8"
    )

    main_benchmark = f"""# Hardware Main Benchmark Comparison Report

## Exact-Match Rows

{markdown_table(exact, ["case", "model", "optimizer", "effective_batch_global", "world_size", "avg_iter_dt_a800", "avg_iter_dt_5090", "speedup_a800_over_5090"])}

## Same-Global-Batch Practical Rows

{markdown_table(same_global, ["case", "model", "optimizer", "effective_batch_global", "world_size", "avg_iter_dt_a800", "avg_iter_dt_5090", "microbs_per_gpu_actual_a800", "acc_steps_actual_a800", "microbs_per_gpu_actual_5090", "acc_steps_actual_5090"])}
"""
    (docs_dir / "hardware-compare-main-benchmark-report.md").write_text(
        main_benchmark, encoding="utf-8"
    )

    capacity = f"""# Hardware Capacity Comparison Report

## Hardware-Best Preview

{markdown_table(best, ["hardware", "phase", "case", "model", "sequence_length", "optimizer", "world_size", "max_stable_microbs", "best_tokens_per_second", "best_wall_clock_iter_dt"])}

## Rows Needing Review

{markdown_table(issues, ["hardware", "phase", "case", "model", "sequence_length", "optimizer", "batch_semantics_status", "failed", "run_name"])}
"""
    (docs_dir / "hardware-compare-capacity-report.md").write_text(
        capacity, encoding="utf-8"
    )

    track3 = normalized[normalized["phase"].eq("track3-probe") | normalized["case"].eq("track3")]
    track3_report = f"""# Hardware Track3 Comparison Report

## Track3 Rows

{markdown_table(track3, ["hardware", "model", "sequence_length", "optimizer", "effective_batch_global", "microbs_per_gpu_actual", "acc_steps_actual", "world_size", "avg_iter_dt", "tokens_per_second_from_global", "gpu_peak_per_gpu_mib", "failed"])}
"""
    (docs_dir / "hardware-compare-track3-report.md").write_text(
        track3_report, encoding="utf-8"
    )


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--a800-results")
    parser.add_argument("--gpu5090-results")
    parser.add_argument("--a800-summary-root")
    parser.add_argument("--gpu5090-summary-root")
    parser.add_argument("--out-dir", default="logs/compare")
    parser.add_argument("--figure-dir", default="figures/hardware")
    parser.add_argument("--docs-dir", default="docs")
    parser.add_argument("--a800-memory-mib", type=float, default=81920.0)
    parser.add_argument("--gpu5090-memory-mib", type=float, default=32607.0)
    args = parser.parse_args()

    out_dir = Path(args.out_dir)
    figure_dir = Path(args.figure_dir)
    docs_dir = Path(args.docs_dir)

    if args.a800_results:
        a800 = normalize_results(
            Path(args.a800_results), "a800", args.a800_memory_mib
        )
    else:
        a800 = normalize_summary_root(
            Path(args.a800_summary_root or ""), "a800", args.a800_memory_mib
        )
    if args.gpu5090_results:
        gpu5090 = normalize_results(
            Path(args.gpu5090_results), "5090", args.gpu5090_memory_mib
        )
    else:
        gpu5090 = normalize_summary_root(
            Path(args.gpu5090_summary_root or ""), "5090", args.gpu5090_memory_mib
        )
    normalized = pd.concat([a800, gpu5090], ignore_index=True)

    write_csv(a800, out_dir / "a800_results_normalized.csv")
    write_csv(gpu5090, out_dir / "5090_results_normalized.csv")
    write_csv(normalized, out_dir / "hardware_compare_merged.csv")

    exact = compare_pairs(a800, gpu5090, EXACT_KEYS, "exact_success")
    same_global = compare_pairs(
        a800, gpu5090, SAME_GLOBAL_KEYS, "same_effbs_fallback"
    )
    best = hardware_best(normalized)
    scaling = internal_scaling(normalized)
    issues = normalized[
        normalized["failed"]
        | normalized["batch_semantics_status"].eq("reported_actual_mismatch")
        | normalized["run_class"].eq("single_only_invalid")
    ].copy()

    write_csv(exact, out_dir / "exact_match_compare.csv")
    write_csv(same_global, out_dir / "same_global_batch_compare.csv")
    write_csv(best, out_dir / "hardware_best_compare.csv")
    write_csv(scaling, out_dir / "ddp_communication_overhead.csv")
    write_csv(issues, out_dir / "unmatched_or_oom.csv")
    write_csv(pd.DataFrame(), out_dir / "profiler_nccl_summary.csv")

    for hardware in ["a800", "5090"]:
        write_csv(
            normalized[normalized["hardware"] == hardware],
            out_dir / f"{hardware}_internal_summary.csv",
        )

    make_figures(normalized, exact, same_global, scaling, figure_dir)
    write_reports(docs_dir, normalized, exact, same_global, best, scaling, issues)

    print(f"normalized rows: {len(normalized)}")
    print(f"exact-match rows: {len(exact)}")
    print(f"same-global-batch rows: {len(same_global)}")
    print(f"rows needing review: {len(issues)}")
    print(f"wrote outputs under {out_dir}, {figure_dir}, and {docs_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
