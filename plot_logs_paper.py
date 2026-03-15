#!/usr/bin/env python3
from pathlib import Path
import re

import matplotlib.pyplot as plt

RUNS = [
    {
        "path": "/root/work/llm-optimizer-benchmark/logs/adamw_124m_15B.log",
        "label": "AdamW",
        "color": "#1f77b4",
        "linestyle": "-",
    },
    {
        "path": "/root/work/llm-optimizer-benchmark/logs/adopt_124m_1p5B.log",
        "label": "ADOPT",
        "color": "#ff7f0e",
        "linestyle": "-",
    },
    {
        "path": "/root/work/llm-optimizer-benchmark/logs/lion_124m_1p5B.log",
        "label": "Lion",
        "color": "#2ca02c",
        "linestyle": "-",
    },
    {
        "path": "/root/work/llm-optimizer-benchmark/logs/marsadamw_124m_1p5B_gpu1.log",
        "label": "MARS-AdamW",
        "color": "#d62728",
        "linestyle": "-",
    },
    {
        "path": "/root/work/llm-optimizer-benchmark/logs/marslion_124m_1p5B_gpu2.log",
        "label": "MARS-Lion",
        "color": "#9467bd",
        "linestyle": "-",
    },
    {
        "path": "/root/work/llm-optimizer-benchmark/logs/marsshampoo_124m_1p5B_gpu3.log",
        "label": "MARS-Shampoo",
        "color": "#8c564b",
        "linestyle": "-",
    },
    {
        "path": "/root/work/llm-optimizer-benchmark/logs/sfadamw_124m_1p5B_gpu3.log",
        "label": "SF-AdamW",
        "color": "#e377c2",
        "linestyle": "-",
    },
    {
        "path": "/root/work/llm-optimizer-benchmark/logs/signum_124m_1p5B.log",
        "label": "Signum",
        "color": "#7f7f7f",
        "linestyle": "-",
    },
    {
        "path": "/root/work/llm-optimizer-benchmark/logs/soap_124m_1p5B_gpu0.log",
        "label": "SOAP",
        "color": "#bcbd22",
        "linestyle": "-",
    },
    {
        "path": "/root/work/llm-optimizer-benchmark/logs/sophiag_124m_1p5B_b16a2_gpu1.log",
        "label": "Sophia-G",
        "color": "#17becf",
        "linestyle": "-",
    },
    {
        "path": "/root/work/llm-optimizer-benchmark/logs/cadamw_124m_1p5B_gpu0.log",
        "label": "C-AdamW",
        "color": "#4c78a8",
        "linestyle": "-",
    },
    {
        "path": "/root/work/llm-optimizer-benchmark/logs/adamw_magma_124m_1p5B_gpu1.log",
        "label": "AdamW+Magma",
        "color": "#1f77b4",
        "linestyle": "--",
    },
]

BATCH = 32
SEQ = 512
ACC = 1
WORLD = 1
TOKENS_PER_ITER = BATCH * SEQ * ACC * WORLD

MIN_TOKENS_B = 0.15
EVAL_RE = re.compile(r">Eval: Iter=(\d+).*?val_loss=([0-9.]+)")
TITLE = "124M SlimPajama 1.5B - Optimizer Curves (>=0.15B)"
OUT = "/root/work/llm-optimizer-benchmark/optimizer_curves_1p5B_ranked_paper.png"


def extract_eval_points(log_path):
    eval_points = []
    for line in Path(log_path).read_text(errors="ignore").splitlines():
        match = EVAL_RE.search(line)
        if not match:
            continue
        iteration = int(match.group(1))
        loss = float(match.group(2))
        tokens_b = iteration * TOKENS_PER_ITER / 1e9
        if tokens_b >= MIN_TOKENS_B:
            eval_points.append((iteration, tokens_b, loss))

    if eval_points:
        max_iter = max(iteration for iteration, _, _ in eval_points)
        eval_points = [point for point in eval_points if point[0] < max_iter]
    return eval_points


def build_ranking_text(final_points):
    ranked = sorted(final_points, key=lambda item: item[1])
    lines = ["Ranking (last retained point)"]
    for index, (label, loss) in enumerate(ranked, start=1):
        lines.append(f"{index}. {label} ({loss:.3f})")
    return "\n".join(lines)


def main():
    fig = plt.figure(figsize=(14.2, 6.9))
    grid = fig.add_gridspec(1, 2, width_ratios=[4.75, 2.0], wspace=0.02)
    ax = fig.add_subplot(grid[0, 0])
    panel_ax = fig.add_subplot(grid[0, 1])
    panel_ax.axis("off")

    all_losses = []
    ranked_series = []

    for run in RUNS:
        log_path = Path(run["path"])
        if not log_path.exists():
            print(f"[WARN] missing {run['path']}")
            continue

        eval_points = extract_eval_points(log_path)
        xs = [tokens_b for _, tokens_b, _ in eval_points]
        ys = [loss for _, _, loss in eval_points]

        if not xs:
            print(f"[WARN] no eval points in {run['path']} after {MIN_TOKENS_B}B")
            continue

        line, = ax.plot(
            xs,
            ys,
            label=run["label"],
            color=run["color"],
            linestyle=run["linestyle"],
            linewidth=2.2,
        )
        all_losses.extend(ys)
        ranked_series.append(
            {
                "line": line,
                "label": run["label"],
                "loss": ys[-1],
            }
        )

    ax.set_xlabel("Tokens (B)", fontsize=14)
    ax.set_ylabel("Validation Loss", fontsize=18)
    ax.set_title(TITLE, fontsize=19, pad=14)
    ax.grid(alpha=0.20, linewidth=0.9)
    ax.tick_params(labelsize=11.5)

    if all_losses:
        ymin, ymax = min(all_losses), max(all_losses)
        pad = (ymax - ymin) * 0.08
        ax.set_ylim(ymin - pad, ymax + pad)

    ranked_series.sort(key=lambda item: item["loss"])
    handles = [item["line"] for item in ranked_series]
    labels = [
        f"{index}. {item['label']} ({item['loss']:.3f})"
        for index, item in enumerate(ranked_series, start=1)
    ]
    panel_ax.legend(
        handles,
        labels,
        loc="upper left",
        bbox_to_anchor=(0.03, 0.98),
        borderaxespad=0.0,
        frameon=True,
        fontsize=11.3,
        title="Methods\n(last retained loss)",
        title_fontsize=12.5,
        handlelength=2.45,
        labelspacing=0.52,
    )

    plt.savefig(OUT, dpi=220, bbox_inches="tight")
    print("Saved:", OUT)


if __name__ == "__main__":
    main()
