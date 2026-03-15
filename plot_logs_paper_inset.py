#!/usr/bin/env python3
from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle

from plot_logs_paper import RUNS, extract_eval_points

TITLE = "124M SlimPajama 1.5B - Optimizer Curves with Tail Zoom"
OUT = "/root/work/llm-optimizer-benchmark/optimizer_curves_1p5B_ranked_paper_inset.png"
ZOOM_XMIN = 1.30
ZOOM_XMAX = 1.50
MAIN_INSET_BOUNDS = [0.66, 0.71, 0.24, 0.23]
INSET_XTICKS = [1.30, 1.35, 1.40, 1.45, 1.50]
LABEL_PANEL_X = 0.12
LABEL_PANEL_Y_TOP = 0.96
LABEL_PANEL_Y_BOTTOM = 0.56
INSET_LABEL_MAP = {
    "AdamW": "AdamW",
    "ADOPT": "ADOPT",
    "Lion": "Lion",
    "MARS-AdamW": "MARS-A",
    "MARS-Lion": "MARS-L",
    "MARS-Shampoo": "MARS-S",
    "SF-AdamW": "SF-AdamW",
    "Signum": "Signum",
    "SOAP": "SOAP",
    "Sophia-G": "Sophia",
    "C-AdamW": "C-AdamW",
    "AdamW+Magma": "AW+Magma",
}


def compute_tail_limits(series, xmin, xmax):
    tail_losses = []
    for _, xs, ys in series:
        for x, y in zip(xs, ys):
            if xmin <= x <= xmax:
                tail_losses.append(y)

    if not tail_losses:
        return None

    ymin = min(tail_losses)
    ymax = max(tail_losses)
    pad = max((ymax - ymin) * 0.08, 0.01)
    return ymin - pad, ymax + pad


def compute_panel_label_positions(tail_points):
    if not tail_points:
        return {}

    ordered = sorted(tail_points, key=lambda item: item["y"], reverse=True)
    count = len(ordered)
    if count == 1:
        ys = [(LABEL_PANEL_Y_TOP + LABEL_PANEL_Y_BOTTOM) / 2]
    else:
        span = LABEL_PANEL_Y_TOP - LABEL_PANEL_Y_BOTTOM
        step = span / (count - 1)
        ys = [LABEL_PANEL_Y_TOP - index * step for index in range(count)]

    positions = {}
    for index, item in enumerate(ordered):
        positions[item["short_label"]] = (
            LABEL_PANEL_X,
            ys[index],
        )
    return positions


def annotate_tail_labels(inset_ax, panel_ax, tail_points):
    positions = compute_panel_label_positions(tail_points)

    for item in sorted(tail_points, key=lambda item: item["y"], reverse=True):
        label_x_frac, label_y_frac = positions[item["short_label"]]
        inset_ax.annotate(
            item["short_label"],
            xy=(item["x"], item["y"]),
            xycoords="data",
            xytext=(label_x_frac, label_y_frac),
            textcoords=panel_ax.transAxes,
            ha="left",
            va="center",
            fontsize=10.2,
            color=item["color"],
            clip_on=False,
            annotation_clip=False,
            arrowprops=dict(
                arrowstyle="-",
                color=item["color"],
                lw=1.05,
                shrinkA=0,
                shrinkB=0,
                connectionstyle="arc3,rad=0.0",
            ),
        )


def find_tail_anchor(xs, ys, xmin, xmax):
    anchor = None
    for x, y in zip(xs, ys):
        if xmin <= x <= xmax:
            anchor = (x, y)
    return anchor


def main():
    fig = plt.figure(figsize=(15.2, 7.1))
    outer = fig.add_gridspec(1, 2, width_ratios=[5.1, 1.9], wspace=0.05)
    ax = fig.add_subplot(outer[0, 0])
    panel_ax = fig.add_subplot(outer[0, 1])
    panel_ax.axis("off")
    inset_ax = ax.inset_axes(MAIN_INSET_BOUNDS)
    inset_ax.set_facecolor((1.0, 1.0, 1.0, 0.96))

    all_losses = []
    plotted_series = []
    ranked_series = []
    tail_points = []

    for run in RUNS:
        log_path = Path(run["path"])
        if not log_path.exists():
            print(f"[WARN] missing {run['path']}")
            continue

        eval_points = extract_eval_points(run["path"])
        xs = [tokens_b for _, tokens_b, _ in eval_points]
        ys = [loss for _, _, loss in eval_points]

        if not xs:
            print(f"[WARN] no eval points in {run['path']}")
            continue

        line, = ax.plot(
            xs,
            ys,
            label=run["label"],
            color=run["color"],
            linestyle=run["linestyle"],
            linewidth=2.2,
        )
        plotted_series.append((run, xs, ys))
        all_losses.extend(ys)
        tail_anchor = find_tail_anchor(xs, ys, ZOOM_XMIN, ZOOM_XMAX)
        if tail_anchor is not None:
            tail_points.append(
                {
                    "x": tail_anchor[0],
                    "y": tail_anchor[1],
                    "short_label": INSET_LABEL_MAP.get(run["label"], run["label"]),
                    "color": run["color"],
                }
            )
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
        ymin = min(all_losses)
        ymax = max(all_losses)
        pad = (ymax - ymin) * 0.08
        ax.set_ylim(ymin - pad, ymax + pad)

    for run, xs, ys in plotted_series:
        inset_ax.plot(
            xs,
            ys,
            color=run["color"],
            linestyle=run["linestyle"],
            linewidth=1.9,
        )

    inset_ax.set_xlim(ZOOM_XMIN, ZOOM_XMAX)
    inset_ax.set_xticks(INSET_XTICKS)
    tail_limits = compute_tail_limits(plotted_series, ZOOM_XMIN, ZOOM_XMAX)
    if tail_limits is not None:
        inset_ax.set_ylim(*tail_limits)
        ax.add_patch(
            Rectangle(
                (ZOOM_XMIN, tail_limits[0]),
                ZOOM_XMAX - ZOOM_XMIN,
                tail_limits[1] - tail_limits[0],
                fill=False,
                linewidth=1.15,
                edgecolor="0.40",
                alpha=0.95,
            )
        )
        annotate_tail_labels(inset_ax, panel_ax, tail_points)

    inset_ax.set_title("Tail Zoom (1.30B-1.50B)", fontsize=12, pad=8)
    inset_ax.grid(alpha=0.18, linewidth=0.8)
    inset_ax.tick_params(labelsize=8)
    inset_ax.set_xlabel("Tokens (B)", fontsize=9)
    inset_ax.set_ylabel("Val Loss", fontsize=9)

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
        bbox_to_anchor=(0.02, 0.48),
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
