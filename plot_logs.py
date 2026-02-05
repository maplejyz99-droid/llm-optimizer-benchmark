#!/usr/bin/env python3
import re
from pathlib import Path
import matplotlib.pyplot as plt

LOGS = [
    "/root/work/llm-optimizer-benchmark/logs/adamw_124m_15B.log",
    "/root/work/llm-optimizer-benchmark/logs/adopt_124m_1p5B.log",
    "/root/work/llm-optimizer-benchmark/logs/lion_124m_1p5B.log",
    "/root/work/llm-optimizer-benchmark/logs/marsadamw_124m_1p5B_gpu1.log",
    "/root/work/llm-optimizer-benchmark/logs/marslion_124m_1p5B_gpu2.log",
    "/root/work/llm-optimizer-benchmark/logs/marsshampoo_124m_1p5B_gpu3.log",
    "/root/work/llm-optimizer-benchmark/logs/sfadamw_124m_1p5B_gpu3.log",
    "/root/work/llm-optimizer-benchmark/logs/signum_124m_1p5B.log",
    "/root/work/llm-optimizer-benchmark/logs/soap_124m_1p5B_gpu0.log",
    "/root/work/llm-optimizer-benchmark/logs/sophiag_124m_1p5B_b16a2_gpu1.log",
]

BATCH = 32
SEQ = 512
ACC = 1
WORLD = 1
TOKENS_PER_ITER = BATCH * SEQ * ACC * WORLD

MIN_TOKENS_B = 0.15
EVAL_RE = re.compile(r">Eval: Iter=(\d+).*?val_loss=([0-9.]+)")

def label_from_name(path):
    stem = Path(path).stem
    stem = re.sub(r"_124m.*", "", stem)
    stem = re.sub(r"_gpu\d+", "", stem)
    return stem

def main():
    plt.figure(figsize=(9,5))
    all_ys = []
    final_points = []

    for log_path in LOGS:
        p = Path(log_path)
        if not p.exists():
            print(f"[WARN] missing {log_path}")
            continue

        eval_points = []
        for line in p.read_text(errors="ignore").splitlines():
            m = EVAL_RE.search(line)
            if m:
                it = int(m.group(1))
                loss = float(m.group(2))
                tokens_b = it * TOKENS_PER_ITER / 1e9
                if tokens_b >= MIN_TOKENS_B:
                    eval_points.append((it, tokens_b, loss))

        if eval_points:
            max_iter = max(it for it, _, _ in eval_points)
            eval_points = [pt for pt in eval_points if pt[0] < max_iter]

        xs = [t for _, t, _ in eval_points]
        ys = [l for _, _, l in eval_points]

        if xs:
            label = label_from_name(log_path)
            plt.plot(xs, ys, label=label)
            all_ys.extend(ys)
            final_points.append((label, ys[-1]))
        else:
            print(f"[WARN] no eval points in {log_path} after {MIN_TOKENS_B}B")

    plt.xlabel("Tokens (B)")
    plt.ylabel("Validation Loss")
    plt.title("124M SlimPajama 1.5B — Optimizer Curves (>=0.15B)")
    plt.grid(alpha=0.3)

    if all_ys:
        ymin, ymax = min(all_ys), max(all_ys)
        pad = (ymax - ymin) * 0.08
        plt.ylim(ymin - pad, ymax + pad)

    # 保留颜色图例
    plt.legend(loc="upper right", frameon=True)

    # 排名框
    final_points.sort(key=lambda x: x[1])
    ranking_text = "Ranking (↓)\n" + "\n".join(
        [f"{i+1}. {name}" for i, (name, _) in enumerate(final_points)]
    )

    ax = plt.gca()
    ax.text(
        1.02, 0.5, ranking_text,
        transform=ax.transAxes,
        fontsize=10,
        va="center",
        ha="left",
        bbox=dict(boxstyle="round,pad=0.4", facecolor="white", alpha=0.9)
    )

    plt.tight_layout()
    out = "/root/work/llm-optimizer-benchmark/optimizer_curves_1p5B_ranked.png"
    plt.savefig(out, dpi=200, bbox_inches="tight")
    print("Saved:", out)

if __name__ == "__main__":
    main()
