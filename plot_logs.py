#!/usr/bin/env python3
import re
import sys
from pathlib import Path
import matplotlib.pyplot as plt

# 训练配置 -> 用来把 iter 转成 tokens
BATCH = 32
SEQ = 512
ACC = 1
WORLD = 1
TOKENS_PER_ITER = BATCH * SEQ * ACC * WORLD

MIN_TOKENS_B = 0.0
SKIP_LAST_EVAL = True  # 如需去掉最后一次 full-eval，改为 True
EVAL_RE = re.compile(r">Eval: Iter=(\d+).*?val_loss=([0-9.]+)")

# 输出图片路径（本机路径）
OUT_PATH = "/Users/jiujiujiu/Desktop/muon_compare.png"

def label_from_name(path):
    return Path(path).stem

def main():
    plt.figure(figsize=(9,5))
    all_ys = []
    final_points = []

    log_paths = sys.argv[1:]
    if not log_paths:
        print("Please provide log files, e.g.:")
        print("  python plot_logs.py /path/to/log1.log /path/to/log2.log")
        return

    for log_path in log_paths:
        p = Path(log_path)
        if not p.exists():
            print(f"[WARN] missing {log_path}")
            continue

        points = []
        for line in p.read_text(errors="ignore").splitlines():
            m = EVAL_RE.search(line)
            if m:
                points.append((int(m.group(1)), float(m.group(2))))
        if SKIP_LAST_EVAL and points:
            points = points[:-1]

        xs, ys = [], []
        for it, loss in points:
            tokens_b = it * TOKENS_PER_ITER / 1e9
            if tokens_b >= MIN_TOKENS_B:
                xs.append(tokens_b)
                ys.append(loss)

        if xs:
            label = label_from_name(log_path)
            plt.plot(xs, ys, label=label)
            all_ys.extend(ys)
            final_points.append((label, ys[-1]))
        else:
            print(f"[WARN] no eval points in {log_path} after {MIN_TOKENS_B}B")

    plt.xlabel("Tokens (B)")
    plt.ylabel("Validation Loss")
    plt.title(f"124M SlimPajama 1.5B — Optimizer Curves (>= {MIN_TOKENS_B}B)")
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
    out = Path(OUT_PATH)
    out.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out, dpi=200, bbox_inches="tight")
    print("Saved:", out)

if __name__ == "__main__":
    main()
