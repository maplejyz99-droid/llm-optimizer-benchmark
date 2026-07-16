# Hardware Compare Runbook: A800 vs RTX 5090

## Goal

Generate normalized CSVs, internal comparisons, cross-hardware comparisons,
communication-overhead estimates, figures, and Markdown reports after both A800
and RTX 5090 experiments have produced either `results.csv` or per-run
`summary.json` files.

## Required Inputs

Preferred CSV inputs:

```text
/root/autodl-tmp/llmopt/logs/a800/results.csv
/root/autodl-tmp/llmopt/logs/5090/results.csv
```

Summary-root fallback inputs:

```text
/root/autodl-tmp/llmopt/exps/a800
/root/autodl-tmp/llmopt/exps/5090-comparison-20260701
```

CSV inputs preserve more fields when stdout logs and SMI samples are available.
Summary-root inputs are enough for offline speed, batch-semantics, internal
scaling, and cross-hardware comparisons, but memory fields stay empty unless
SMI-derived results were already summarized.

## Run

```bash
cd /root/work/llm-optimizer-benchmark

/root/miniconda3/envs/llmopt310/bin/python scripts/compare/hardware_compare.py \
  --a800-results /root/autodl-tmp/llmopt/logs/a800/results.csv \
  --gpu5090-results /root/autodl-tmp/llmopt/logs/5090/results.csv \
  --out-dir /root/autodl-tmp/llmopt/logs/compare \
  --figure-dir /root/autodl-tmp/llmopt/figures/hardware \
  --docs-dir /root/work/llm-optimizer-benchmark/docs
```

If the machine only has experiment summaries:

```bash
cd /root/work/llm-optimizer-benchmark

/root/miniconda3/envs/llmopt310/bin/python scripts/compare/hardware_compare.py \
  --a800-summary-root /root/autodl-tmp/llmopt/exps/a800 \
  --gpu5090-summary-root /root/autodl-tmp/llmopt/exps/5090-comparison-20260701 \
  --out-dir /root/autodl-tmp/llmopt/logs/compare \
  --figure-dir /root/autodl-tmp/llmopt/figures/hardware \
  --docs-dir /root/work/llm-optimizer-benchmark/docs
```

## Output CSVs

```text
/root/autodl-tmp/llmopt/logs/compare/a800_results_normalized.csv
/root/autodl-tmp/llmopt/logs/compare/5090_results_normalized.csv
/root/autodl-tmp/llmopt/logs/compare/hardware_compare_merged.csv
/root/autodl-tmp/llmopt/logs/compare/exact_match_compare.csv
/root/autodl-tmp/llmopt/logs/compare/same_global_batch_compare.csv
/root/autodl-tmp/llmopt/logs/compare/hardware_best_compare.csv
/root/autodl-tmp/llmopt/logs/compare/a800_internal_summary.csv
/root/autodl-tmp/llmopt/logs/compare/5090_internal_summary.csv
/root/autodl-tmp/llmopt/logs/compare/ddp_communication_overhead.csv
/root/autodl-tmp/llmopt/logs/compare/profiler_nccl_summary.csv
/root/autodl-tmp/llmopt/logs/compare/unmatched_or_oom.csv
```

## Output Figures

```text
/root/autodl-tmp/llmopt/figures/hardware/throughput_by_case.png
/root/autodl-tmp/llmopt/figures/hardware/iter_dt_speedup_heatmap.png
/root/autodl-tmp/llmopt/figures/hardware/peak_memory_by_case.png
/root/autodl-tmp/llmopt/figures/hardware/capacity_max_batch.png
/root/autodl-tmp/llmopt/figures/hardware/exact_vs_practical_comparison.png
/root/autodl-tmp/llmopt/figures/hardware/ddp_overhead_seconds.png
/root/autodl-tmp/llmopt/figures/hardware/parallel_efficiency.png
/root/autodl-tmp/llmopt/figures/a800/internal_single_vs_ddp.png
/root/autodl-tmp/llmopt/figures/5090/internal_single_vs_ddp.png
```

The NCCL, all-reduce, and topology figures are placeholders until profiler and
microbenchmark evidence is added.

## Output Reports

```text
docs/hardware-compare-a800-vs-5090-report.md
docs/a800-internal-comparison-report.md
docs/5090-internal-comparison-report.md
docs/hardware-compare-main-benchmark-report.md
docs/hardware-compare-capacity-report.md
docs/hardware-compare-track3-report.md
docs/hardware-communication-analysis-report.md
docs/hardware-compare-methodology.md
```

## Interpretation Rules

- Use `exact_match_compare.csv` for the cleanest paper-facing hardware claims.
- Use `same_global_batch_compare.csv` for practical claims when RTX 5090 cannot
  run the A800 micro batch but can keep the same global effective batch by
  increasing accumulation.
- Treat `observed_ddp_overhead_s` as observed DDP overhead, not pure
  communication time.
- Keep OOM and batch-semantics mismatches in `unmatched_or_oom.csv`; do not
  silently drop them.
