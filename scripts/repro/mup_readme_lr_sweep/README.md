# muP README LR Sweep Reproduction

This directory runs a project-local analogue of the first figure in
`microsoft/mup`'s README: training loss against learning rate for Transformers
of different widths.

This is not an exact upstream reproduction. It uses this repository's Llama
training stack, SlimPajama data, and the current `mup_llama` implementation.

## Workflow

From the repository root:

```bash
python scripts/diagnostics/check-mup-llama-param-groups.py
python scripts/diagnostics/check-mup-llama-coordinate.py

python scripts/repro/mup_readme_lr_sweep/run_sweep.py --phase calibrate --dry-run
python scripts/repro/mup_readme_lr_sweep/run_sweep.py --phase calibrate

python scripts/repro/mup_readme_lr_sweep/run_sweep.py --phase coarse --dry-run
python scripts/repro/mup_readme_lr_sweep/run_sweep.py --phase coarse

python scripts/repro/mup_readme_lr_sweep/parse_logs.py
python scripts/repro/mup_readme_lr_sweep/plot_results.py
```

If the best coarse learning rate is at an edge, run a fine sweep with an outer
point. If it is in the middle, run midpoint values:

```bash
python scripts/repro/mup_readme_lr_sweep/run_sweep.py --phase fine --lrs 5e-4,2e-3
python scripts/repro/mup_readme_lr_sweep/parse_logs.py
python scripts/repro/mup_readme_lr_sweep/plot_results.py
```

After choosing one LR for SP and one LR for muP:

```bash
python scripts/repro/mup_readme_lr_sweep/run_sweep.py --phase long --sp-lr 1e-3 --mup-lr 1e-3
python scripts/repro/mup_readme_lr_sweep/parse_logs.py
python scripts/repro/mup_readme_lr_sweep/plot_results.py --plot-long
```

## Outputs

- Raw logs: `logs/mup_readme_lr_sweep/<phase>/*.log`
- Per-run metadata: `logs/mup_readme_lr_sweep/<phase>/*.meta.json`
- Summary CSV: `logs/mup_readme_lr_sweep/results.csv`
- Figures: `logs/mup_readme_lr_sweep/figures/`

Failed or diverged runs are kept in the CSV and excluded from lines in the
default plot. Inspect the CSV before interpreting the final figure.

## V2 Larger-Width Sweep

V2 keeps the same external-script boundary and still does not change `src/`.
It uses `sequence_length=512`, `batch_size=64`, `acc_steps=1`,
`scheduler=cos`, and starts with a GPU-memory probe.

```bash
python scripts/repro/mup_readme_lr_sweep/run_sweep.py --phase probe_v2 --dry-run
python scripts/repro/mup_readme_lr_sweep/run_sweep.py --phase probe_v2 --skip-existing

python scripts/repro/mup_readme_lr_sweep/parse_logs.py \
  --output logs/mup_readme_lr_sweep/results_v2.csv \
  --include-phases probe_v2 \
  --probe-summary logs/mup_readme_lr_sweep/probe_summary.csv
```

Choose up to four widths that pass the probe for both SP and muP, always
including 768. Then run:

```bash
python scripts/repro/mup_readme_lr_sweep/run_sweep.py \
  --phase sweep_v2 --widths 768,1536,2048,3072 --skip-existing

python scripts/repro/mup_readme_lr_sweep/parse_logs.py \
  --output logs/mup_readme_lr_sweep/results_v2.csv \
  --include-phases probe_v2,sweep_v2

python scripts/repro/mup_readme_lr_sweep/plot_results.py \
  --csv logs/mup_readme_lr_sweep/results_v2.csv \
  --phases sweep_v2 \
  --lr-output-name sp_vs_mup_remote_lr_sweep_v2.png
```

If 2048 exceeds the 70GB target, re-probe with the uniform fallback batch:

```bash
python scripts/repro/mup_readme_lr_sweep/run_sweep.py --phase probe_v2_b32 --skip-existing
python scripts/repro/mup_readme_lr_sweep/parse_logs.py \
  --output logs/mup_readme_lr_sweep/results_v2.csv \
  --include-phases probe_v2,probe_v2_b32 \
  --probe-summary logs/mup_readme_lr_sweep/probe_summary.csv
```

Use `sweep_v2_b32` / `long_v2_b32` with the selected widths if the b32 probe is
the accepted setting.

## V2 L4 Full Width Transfer Sweep

The L4 plan keeps the experiment close to the first successful 4-layer run, but
uses a larger token batch and a denser LR grid.

```bash
python scripts/repro/mup_readme_lr_sweep/run_sweep.py --phase probe_v2_l4 --dry-run
python scripts/repro/mup_readme_lr_sweep/run_sweep.py --phase probe_v2_l4 --skip-existing

python scripts/repro/mup_readme_lr_sweep/parse_logs.py \
  --output logs/mup_readme_lr_sweep/results_v2_l4.csv \
  --include-phases probe_v2_l4 \
  --probe-summary logs/mup_readme_lr_sweep/probe_summary_v2_l4.csv

python scripts/repro/mup_readme_lr_sweep/run_sweep.py \
  --phase sweep_v2_l4 --widths 128,256,512,768,1024,1536,2048 --skip-existing

python scripts/repro/mup_readme_lr_sweep/parse_logs.py \
  --output logs/mup_readme_lr_sweep/results_v2_l4.csv \
  --include-phases probe_v2_l4,sweep_v2_l4

python scripts/repro/mup_readme_lr_sweep/plot_results.py \
  --csv logs/mup_readme_lr_sweep/results_v2_l4.csv \
  --phases sweep_v2_l4 \
  --lr-output-name sp_vs_mup_remote_lr_sweep_v2_l4.png
```

Use the sweep CSV to choose:

- global-best LR for `long_v2_l4_global`
- width-128-selected LR for `long_v2_l4_anchor128`
- width-768-selected LR for `long_v2_l4_anchor768`

```bash
python scripts/repro/mup_readme_lr_sweep/select_lrs.py \
  --csv logs/mup_readme_lr_sweep/results_v2_l4.csv \
  --phase sweep_v2_l4
```

For a resumable remote background run, use the pipeline wrapper. It reuses
completed jobs via `--skip-existing`, then parses, plots, selects global and
anchor LRs, and runs the three main long-run phases:

```bash
nohup python scripts/repro/mup_readme_lr_sweep/run_v2_l4_pipeline.py \
  --skip-probe \
  > logs/mup_readme_lr_sweep/v2_l4_pipeline.log 2>&1 &
```

Manual long-run example after selecting one LR per alias:

```bash
python scripts/repro/mup_readme_lr_sweep/run_sweep.py \
  --phase long_v2_l4_global --widths 128,512,768,1024,2048 \
  --sp-lr 7e-4 --mup-lr 3e-3 --skip-existing

python scripts/repro/mup_readme_lr_sweep/plot_results.py \
  --csv logs/mup_readme_lr_sweep/results_v2_l4.csv \
  --phases sweep_v2_l4 \
  --plot-long --long-phases long_v2_l4_global \
  --lr-output-name sp_vs_mup_remote_lr_sweep_v2_l4.png \
  --long-output-name sp_vs_mup_remote_longrun_global_v2_l4.png
```
