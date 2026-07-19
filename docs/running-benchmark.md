# Running benchmark experiments

## Output policy

There are two machine-local roots with different jobs:

- `runs/` is the raw, append-free evidence store. Every launch gets one unique
  run directory containing metrics, logs, command metadata, and status.
- `results/` is reserved for reviewed/curated comparisons, tables, and plots.
  The launcher does not promote a run into results automatically.

Both names are ignored by Git. On the current remote machine the recommended
targets are:

```text
runs    -> /root/autodl-tmp/llmopt/runs
results -> /root/work/llmopt-results
```

Historical `./exps`, `/root/autodl-tmp/llmopt/exps`, standalone logs, and curated
results are not moved or deleted. Direct `python ./src/main.py` remains compatible
with the upstream default `--results_base_folder ./exps`; the new layout applies
when using `scripts/launch_run.py`.

## One-time setup per clone

From the repository root:

```bash
source /root/miniconda3/etc/profile.d/conda.sh
conda activate llmopt310
cd /root/work/llm-optimizer-benchmark-staging

python ./scripts/setup_runtime_paths.py \
  --runs-root /root/autodl-tmp/llmopt/runs \
  --results-root /root/work/llmopt-results
```

The command is idempotent when the two links already point at the requested
targets. It refuses to replace an existing file, directory, wrong symlink, or
broken symlink. Use `--dry-run` to inspect the plan without creating anything.

## What one completed run contains

For suite `benchmark/optimizer-comparison` and run ID
`sophiag-124m-slimpajama-1p5b-seed0`:

```text
runs/
└── benchmark/
    └── optimizer-comparison/
        └── sophiag-124m-slimpajama-1p5b-seed0/
            ├── run_manifest.json
            ├── summary.json
            ├── evaluations/
            │   └── <evaluation-protocol-id>/
            │       └── summary.json
            ├── ckpts/                         # only when checkpoint intervals > 0
            ├── launch/
            │   ├── command.txt                # command with credentials redacted
            │   └── status.json                # starting/running/completed/failed
            └── logs/
                ├── stdout.log
                ├── gpu_smi.csv                # only with --monitor-gpu
                └── gpu_smi.error.log           # only if sampling later degrades
```

`src/main.py` writes the manifest, summaries, evaluation records, and optional
checkpoints. The launcher writes `launch/` and `logs/`. A child exit code of zero
is not enough for success: the launcher marks a run `completed` only when the
manifest, root summary, and versioned evaluation summary exist and carry the same
run identity.

## Full SophiaG example

This example keeps the original 124M Llama, SlimPajama, SophiaG, batch, and
1.5B-token training settings. It uses a fixed final validation-token budget for
an optimizer-comparison suite. Remove `--final_eval_tokens 10485760` when strict
EPFL-compatible full final evaluation is required.

First establish the environment and shared data path:

```bash
source /root/miniconda3/etc/profile.d/conda.sh
conda activate llmopt310
cd /root/work/llm-optimizer-benchmark-staging

export HF_HOME=/root/autodl-tmp/hf
export HF_DATASETS_CACHE=/root/autodl-tmp/hf/datasets
export TRANSFORMERS_CACHE=/root/autodl-tmp/hf/transformers
export HUGGINGFACE_HUB_CACHE=/root/autodl-tmp/hf/hub
export TMPDIR=/root/autodl-tmp/hf/tmp
export LLMOPT_DATASETS_DIR=/root/autodl-tmp/llmopt/datasets

RUN_ID=sophiag-124m-slimpajama-1p5b-seed0
RUNS_ROOT=/root/autodl-tmp/llmopt/runs
```

Check the final command without creating a run directory:

```bash
python ./scripts/launch_run.py \
  --runs-root "$RUNS_ROOT" \
  --suite benchmark/optimizer-comparison \
  --run-id "$RUN_ID" \
  --gpu-ids 1 \
  --monitor-gpu \
  --dry-run \
  -- \
  --config_format base \
  --model llama \
  --n_layer 12 --n_head 12 --n_embd 768 \
  --sequence_length 512 \
  --dataset slimpajama \
  --batch_size 32 --acc_steps 1 \
  --opt sophiag \
  --lr 3e-4 \
  --warmup_steps 2000 \
  --weight_decay 0.1 \
  --beta1 0.9 --beta2 0.999 \
  --grad_clip 0.5 \
  --scheduler cos \
  --sophia_rho 0.04 \
  --sophia_bs 480 \
  --dtype bfloat16 \
  --device cuda:0 \
  --iterations 91553 \
  --seed 0 --data_seed 1337 \
  --eval_interval 200 \
  --log_interval 50 \
  --final_eval_tokens 10485760
```

Then start the same command under `nohup`; the small bootstrap log is only for a
failure before the run bundle is initialized. Training output goes to the run's
`logs/stdout.log`:

```bash
nohup python -u ./scripts/launch_run.py \
  --runs-root "$RUNS_ROOT" \
  --suite benchmark/optimizer-comparison \
  --run-id "$RUN_ID" \
  --gpu-ids 1 \
  --monitor-gpu \
  -- \
  --config_format base \
  --model llama \
  --n_layer 12 --n_head 12 --n_embd 768 \
  --sequence_length 512 \
  --dataset slimpajama \
  --batch_size 32 --acc_steps 1 \
  --opt sophiag \
  --lr 3e-4 \
  --warmup_steps 2000 \
  --weight_decay 0.1 \
  --beta1 0.9 --beta2 0.999 \
  --grad_clip 0.5 \
  --scheduler cos \
  --sophia_rho 0.04 \
  --sophia_bs 480 \
  --dtype bfloat16 \
  --device cuda:0 \
  --iterations 91553 \
  --seed 0 --data_seed 1337 \
  --eval_interval 200 \
  --log_interval 50 \
  --final_eval_tokens 10485760 \
  > "/tmp/${RUN_ID}.launcher.log" 2>&1 &

echo "launcher PID=$!"
```

Monitor the run through the repository-visible link:

```bash
tail -f "runs/benchmark/optimizer-comparison/${RUN_ID}/logs/stdout.log"
cat "runs/benchmark/optimizer-comparison/${RUN_ID}/launch/status.json"
```

The launcher PID and training child PID are also recorded in `status.json`, so a
separate `.pid` file is unnecessary.

## Checkpoint and resume boundary

Checkpoint creation remains opt-in. Without these training arguments, no
periodic checkpoint is requested:

```bash
--latest_ckpt_interval 1000 \
--permanent_ckpt_interval 10000
```

Add them only when the experiment protocol calls for checkpoints. The v1
launcher deliberately accepts only a new run directory and rejects
`--resume_from`; it never truncates or disguises an earlier attempt. Exact resume
is supported by the training code, but standardized multi-attempt launch history
needs a separate launcher extension before resume is exposed through this
workflow. Keep any interrupted checkpoint directory unchanged.

## Single process and DDP

Without `--nproc-per-node`, the launcher executes the active interpreter as:

```text
python src/main.py ...
```

For DDP (Distributed Data Parallel, one synchronized worker per GPU), request the
worker count on the launcher and the backend in the training arguments:

```bash
python ./scripts/launch_run.py \
  --runs-root /root/autodl-tmp/llmopt/runs \
  --suite benchmark/ddp-smoke \
  --run-id adamw-124m-2gpu-smoke \
  --gpu-ids 0,1 \
  --nproc-per-node 2 \
  --monitor-gpu \
  -- \
  --config_format base \
  --model llama \
  --dataset fineweb \
  --distributed_backend nccl \
  --opt adamw \
  --iterations 20 \
  --final_eval_batches 2
```

The launcher uses `python -m torch.distributed.run`, ensuring DDP uses the same
Python environment as the launcher. The GPU-ID count must match the worker count,
and global `batch_size * acc_steps` must be divisible by that count.

## Safety rules

- Run `--dry-run` first, especially before a long background launch.
- Use a unique `run-id`; existing run directories are never overwritten.
- Do not pass `--results_base_folder`, `--experiment_name`, or `--resume_from`
  after `--`; those conflict with the run-bundle contract.
- `--gpu-ids` sets `CUDA_VISIBLE_DEVICES`. Inside training, `cuda:0` means the
  first GPU in that visible list, not necessarily physical GPU 0.
- `--monitor-gpu` requires explicit GPU IDs. Sampling failures are recorded
  separately and do not erase a valid training result.
- Credentials in recognized secret/notification CLI options are redacted from
  `command.txt`; prefer environment variables for credentials and never place
  secrets in a run ID.
- A status left at `running` after `SIGKILL`, host reboot, or power loss is stale
  evidence, not proof that the process is still alive.
