# 1B benchmark: model contract and collaborator handoff

Read this document first when adding an optimizer or preparing a 1B comparison.
The authoritative recipes are in [`scripts/1b/`](../scripts/1b/); their filenames
match the 17 recipes in [`scripts/720m/`](../scripts/720m/). The existing Llama
implementation already supports this shape, so there is no separate model class
or training-budget subsystem.

## Model contract

This fork adopts the **1026M** configuration from the optimizer benchmark paper's
[Appendix D.3, Table 2](https://arxiv.org/html/2509.01440v1#A4.T2). That table is
from a timing experiment; it does not establish tuned 1B convergence results.

| Setting | Value |
| --- | --- |
| Config / model | `base` / `llama` (dense, standard parameterization) |
| Layers | 24 |
| Hidden width | 1792 |
| Attention | 14 heads, head dimension 128, ordinary multi-head attention |
| SwiGLU intermediate width | 4864, derived with `multiple_of=256` |
| Vocabulary | 50304 model entries, GPT-2 tokenizer |
| Input / output embeddings | Tied |
| Position / normalization | RoPE / RMSNorm, epsilon `1e-5` |
| Bias / dropout | `False` / `0.0` |
| Initialization setting | `init_std=0.02`, using the existing model initializer |
| Exact trainable parameters | **1,026,086,656**, shared embeddings counted once |
| Context length | **512** |
| Training dtype | `bfloat16` autocast; this does not imply BF16 optimizer state |
| Objective | Next-token cross-entropy, dense model with no auxiliary z-loss |

The model flags are:

```sh
--config_format base --model llama \
--n_layer 24 --n_embd 1792 --n_head 14 \
--sequence_length 512 --vocab_size 50304 --multiple_of 256 \
--dropout 0.0 --init_std 0.02 --rmsnorm_eps 1e-5 --bias False
```

Leave `--untied_embeds`, `--moe`, and muP disabled to preserve this contract.
See [`src/models/llama.py`](../src/models/llama.py) and the defaults in
[`src/config/base.py`](../src/config/base.py). This shape differs from a
17-layer, width-2048 capacity probe sometimes also called "1B".

## Shared comparison protocol

- Use the same versioned **FineWeb-30B** token artifact and held-out validation
  split for every optimizer: GPT-2 tokens, `uint16`, 30B training tokens and
  100M validation tokens. This is the fork's comparison dataset, whereas the
  paper used a 100B-token pool. Do not silently mix those datasets.
- Keep sequence length **512** and global batch **1984 sequences**, hence
  **1,015,808 training tokens per optimizer update**.
- The recipes use `--batch_size 62 --acc_steps 32`. These are the CLI values
  **before** DDP adjusts them. In this repository the CLI product is already the
  global batch; do not multiply it by the number of ranks again. DDP requires
  that product to be divisible by world size. The actual per-rank batch and
  accumulation may differ; see [`src/distributed/ddp.py`](../src/distributed/ddp.py).
- Start with `--seed 0 --data_seed 1337`. For replicated comparisons, use the
  same predeclared set of run seeds for every optimizer and keep the data seed
  fixed. The base run seed is recorded separately from rank-adjusted seeds.
- Preserve the matching recipe's optimizer hyperparameters. For the AdamW
  reference these are peak LR `1e-3`, betas `(0.9, 0.999)`, weight decay `0.1`,
  gradient clipping `0.1`, cosine schedule, and 2000 warmup updates. LR ends at
  `0.01` times peak with the current `final_div_factor=1.0` setting.
- Evaluate every 200 updates with the existing 64-batch interim evaluation;
  final evaluation defaults to the complete validation reader. Interim token
  counts depend on evaluation batch size, so preserve that setup or define one
  common token-based final evaluation for the entire comparison. Retain actual
  evaluated-token counts and evaluation protocol IDs in the outputs.

The 720M optimizer settings are inherited starting points, **not tuned 1B
hyperparameters**. Apply a declared tuning policy equally to your optimizer and
the references. Execution batch splitting may be adapted while preserving the
global batch, but methods with microbatch-sensitive auxiliary computations
(such as Sophia's legacy Hessian estimator mode) need a separate semantic check.

## Training horizons

Use three independent training runs per optimizer for the initial comparison:

| Comparison | `--iterations` | Training tokens processed |
| --- | ---: | ---: |
| Short, matching the old 8k horizon | 8000 | 8,126,464,000 |
| Medium, matching the old 16k horizon | 16000 | 16,252,928,000 |
| Approximate Chinchilla reference | **20203** | **20,522,369,024** |
| Optional extended 48k horizon | 48000 | 48,758,784,000 |

Each iteration is an optimizer update, including all accumulation microsteps.
For the Chinchilla reference we use the approximate 20 training tokens per
parameter rule, informed by
[Hoffmann et al.](https://arxiv.org/html/2203.15556v1). This is a convenient
comparison convention, not a measured optimum for this model, data, or optimizer.
The target is 20,521,733,120 tokens; rounding up to a whole update gives 20203.
In particular, **16000 updates is below this 1B reference**.

The arithmetic can be reproduced without importing the training code:

```python
parameters = 1_026_086_656
tokens_per_update = 62 * 32 * 512
target_tokens = 20 * parameters
chinchilla_updates = (target_tokens + tokens_per_update - 1) // tokens_per_update
assert chinchilla_updates == 20_203
```

Choose the horizon before training and let the scheduler decay over that exact
number of updates. An 8k checkpoint from a 48k cosine run is not the completed
8k comparison; its LR schedule is different. Restart each horizon from the same
seeded initialization, rather than resuming the shorter finished run into the
longer one.

The checked-in scripts keep the old editable `--iterations 48000` default for
format consistency. That default does not select a campaign. The extended run
processes more tokens than the 30B training pool contains, so it involves data
reuse; processed tokens are not unique tokens.

Most recipes use 2000 warmup updates. **SF-AdamW is an exception:** its inherited
recipe has `--scheduler none --warmup_steps 8000`. At an 8k horizon that would
mean warming up throughout the run. Choose and record its warmup explicitly
before a short-horizon comparison; do not apply a cosine scheduler to it just
to match AdamW.

## Prepare and launch a reference run

Use the exact shared Git commit for all methods and record your integration
commit. Install the normal dependencies as described in the [README](../README.md).
The hashed [`requirements-ci.lock`](../requirements-ci.lock) targets Linux CPU
tests, not a CUDA training environment; see [reproducibility](reproducibility.md).

Point directly to the agreed token artifact and validate its metadata and sizes:

```sh
export LLMOPT_DATASETS_DIR=/path/to/fineweb-30B
python ./scripts/data/check_fineweb_30b.py --json
```

That preflight does not hash every token. Share the same prepared artifact and
preserve its metadata and run-manifest fingerprints; equal sizes alone do not
prove two separately built datasets are identical. Missing data is not
downloaded automatically.

From the repository root, the following command previews an AdamW 8k run without
starting training or creating a run directory:

```sh
ITERATIONS=8000
SEED=0
NPROC_PER_NODE=1
RUNS_ROOT=/path/to/large-disk/llmopt/runs

python ./scripts/launch_run.py \
  --runs-root "$RUNS_ROOT" \
  --suite fineweb30b-1b \
  --run-id "adamw-1b-t${ITERATIONS}-seed${SEED}" \
  --nproc-per-node "$NPROC_PER_NODE" \
  --dry-run \
  -- \
  --config_format base --model llama --distributed_backend nccl \
  --n_embd 1792 --n_head 14 --n_layer 24 \
  --batch_size 62 --sequence_length 512 --acc_steps 32 \
  --dataset fineweb --iterations "$ITERATIONS" \
  --dropout 0.0 --warmup_steps 2000 --grad_clip 0.1 \
  --seed "$SEED" --data_seed 1337 --dtype bfloat16 \
  --opt adamw --lr 1e-3 --weight_decay 0.1 --scheduler cos \
  --beta1 0.9 --beta2 0.999 \
  --eval_interval 200 --eval_batches 64 --permanent_ckpt_interval 2000
```

Set `ITERATIONS` to `8000`, `16000`, or `20203` for separate runs. The launcher
refuses to overwrite a run directory, so use a new run ID for every attempt.
`NPROC_PER_NODE=1` is an example execution setting, not a benchmark requirement.
Remove `--dry-run` only when ready to start the actual training.
Before that launch, create the selected `RUNS_ROOT` directory (or use
`scripts/setup_runtime_paths.py` as described in the launcher guide); actual
execution requires that root to exist, while a dry run does not.

This example omits W&B. The checked-in shell recipes preserve the old W&B
placeholders; fill them in or remove the `--wandb`, `--wandb_project`, and
`--wandb_entity` options before using those recipes directly. The shell recipes
are static commands and do not forward extra positional arguments: changing
the horizon requires editing a copy of the command or using the launcher above,
not appending `--iterations` to `bash scripts/1b/adamw.sh`.

GN and Magma (`gn-full`, `gn-prox`, `adamw-magma`, `muon-magma`) are currently
restricted to the single-device entry by `src/main.py`. Their 1B scripts use
`python` and omit `--distributed_backend`. If using the launcher for these
methods, omit both `--nproc-per-node` and `--distributed_backend`. The other
recipes retain `torchrun`; recipe availability alone does not certify every
optimizer's distributed semantics.

## Add your own optimizer: coding-agent checklist

Use a new optimizer ID, for example `my-optimizer`, and leave the existing
reference algorithms unchanged. Inspect these integration points:

1. **Implementation:** add the optimizer under [`src/optim/`](../src/optim/).
   For an ordinary first-order method, implement the normal PyTorch optimizer
   contract, including `step`, `zero_grad`, `state_dict`, and `load_state_dict`.
2. **CLI:** add the ID in `register_optimizer_choice_args` in
   [`src/config/base.py`](../src/config/base.py), and register its own active
   hyperparameters using the existing argument-group style.
3. **Construction and schedule:** import and assemble it in `build_optimizer`
   in [`src/main.py`](../src/main.py). Reuse the actual model parameter groups;
   keep tied weights unique and cover all trainable parameters. Inspect
   `build_scheduler` if the method has internal scheduling or split updater
   groups. A Muon-style matrix updater also needs its embedding/vector fallback
   and both learning rates represented explicitly.
4. **Run identity:** add a branch in `build_optimization_plan` in
   [`src/run_manifest.py`](../src/run_manifest.py). Describe the actual updater,
   active hyperparameters, scheduler, and parameter routing. Unknown IDs fail
   before training, so changing only the CLI and constructor is insufficient.
   If introducing new argument names, inspect `OPTIMIZATION_CONFIG_KEYS` as well,
   so inactive defaults do not contaminate other optimizers' run identities.
5. **Training semantics, only if needed:** ordinary methods use the standard
   update path in [`src/optim/base.py`](../src/optim/base.py). Methods requiring
   closures, extra forward/backward passes, custom gradient synchronization, or
   train/eval transitions need explicit integration and tests at those call
   sites. Do not assume a successful single-device step proves DDP correctness.
6. **Recipe and inventory:** start from the closest `scripts/1b/*.sh`, retain
   the model/data/global-batch contract, and add the recipe to
   [`scripts/script_manifest.json`](../scripts/script_manifest.json), including
   command metadata. Update the inventory expectations in
   [`test_script_args.py`](../tests/behavior/test_script_args.py) and
   [`test_script_generator.py`](../tests/behavior/test_script_generator.py).
   The script generator produces previews; it does not replace the authoritative
   shell recipes automatically.
7. **Validate the integration:** test CLI parsing, real parameter routing,
   scheduler ownership, a small-model forward/backward/update, finite outputs,
   and save/resume of optimizer state. Follow the patterns in
   [`test_optimizer_assembly.py`](../tests/behavior/test_optimizer_assembly.py)
   and [`test_run_manifest.py`](../tests/behavior/test_run_manifest.py). When
   claiming distributed support, also validate gradient/state agreement on
   multiple ranks. Keep full 1B training separate from these small checks.

Compare final validation cross-entropy/perplexity at matched training tokens and
the same evaluation protocol. Keep the source commit, environment, dataset
identity, seed, full config, resolved optimization plan, token counts, and
failure status in the run artifacts. Report wall time/throughput with the actual
execution configuration. Use [the run launcher guide](running-benchmark.md) for
artifact layout and checkpoint behavior.

## What this addition verifies

The configuration can be parsed by the existing CLI, the script inventory can
round-trip through the preview generator, and the real model can be constructed
on the PyTorch `meta` device to check shapes, tied weights, and exact parameter
count without allocating 1B weights. Those checks do not establish CUDA memory
fit, 1B convergence, or a ranking of optimizers; those are outcomes of the
collaborators' subsequent experiments.
