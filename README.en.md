# LLM Optimizer Benchmark

**A research platform for optimizers and reproducible experiments in LLM pretraining.**

[简体中文](README.md) · **English**

[Experiment recipes](#experiment-recipes) · [Quick start](#quick-start) · [Runs and results](#runs-and-results) · [Documentation and development](#documentation-and-development)

We maintain optimizer baselines, research variants, and experiment recipes in a shared Llama training framework. Our comparisons focus on validation loss, training stability, computational cost, and behavior across model scales. Current work covers the Muon family, Gauss–Newton methods, preconditioning and parameter-selection experiments, and dense Llama configurations from 124M to 1B.

The repository combines algorithm implementation with experiment management. Each run records its source revision, data identity, effective optimization configuration, parameter update routes, and evaluation protocol, so a curve can be traced back to the training process that produced it.

## Current focus

- **Optimizer research:** GN-Prox / GN-Full, Newton-Muon, SoftEq K=2000 Muon, and Magma variants of AdamW / Muon.
- **Comparisons across scales:** recipes for 124M, 210M, 720M, and 1B dense Llama, plus 520M MoE experiment entries.
- **Parameterization experiments:** standard and muP Llama paths, with parameter-group checks, coordinate checks, and learning-rate sweep tools.
- **Traceable runs:** FineWeb-30B preflight checks, a shared launcher, versioned run manifests, evaluation records, checkpoint state, and behavior tests.

An implementation, a recipe, and a completed training result provide different levels of evidence. Method-specific scope is described below; performance claims should be tied to the corresponding run artifacts.

## Optimizers and research paths

### Comparison baselines

The CLI includes AdamW, Muon, Distributed Muon, SOAP, SophiaG, Lion, AdEMAMix, ADOPT, MARS, Prodigy, Schedule-Free, Adafactor, LAMB, and other options. See [`src/config/base.py`](src/config/base.py) for the complete list and arguments, and [`src/main.py`](src/main.py) for construction and parameter grouping.

Implementations of the same named method may differ in parameter grouping, fallback updates, and learning-rate policy. Report the implementation actually used here; known differences are documented in the [optimizer implementation audit](docs/optimizer-implementation-audit.md).

### Research variants

- **GN-Prox / GN-Full:** inner updates constructed through linearization or a Gauss–Newton approximation, selected with `--opt gn-prox` / `--opt gn-full`. The training entry currently restricts these methods to one device. See the [GN-Full implementation note](docs/gn-full-onepager.md).
- **Newton-Muon:** activation-covariance right-preconditioning in the Muon path, selected with `--opt newton-muon`. Currently restricted to single-device dense `llama`. See the [Newton-Muon note](docs/newton-muon-onepager.md).
- **SoftEq K=2000 Muon:** selected with `--opt softeq-k2000-muon`, for dense `llama` / `mup_llama`. Maintained as an experimental variant; GPU and checkpoint-resume validation around the cutoff are required before formal comparisons. See the [SoftEq integration note](docs/softeq-k2000-muon-integration.md).
- **Magma:** parameter-selection interventions through `--opt adamw-magma` / `--opt muon-magma`. The training entry currently restricts these methods to one device.

## Experiment recipes

- [124M](scripts/124m/): baseline comparisons, Newton-Muon / SoftEq entries, and GPU-memory probes.
- [210M](scripts/210m/) and [720M](scripts/720m/): optimizer recipes for larger models.
- [1B](scripts/1b/): dense Llama recipes for 17 optimizer variants, matching the 720M recipe set.
- [520M MoE](scripts/moe-520m/): MoE experiment recipes; check each optimizer's model and distributed restrictions before use.
- [muP learning-rate sweeps](scripts/repro/mup_readme_lr_sweep/README.md): standard-parameterization / muP comparisons using this repository's Llama training stack.

The **1B comparison configuration** has 24 layers, hidden width 1792, 14 attention heads, and tied embeddings, for **1,026,086,656 trainable parameters**. Context length is 512, with a global batch of 1984 sequences and **1,015,808 tokens per update**.

The 1B recipes inherit 720M hyperparameters as starting points; their availability does not establish tuned 1B performance or an optimizer ranking. The scripts retain an editable 48,000-update default. See the [1B benchmark guide](docs/1b-benchmark.md) for the 8,000-, 16,000-, and 20,203-update comparison conventions, training budgets, and collaborator integration requirements.

## Quick start

### 1. Install

Training primarily targets Python 3.10 on Linux / CUDA. CPU execution is useful for small-model functional checks.

```bash
git clone https://github.com/maplejyz99-droid/llm-optimizer-benchmark.git
cd llm-optimizer-benchmark

conda create -n llmopt python=3.10 -y
conda activate llmopt
python -m pip install -r requirements.txt
```

[`requirements.txt`](requirements.txt) is a base dependency list, not a complete CUDA environment lock. Select a PyTorch / CUDA build appropriate for the machine. [`requirements-ci.lock`](requirements-ci.lock) is specifically for Linux x86_64 / Python 3.10 CPU behavior tests. See [reproducibility](docs/reproducibility.md) for environment records and their limits.

### 2. Run a small model

This example performs 2 AdamW updates on CPU to exercise data loading, model construction, training, and evaluation. The first use downloads Tiny Shakespeare; W&B is not required.

```bash
python ./src/main.py \
  --config_format base --model llama --opt adamw \
  --dataset shakespeare-char --device cpu --dtype float32 \
  --n_layer 1 --n_head 2 --n_embd 64 --vocab_size 96 \
  --sequence_length 32 --batch_size 2 --acc_steps 1 \
  --iterations 2 --warmup_steps 1 --scheduler none \
  --eval_interval 2 --eval_batches 1 --final_eval_batches 1 \
  --latest_ckpt_interval 0 --permanent_ckpt_interval 0 \
  --results_base_folder ./exps/readme-smoke \
  --experiment_name adamw-cpu-smoke
```

Use a new `--experiment_name` for each repeat. This is a functional check; formal performance comparisons require a shared CUDA environment and experiment protocol.

### 3. Prepare comparison data

Current FineWeb comparisons use a versioned **FineWeb-30B** artifact: GPT-2 tokenization, `uint16` token files, 30B training tokens, and 100M validation tokens. The data directory must contain `train.bin`, `val.bin`, and `meta.json`.

```bash
export LLMOPT_DATASETS_DIR=/path/to/fineweb-30B
python ./scripts/data/check_fineweb_30b.py --json
```

Replace the example path with the actual data directory, or a parent containing `fineweb-30B/`. The training argument `--datasets_dir` takes precedence over the environment variable. Preflight checks metadata and file sizes without scanning every token; training does not download missing FineWeb data by default. To build an artifact, inspect the arguments in [`scripts/data/build_fineweb_30b.py`](scripts/data/build_fineweb_30b.py) and prepare it separately.

## Runs and results

Use the launcher for new experiments, keeping raw run artifacts and curated results in separate locations:

```bash
python ./scripts/setup_runtime_paths.py \
  --runs-root /path/to/large-disk/llmopt/runs \
  --results-root /path/to/curated/llmopt-results
```

This creates Git-ignored local `runs/` and `results/` links. After data preflight, preview a 20-step CUDA functional check with 124M AdamW:

```bash
python ./scripts/launch_run.py \
  --runs-root /path/to/large-disk/llmopt/runs \
  --suite smoke/fineweb30b \
  --run-id adamw-124m-20steps-seed0 \
  --gpu-ids 0 --monitor-gpu --dry-run \
  -- \
  --config_format base --model llama --opt adamw \
  --dataset fineweb --device cuda:0 --dtype bfloat16 \
  --n_layer 12 --n_head 12 --n_embd 768 \
  --sequence_length 512 --batch_size 1 --acc_steps 1 \
  --iterations 20 --warmup_steps 2 --scheduler cos \
  --lr 1e-3 --weight_decay 0.1 --seed 0 --data_seed 1337 \
  --eval_interval 10 --eval_batches 1 --final_eval_batches 1
```

`--dry-run` prints the command without starting training or creating a run directory. Remove it after checking paths, the environment, and GPU memory. The batch, horizon, and evaluation caps here are only for a functional check; use matched [experiment recipes](#experiment-recipes) and evaluation protocols for formal comparisons.

Use a new `--run-id` for every attempt. The launcher owns `--results_base_folder` and `--experiment_name`; do not pass these again after the `--` separator.

Run artifacts include:

- `run_manifest.json`: configuration, source and data identity, effective optimization plan, and realized parameter update routes.
- `summary.json` and `evaluations/`: training metrics, validation metrics, and versioned evaluation records.
- `launch/` and `logs/`: launch command, lifecycle state, terminal output, and optional GPU sampling.
- `ckpts/`: training state when checkpoint intervals are enabled.

The launcher checks consistency between the manifest and result files before marking a run complete. See the [running guide](docs/running-benchmark.md) for background execution, multiple GPUs, output layout, and resume restrictions.

## Comparison conventions

- Match the data version, model configuration, training-token budget, seed set, and evaluation protocol. Report the actual execution configuration and computational cost.
- In this repository's CLI, the product of `--batch_size` and `--acc_steps` is already the global sequence batch. DDP distributes it across processes; **do not multiply it by the GPU count again**. The product must be divisible by world size.
- GN, Newton-Muon, and Magma currently use the single-device entry. When using the launcher, omit `--nproc-per-node` and `--distributed_backend` for these methods. A recipe using `torchrun` is not, by itself, evidence that its distributed semantics have been fully validated.
- Schedule-Free methods use `--scheduler none`. Record the matrix updates, weight decay, and AdamW fallback policy separately for each Muon variant.
- Existing shell recipes contain W&B placeholders. Fill them in or remove the corresponding options before execution. These scripts do not forward extra positional arguments: edit the command to change its horizon instead of appending options to `bash scripts/1b/adamw.sh`.
- Resume must satisfy data, configuration, run-identity, and checkpoint-state contracts. The launcher creates a new run each time; see [reproducibility](docs/reproducibility.md) for resume through the direct training entry.

## Documentation and development

- [1B benchmark guide](docs/1b-benchmark.md): model configuration, training budgets, launch examples, and the optimizer integration checklist.
- [Running guide](docs/running-benchmark.md): launcher, logs, output directories, GPU monitoring, and checkpoint boundaries.
- [Reproducibility](docs/reproducibility.md): environment records, CPU dependency locking, run identity, resume, and CI.
- [Optimizer implementation audit](docs/optimizer-implementation-audit.md): local implementations, declared sources, known differences, and validation scope.
- [muP learning-rate sweeps](scripts/repro/mup_readme_lr_sweep/README.md): parameterization comparisons and plotting workflows.

Implementations primarily live in [`src/models/`](src/models/), [`src/optim/`](src/optim/), [`src/config/`](src/config/), and [`src/main.py`](src/main.py); [`src/run_manifest.py`](src/run_manifest.py) manages run identity. Adding an optimizer also requires updates to the CLI, construction logic, effective optimization plan, recipes, and behavior tests. Follow the checklist in the 1B benchmark guide.

In a **Linux CPU test environment** prepared according to the reproducibility guide, run:

```bash
python repro/verify_cpu_environment.py
python repro/run_behavior_tests.py full
python repro/run_behavior_tests.py isolated
python repro/run_behavior_tests.py reverse
```

These modes exercise the suite together, with isolated modules, and in reverse order. See [`tests/integration/`](tests/integration/) and [GitHub Actions](https://github.com/maplejyz99-droid/llm-optimizer-benchmark/actions) for distributed integration tests and CI. Passing CPU behavior tests does not establish CUDA performance or convergence.

Submit and discuss code or documentation improvements through [this repository's Pull Requests](https://github.com/maplejyz99-droid/llm-optimizer-benchmark/pulls).

## Acknowledgments, license, and citation

This project builds on [EPFL's llm-optimizer-benchmark](https://github.com/epfml/llm-optimizer-benchmark), retaining its training framework and baseline implementations while maintaining this repository's research variants, experiment configurations, and reproducibility tools. We thank the upstream authors and the contributors to [llm-baselines](https://github.com/epfml/llm-baselines) and [nanoGPT](https://github.com/karpathy/nanoGPT).

The project uses the [Apache-2.0 license](LICENSE) and retains provenance notes for upstream and third-party implementations. Cite the original paper when using its benchmark design or results. When using additions from this repository, also record the repository URL and exact commit.

Upstream paper: [Benchmarking Optimizers for Large Language Model Pretraining](https://arxiv.org/abs/2509.01440), Andrei Semenov, Matteo Pagliardini, and Martin Jaggi, 2025.

```bibtex
@article{semenov2025benchmarking,
  title={Benchmarking {O}ptimizers for {L}arge {L}anguage {M}odel {P}retraining},
  author={Semenov, Andrei and Pagliardini, Matteo and Jaggi, Martin},
  journal={arXiv preprint arXiv:2509.01440},
  url={https://arxiv.org/abs/2509.01440},
  year={2025}
}
```
