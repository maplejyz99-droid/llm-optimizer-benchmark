# Benchmark reproducibility

## Scope

This repository keeps the Benchmark CPU test environment separate from the CUDA
training environment and from the Track 3 runner. The CPU lock is for behavior
tests only; it is not a complete training dependency set.

The working Benchmark environment is currently located at
`/root/miniconda3/envs/llmopt310`. Treat it as read-only evidence. Do not upgrade,
remove, or repair packages in place when regenerating repository artifacts.

## Three different artifacts

- `repro/environments/benchmark-llmopt310-20260716.json` records the observed
  environment. It preserves duplicate package metadata and a failing `pip check`
  instead of hiding them.
- `requirements-ci.in` lists the direct dependencies imported by the CPU behavior
  suite.
- `requirements-ci.lock` is the fully resolved Linux x86_64 / CPython 3.10 lock.
  It must remain installable with `--require-hashes` and must contain CPU-only
  PyTorch.

An observed environment may be operational even when its package metadata is
inconsistent. For that reason, never replace the direct input or lock with a raw
`pip freeze`.

## Collect the Benchmark environment

Run the collector with the environment's own Python. The collector uses an
allowlist: it records package names and versions, platform metadata, `pip check`,
and `nvidia-smi`; it never reads environment variables, pip/conda configuration,
Git remotes, shell history, SSH files, W&B configuration, or package direct URLs.

```bash
/root/miniconda3/envs/llmopt310/bin/python \
  repro/collect_environment.py \
  --label benchmark-llmopt310 \
  --output repro/environments/benchmark-llmopt310-20260716.json
```

## Regenerate the CPU lock

Generate the lock only on Linux x86_64 with CPython 3.10.19. Use a disposable
tool environment; do not install the resolver into `llmopt310`. The wheel-first
process deliberately locks the one Linux/Python artifact selected for every
package instead of downloading and hashing candidates for unrelated platforms.

```bash
/root/miniconda3/envs/llmopt310/bin/python -m venv /tmp/llmopt-lock-tools
wheel_dir="$(mktemp -d /tmp/llmopt-ci-wheels.XXXXXX)"
PIP_CONFIG_FILE=/dev/null /tmp/llmopt-lock-tools/bin/python -m pip download \
  --disable-pip-version-check \
  --no-cache-dir \
  --only-binary=:all: \
  --dest "$wheel_dir" \
  -r requirements-ci.in
/tmp/llmopt-lock-tools/bin/python repro/build_ci_lock.py \
  --input requirements-ci.in \
  --wheel-dir "$wheel_dir" \
  --output requirements-ci.lock
```

Review the generated index URLs, versions, hashes, and target header before
replacing the lock. Regeneration must fail if the wheel set contains `triton` or
an `nvidia-*` distribution.

## Validate in a clean CPU environment

```bash
python3.10 -m venv /tmp/llmopt-ci-verify
/tmp/llmopt-ci-verify/bin/python -m pip install \
  --require-hashes \
  -r requirements-ci.lock
/tmp/llmopt-ci-verify/bin/python -m pip check
/tmp/llmopt-ci-verify/bin/python repro/verify_cpu_environment.py
/tmp/llmopt-ci-verify/bin/python repro/run_behavior_tests.py full
/tmp/llmopt-ci-verify/bin/python repro/run_behavior_tests.py isolated
/tmp/llmopt-ci-verify/bin/python repro/run_behavior_tests.py reverse
```

The runner treats unexpected skips as failures, starts every behavior module in a
fresh Python process for `isolated`, and reverses both module import order and
test execution order for `reverse`. It re-executes itself when needed so the hash
seed and bytecode settings apply from interpreter startup.

Before and after each mode, it fingerprints the tracked diff and normal-sized
untracked files. Large untracked files use metadata instead of content hashes so
the check never hashes datasets or checkpoints. In clean CI, any tracked or
untracked repository change is still an unconditional failure; the final workflow
step independently checks `git status`.

## CI boundary

GitHub Actions runs the three modes independently on a fixed Linux/Python image.
Adding the workflow file locally does not activate CI. A real CI acceptance check
requires an authorized commit and push or pull request; this repository workflow
does not run GPU training and does not install Track 3 dependencies.

## Run and data identity

Every training entry now writes `run_manifest.json` before model construction or
W&B initialization. Its `run_identity` covers the effective training config,
source commit plus local `src/`/`scripts/` changes, installed runtime versions,
and token artifact identity. Installation paths, results paths, notification
credentials, and other location-only controls are recorded or redacted but do
not change compatibility identity.

The compatibility identity includes the normalized compute device type: CPU,
CUDA, and MPS runs are distinct, while `cuda:0` and `cuda:1` remain compatible
for different DDP local ranks. Notification credentials are redacted from the
console, W&B config, `summary.json`, and generated experiment name.

Token files up to 64 MiB use a full SHA-256. Larger files use a deterministic
three-region sampled fingerprint so a 30B-token file is not read in full for
every run. The manifest records that method explicitly; a sampled fingerprint is
an artifact-identity guard, not a cryptographic proof that every byte is equal.

The inherited EPFL task-data behavior is identified as
`epfl-flat-eotpad-v1`: examples are padded with EOT, flattened, and trained as a
continuous next-token stream. EOT padding still contributes to loss and windows
may cross example boundaries. This compatibility behavior is documented rather
than silently changed; results from any future masked/boundary-aware format must
use a new semantics ID and data artifact version.

`summary.json` includes the run identity, metric semantics, and populated
training/validation metric histories. Existing output directories with a
different identity are rejected before model construction.

## Exact checkpoint resume

New checkpoints use a versioned schema and atomically replace `main.pt` and each
`worker_<rank>.pt`. They save the real training-reader position, iteration,
world size, run identity, RNG state, scheduler state, and active WA/EWA state.
Every save also has a unique `snapshot_id`; a worker file from a different save
is rejected before its optimizer or RNG state can be restored.
For owner-sharded multi-rank Muon and SoftEq optimizers, optimizer state
is stored per rank; ordinary optimizers remain in `main.pt` and are not duplicated
in worker files.

Uniform WA also depends on its completed horizon files under the run's `avgs/`
directory. Resume validates that history and fails clearly when it is missing;
moving a WA checkpoint alone (or using an ephemeral WA directory after completed
horizons) is therefore not advertised as exact resume. EWA has no external
history-file dependency.

Legacy checkpoints are rejected by default because they cannot prove exact
reader and averager state. A standard optimizer can explicitly request the old,
non-exact behavior with:

```bash
--allow_legacy_checkpoint_resume True
```

GN legacy resume remains rejected because `iteration * acc_steps` does not
reconstruct GN inner-loop and line-search data consumption. Benchmark GN is also
single-rank only until its manually assigned gradients have verified collective
synchronization semantics.

Evaluation and text generation preserve Python, NumPy, CPU Torch, and CUDA Torch
RNG state, so changing evaluation cadence does not change the subsequent training
trajectory. Flash attention disables dropout in eval mode. A bounded final/full
evaluation can be requested with:

```bash
--final_eval_batches 64
```

The cap applies consistently to the raw, uniform-weight-average, and exponential-
weight-average models. DDP synchronizes checkpoint/evaluation failures before
entering the next collective. Both `adamw-magma` and `muon-magma` are single-rank
until every stochastic Magma mask, including Muon's AdamW backup branch, is
synchronized across ranks.
