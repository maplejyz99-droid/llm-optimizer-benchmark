#!/bin/bash
set -euo pipefail

GPU_LABEL="${GPU_LABEL:-5090}"
REPO_DIR="${REPO_DIR:-/root/work/llm-optimizer-benchmark}"
DATASETS_DIR="${DATASETS_DIR:-/root/autodl-tmp/llmopt/datasets/fineweb-30B}"
CONDA_SH="${CONDA_SH:-/root/miniconda3/etc/profile.d/conda.sh}"
CONDA_ENV="${CONDA_ENV:-llmopt310}"

cd "${REPO_DIR}"
if [[ -f "${CONDA_SH}" ]]; then
    source "${CONDA_SH}"
    conda activate "${CONDA_ENV}"
fi

export LLMOPT_FINEWEB_NO_DOWNLOAD=1

echo "== Host =="
hostname
uname -a

echo
echo "== Repo =="
git rev-parse --abbrev-ref HEAD
git rev-parse --short HEAD
git status --short

echo
echo "== Python/PyTorch =="
python - <<'INNER_PY'
import sys
import torch
print("python", sys.version.split()[0])
print("torch", torch.__version__)
print("cuda_available", torch.cuda.is_available())
print("cuda_device_count", torch.cuda.device_count())
INNER_PY

echo
echo "== FineWeb =="
python scripts/gpu_compare/check_fineweb.py --datasets-dir "${DATASETS_DIR}"

echo
echo "No-card preflight OK. CUDA availability is not required in this phase."
