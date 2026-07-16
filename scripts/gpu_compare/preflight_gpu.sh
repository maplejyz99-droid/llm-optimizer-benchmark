#!/bin/bash
set -euo pipefail

GPU_LABEL="${GPU_LABEL:-5090}"
REPO_DIR="${REPO_DIR:-/root/work/llm-optimizer-benchmark}"
DATASETS_DIR="${DATASETS_DIR:-/root/autodl-tmp/llmopt/datasets/fineweb-30B}"
CONDA_SH="${CONDA_SH:-/root/miniconda3/etc/profile.d/conda.sh}"
CONDA_ENV="${CONDA_ENV:-llmopt310}"
EXPECTED_GPU_COUNT="${EXPECTED_GPU_COUNT:-2}"
MIN_GPU_MEMORY_MIB="${MIN_GPU_MEMORY_MIB:-76000}"

cd "${REPO_DIR}"
if [[ -f "${CONDA_SH}" ]]; then
    source "${CONDA_SH}"
    conda activate "${CONDA_ENV}"
fi

export LLMOPT_FINEWEB_NO_DOWNLOAD=1
python scripts/gpu_compare/check_fineweb.py --datasets-dir "${DATASETS_DIR}"

echo
echo "== nvidia-smi =="
nvidia-smi

echo
echo "== CUDA/NCCL =="
python - <<INNER_PY
import torch
expected = int("${EXPECTED_GPU_COUNT}")
min_mem = int("${MIN_GPU_MEMORY_MIB}")
print("torch", torch.__version__)
print("cuda_available", torch.cuda.is_available())
print("cuda_device_count", torch.cuda.device_count())
if not torch.cuda.is_available():
    raise SystemExit("CUDA is not available")
if torch.cuda.device_count() != expected:
    raise SystemExit(f"Expected {expected} CUDA devices, got {torch.cuda.device_count()}")
for idx in range(torch.cuda.device_count()):
    props = torch.cuda.get_device_properties(idx)
    mem_mib = props.total_memory // 1024 // 1024
    print(idx, props.name, mem_mib, "MiB")
    if "A800" not in props.name:
        raise SystemExit(f"GPU {idx} is not A800: {props.name}")
    if mem_mib < min_mem:
        raise SystemExit(f"GPU {idx} memory below {min_mem} MiB: {mem_mib}")
print("nccl_available", torch.distributed.is_nccl_available())
if not torch.distributed.is_nccl_available():
    raise SystemExit("NCCL is not available")
INNER_PY

echo
echo "GPU preflight OK."
