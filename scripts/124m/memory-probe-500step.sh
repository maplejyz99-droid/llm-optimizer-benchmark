#!/bin/bash

set -euo pipefail

if [[ $# -ne 1 ]]; then
    echo "Usage: bash scripts/124m/memory-probe-500step.sh {muon|newton-muon}" >&2
    exit 2
fi

OPT_NAME="$1"
case "${OPT_NAME}" in
    muon)
        RUN_PREFIX="muon_124m_1gpu_bs32_acc1_500step"
        ;;
    newton-muon)
        RUN_PREFIX="newton_muon_124m_1gpu_bs32_acc1_500step"
        ;;
    *)
        echo "Unsupported optimizer: ${OPT_NAME}. Expected muon or newton-muon." >&2
        exit 2
        ;;
esac

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd -P)"
DEFAULT_REPO_DIR="$(cd -- "${SCRIPT_DIR}/../.." && pwd -P)"
REPO_DIR="${REPO_DIR:-${DEFAULT_REPO_DIR}}"
CONDA_SH="${CONDA_SH:-/root/miniconda3/etc/profile.d/conda.sh}"
CONDA_ENV="${CONDA_ENV:-llmopt310}"
PHYSICAL_GPU="${PHYSICAL_GPU:-1}"
SAMPLE_INTERVAL_SEC="${SAMPLE_INTERVAL_SEC:-2}"
LOG_DIR="${LOG_DIR:-logs}"

TRAIN_LOG="${LOG_DIR}/${RUN_PREFIX}.log"
SMI_LOG="${LOG_DIR}/${RUN_PREFIX}_smi.csv"
PID_FILE="${LOG_DIR}/${RUN_PREFIX}.pid"
SMI_PID_FILE="${LOG_DIR}/${RUN_PREFIX}_smi.pid"

monitor_smi() {
    echo "timestamp,gpu_index,memory_used_mib,pid,process_name" > "${SMI_LOG}"
    while true; do
        timestamp="$(date -Iseconds)"
        rows="$(nvidia-smi --id="${PHYSICAL_GPU}" \
            --query-compute-apps=pid,process_name,used_memory \
            --format=csv,noheader,nounits 2>/dev/null || true)"
        if [[ -n "${rows}" ]]; then
            while IFS= read -r row; do
                pid="$(echo "${row}" | awk -F', *' '{print $1}')"
                process_name="$(echo "${row}" | awk -F', *' '{print $2}')"
                memory_used="$(echo "${row}" | awk -F', *' '{print $3}')"
                echo "${timestamp},${PHYSICAL_GPU},${memory_used},${pid},${process_name}" >> "${SMI_LOG}"
            done <<< "${rows}"
        else
            echo "${timestamp},${PHYSICAL_GPU},0,," >> "${SMI_LOG}"
        fi
        sleep "${SAMPLE_INTERVAL_SEC}"
    done
}

cleanup() {
    if [[ -n "${SMI_PID:-}" ]] && kill -0 "${SMI_PID}" 2>/dev/null; then
        kill "${SMI_PID}" 2>/dev/null || true
        wait "${SMI_PID}" 2>/dev/null || true
    fi
}
trap cleanup EXIT

source "${CONDA_SH}"
conda activate "${CONDA_ENV}"
cd "${REPO_DIR}"
mkdir -p "${LOG_DIR}"

export HF_HOME=/root/autodl-tmp/hf
export HF_DATASETS_CACHE=/root/autodl-tmp/hf/datasets
export TRANSFORMERS_CACHE=/root/autodl-tmp/hf/transformers
export HUGGINGFACE_HUB_CACHE=/root/autodl-tmp/hf/hub
export TMPDIR=/root/autodl-tmp/hf/tmp
export CUDA_VISIBLE_DEVICES="${PHYSICAL_GPU}"

monitor_smi &
SMI_PID=$!
echo "${SMI_PID}" > "${SMI_PID_FILE}"

COMMON_ARGS=(
    --config_format base
    --device cuda:0
    --model llama
    --n_layer 12 --n_head 12 --n_embd 768
    --sequence_length 512
    --dataset slimpajama
    --batch_size 32 --acc_steps 1
    --lr 1e-3
    --muon_lr_factor 1e-2
    --nesterov True
    --muon_ns_steps 5
    --warmup_steps 50
    --weight_decay 0.1
    --scheduler wsd
    --grad_clip 0.5
    --momentum 0.99
    --beta1 0.8 --beta2 0.999
    --dtype bfloat16
    --iterations 500
    --eval_interval 1000000
    --log_interval 10
    --run_prefix "${RUN_PREFIX}"
)

OPT_ARGS=(--opt "${OPT_NAME}")
if [[ "${OPT_NAME}" == "newton-muon" ]]; then
    OPT_ARGS+=(
        --newton_muon_precond_every 32
        --newton_muon_precond_ewma 0.95
        --newton_muon_precond_init_diag 1e-3
        --newton_muon_precond_ridge_mult 0.2
        --newton_muon_precond_eps 1e-8
    )
fi

python ./src/main.py "${COMMON_ARGS[@]}" "${OPT_ARGS[@]}" > "${TRAIN_LOG}" 2>&1 &
TRAIN_PID=$!
echo "${TRAIN_PID}" > "${PID_FILE}"
wait "${TRAIN_PID}"
