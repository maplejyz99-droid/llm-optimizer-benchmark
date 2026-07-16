#!/bin/bash

set -euo pipefail

GPU_LABEL="${GPU_LABEL:-5090}"
usage() {
    cat >&2 <<'EOF'
Usage:
  RUN_NAME=name [LOG_DIR=/root/work/llmopt-results/5090-comparison-20260701/logs] [GPU_IDS=0] bash scripts/gpu_compare/run_with_smi.sh -- <command...>

Runs a training command while sampling nvidia-smi into LOG_DIR/name_smi.csv.
The command stdout/stderr goes to LOG_DIR/name.log.
EOF
}

if [[ $# -lt 2 || "${1:-}" != "--" ]]; then
    usage
    exit 2
fi
shift

if [[ -z "${RUN_NAME:-}" ]]; then
    echo "ERROR: RUN_NAME is required." >&2
    exit 2
fi

if ! command -v nvidia-smi >/dev/null 2>&1; then
    echo "ERROR: nvidia-smi not found." >&2
    exit 127
fi

LOG_DIR="${LOG_DIR:-/root/work/llmopt-results/5090-comparison-20260701/logs}"
GPU_IDS="${GPU_IDS:-${CUDA_VISIBLE_DEVICES:-0}}"
SAMPLE_INTERVAL_SEC="${SAMPLE_INTERVAL_SEC:-2}"

mkdir -p "${LOG_DIR}"
TRAIN_LOG="${LOG_DIR}/${RUN_NAME}.log"
SMI_LOG="${LOG_DIR}/${RUN_NAME}_smi.csv"
CMD_LOG="${LOG_DIR}/${RUN_NAME}.cmd.txt"

monitor_smi() {
    echo "timestamp,gpu_index,memory_used_mib,utilization_gpu_pct,power_draw_w,temperature_c" > "${SMI_LOG}"
    while true; do
        nvidia-smi --id="${GPU_IDS}" \
            --query-gpu=timestamp,index,memory.used,utilization.gpu,power.draw,temperature.gpu \
            --format=csv,noheader,nounits 2>/dev/null \
            | awk -F', *' '{print $1 "," $2 "," $3 "," $4 "," $5 "," $6}' >> "${SMI_LOG}" || true
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

{
    echo "run_name=${RUN_NAME}"
    echo "cwd=$(pwd)"
    echo "gpu_ids=${GPU_IDS}"
    echo "started_at=$(date -Iseconds)"
    printf 'command='
    printf '%q ' "$@"
    printf '\n'
} > "${CMD_LOG}"

monitor_smi &
SMI_PID=$!

set +e
"$@" > "${TRAIN_LOG}" 2>&1
RC=$?
set -e

{
    echo "# returncode: ${RC}"
    echo "# finished_at: $(date -Iseconds)"
} >> "${TRAIN_LOG}"

exit "${RC}"
