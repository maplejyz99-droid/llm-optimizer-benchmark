#!/bin/bash

set -euo pipefail

GPU_LABEL="${GPU_LABEL:-5090}"
REPO_DIR="${REPO_DIR:-$(pwd)}"
DATASETS_DIR="${DATASETS_DIR:-/root/autodl-tmp/llmopt/datasets/fineweb-30B}"
RESULTS_BASE_FOLDER="${RESULTS_BASE_FOLDER:-/root/autodl-tmp/llmopt/exps/5090-comparison-20260701}"
LOG_DIR="${LOG_DIR:-/root/work/llmopt-results/5090-comparison-20260701/logs}"
GPU_IDS="${GPU_IDS:-0}"
SEED="${SEED:-0}"
MODEL_PRESET="${MODEL_PRESET:-124m}"
SEQUENCE_LENGTH="${SEQUENCE_LENGTH:-1024}"
TARGET_EFF_BATCH="${TARGET_EFF_BATCH:-512}"
MICRO_BATCH="${MICRO_BATCH:-32}"
MICRO_BATCH_LIST="${MICRO_BATCH_LIST:-${MICRO_BATCH} 24 16 8}"
ITERATIONS="${ITERATIONS:-20}"
WARMUP_STEPS="${WARMUP_STEPS:-2}"
LOG_INTERVAL="${LOG_INTERVAL:-5}"
FINAL_EVAL_BATCHES="${FINAL_EVAL_BATCHES:-1}"
OPTS="${OPTS:-softeq-k2000-muon muon sophiag newton-muon}"
MUON_LR_FACTOR="${MUON_LR_FACTOR:-1e-2}"
ENABLE_WANDB="${ENABLE_WANDB:-0}"
WANDB_PROJECT="${WANDB_PROJECT:-}"
WANDB_ENTITY="${WANDB_ENTITY:-}"
SKIP_EXISTING="${SKIP_EXISTING:-1}"
CONDA_SH="${CONDA_SH:-/root/miniconda3/etc/profile.d/conda.sh}"
CONDA_ENV="${CONDA_ENV:-llmopt310}"

if [[ -z "${DATASETS_DIR}" ]]; then
    echo "ERROR: DATASETS_DIR is required and must point to fineweb-30B, or a parent containing fineweb-30B/fineweb-100BT." >&2
    exit 2
fi
if (( WARMUP_STEPS >= ITERATIONS )); then
    echo "ERROR: WARMUP_STEPS must be smaller than ITERATIONS." >&2
    exit 2
fi

cd "${REPO_DIR}"
if [[ -f "${CONDA_SH}" ]]; then
    source "${CONDA_SH}"
    conda activate "${CONDA_ENV}"
fi

python scripts/gpu_compare/check_fineweb.py --datasets-dir "${DATASETS_DIR}"
mkdir -p "${LOG_DIR}" "${RESULTS_BASE_FOLDER}"
export CUDA_VISIBLE_DEVICES="${GPU_IDS}"
export LLMOPT_FINEWEB_NO_DOWNLOAD=1

case "${MODEL_PRESET}" in
    124m) MODEL_ARGS=(--n_embd 768 --n_head 12 --n_layer 12) ;;
    210m) MODEL_ARGS=(--n_embd 768 --n_head 12 --n_layer 24) ;;
    720m) MODEL_ARGS=(--n_embd 2048 --n_head 16 --n_layer 12) ;;
    *) echo "ERROR: unsupported MODEL_PRESET '${MODEL_PRESET}'." >&2; exit 2 ;;
esac

optimizer_args() {
    case "$1" in
        softeq-k2000-muon)
            echo "--opt softeq-k2000-muon --lr 6e-4 --muon_lr_factor ${MUON_LR_FACTOR} --weight_decay 0.1 --scheduler wsd --wsd_fract_decay 0.0 --beta1 0.9 --beta2 0.95 --momentum 0.95"
            ;;
        muon)
            echo "--opt muon --lr 6e-4 --muon_lr_factor ${MUON_LR_FACTOR} --weight_decay 0.1 --scheduler wsd --wsd_fract_decay 0.0 --beta1 0.9 --beta2 0.95 --momentum 0.95 --nesterov True --muon_ns_steps 5"
            ;;
        sophiag)
            echo "--opt sophiag --lr 6e-4 --weight_decay 0.1 --scheduler wsd --wsd_fract_decay 0.0 --beta1 0.9 --beta2 0.95 --sophia_bs ${TARGET_EFF_BATCH}"
            ;;
        newton-muon)
            echo "--opt newton-muon --lr 6e-4 --muon_lr_factor ${MUON_LR_FACTOR} --weight_decay 0.1 --scheduler wsd --wsd_fract_decay 0.0 --beta1 0.9 --beta2 0.95 --momentum 0.95 --nesterov True --muon_ns_steps 5 --newton_muon_precond_every 32 --newton_muon_precond_ewma 0.95 --newton_muon_precond_init_diag 1e-3 --newton_muon_precond_ridge_mult 0.2 --newton_muon_precond_eps 1e-8"
            ;;
        *) echo "ERROR: unknown optimizer '$1'." >&2; return 2 ;;
    esac
}

run_one() {
    local opt="$1"
    local microbs="$2"
    local acc_steps="$3"
    local suffix="$4"
    local run_name log_path
    run_name="gpu-${GPU_LABEL}_track3-probe_model-${MODEL_PRESET}_seq-${SEQUENCE_LENGTH}_effbs-${TARGET_EFF_BATCH}_microbs-${microbs}_acc-${acc_steps}_world-1_opt-${opt}_seed-${SEED}_steps-${ITERATIONS}${suffix}"
    log_path="${LOG_DIR}/${run_name}.log"
    if [[ "${SKIP_EXISTING}" == "1" && -f "${log_path}" ]]; then
        if grep -q "# returncode: 0" "${log_path}"; then
            echo "[skip] ${run_name}"
            return 0
        fi
        echo "[rerun-failed] ${run_name}"
    fi
    read -r -a opt_argv <<< "$(optimizer_args "${opt}")"
    cmd=(
        python ./src/main.py
        --config_format base
        --model llama
        "${MODEL_ARGS[@]}"
        --batch_size "${microbs}"
        --sequence_length "${SEQUENCE_LENGTH}"
        --acc_steps "${acc_steps}"
        --dataset fineweb
        --datasets_dir "${DATASETS_DIR}"
        --results_base_folder "${RESULTS_BASE_FOLDER}"
        --experiment_name "${run_name}"
        --dropout 0.0
        --warmup_steps "${WARMUP_STEPS}"
        --grad_clip 1.0
        --seed "${SEED}"
        --dtype bfloat16
        --iterations "${ITERATIONS}"
        --eval_interval 1000000
        --eval_batches 1
        --final_eval_batches "${FINAL_EVAL_BATCHES}"
        --latest_ckpt_interval 0
        --permanent_ckpt_interval 0
        --log_interval "${LOG_INTERVAL}"
        "${opt_argv[@]}"
    )
    if [[ "${ENABLE_WANDB}" == "1" ]]; then
        cmd+=(--wandb --wandb_project "${WANDB_PROJECT}" --wandb_entity "${WANDB_ENTITY}")
    fi
    echo "[run] ${run_name}"
    set +e
    RUN_NAME="${run_name}" LOG_DIR="${LOG_DIR}" GPU_IDS="${GPU_IDS}"         bash scripts/gpu_compare/run_with_smi.sh -- "${cmd[@]}"
    local rc=$?
    if [[ "${rc}" -ne 0 ]]; then
        if grep -Eiq "out of memory|CUDA out of memory|OutOfMemoryError" "${log_path}"; then
            echo "[oom] ${run_name}"
        else
            echo "[failed] ${run_name}; rc=${rc}" >&2
        fi
    fi
    return "${rc}"
}

for opt in ${OPTS}; do
    if ! optimizer_args "${opt}" >/dev/null; then
        echo "[unsupported] ${opt}; skipping" >&2
        continue
    fi
    first=1
    for candidate_micro in ${MICRO_BATCH_LIST}; do
        if (( TARGET_EFF_BATCH % candidate_micro != 0 )); then
            echo "[skip] microbs=${candidate_micro} does not divide effbs=${TARGET_EFF_BATCH}"
            continue
        fi
        acc_steps=$((TARGET_EFF_BATCH / candidate_micro))
        suffix=""
        if [[ "${first}" != "1" ]]; then
            suffix="_fallback-effbs-${TARGET_EFF_BATCH}"
        fi
        set +e
        run_one "${opt}" "${candidate_micro}" "${acc_steps}" "${suffix}"
        rc=$?
        set -e
        if [[ "${rc}" -eq 0 ]]; then
            break
        fi
        first=0
        echo "[fallback] ${opt}: trying next Track3 micro batch candidate after rc=${rc}"
    done
done
