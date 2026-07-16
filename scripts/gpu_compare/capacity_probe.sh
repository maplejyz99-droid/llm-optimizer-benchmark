#!/bin/bash

set -euo pipefail

GPU_LABEL="${GPU_LABEL:-5090}"
REPO_DIR="${REPO_DIR:-$(pwd)}"
DATASETS_DIR="${DATASETS_DIR:-/root/autodl-tmp/llmopt/datasets/fineweb-30B}"
RESULTS_BASE_FOLDER="${RESULTS_BASE_FOLDER:-/root/autodl-tmp/llmopt/exps/5090-comparison-20260701}"
LOG_DIR="${LOG_DIR:-/root/work/llmopt-results/5090-comparison-20260701/logs}"
GPU_IDS="${GPU_IDS:-0}"
SEED="${SEED:-0}"
START_BATCH="${START_BATCH:-8}"
BATCH_STEP="${BATCH_STEP:-8}"
BATCH_LIST="${BATCH_LIST:-}"
PROBE_MODE="${PROBE_MODE:-linear}"
MAX_BATCH="${MAX_BATCH:-256}"
ITERATIONS="${ITERATIONS:-50}"
WARMUP_STEPS="${WARMUP_STEPS:-10}"
LOG_INTERVAL="${LOG_INTERVAL:-10}"
FINAL_EVAL_BATCHES="${FINAL_EVAL_BATCHES:-1}"
SEQUENCE_LENGTHS="${SEQUENCE_LENGTHS:-512 1024}"
MODEL_PRESETS="${MODEL_PRESETS:-124m 210m 720m}"
OPTS="${OPTS:-adamw softeq-k2000-muon}"
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

cd "${REPO_DIR}"
if [[ -f "${CONDA_SH}" ]]; then
    source "${CONDA_SH}"
    conda activate "${CONDA_ENV}"
fi

python scripts/gpu_compare/check_fineweb.py --datasets-dir "${DATASETS_DIR}"
mkdir -p "${LOG_DIR}" "${RESULTS_BASE_FOLDER}"
export CUDA_VISIBLE_DEVICES="${GPU_IDS}"
export LLMOPT_FINEWEB_NO_DOWNLOAD=1

model_args() {
    case "$1" in
        124m) echo "--n_embd 768 --n_head 12 --n_layer 12" ;;
        210m) echo "--n_embd 768 --n_head 12 --n_layer 24" ;;
        720m) echo "--n_embd 2048 --n_head 16 --n_layer 12" ;;
        1p3b) echo "--n_embd 2048 --n_head 16 --n_layer 24" ;;
        1p5b) echo "--n_embd 2048 --n_head 16 --n_layer 28" ;;
        2b) echo "--n_embd 2560 --n_head 20 --n_layer 24" ;;
        *)
            echo "ERROR: unknown MODEL_PRESET '$1'" >&2
            return 2
            ;;
    esac
}

optimizer_args() {
    case "$1" in
        adamw)
            echo "--opt adamw --lr 1e-3 --weight_decay 0.1 --scheduler cos --beta1 0.8 --beta2 0.999"
            ;;
        muon)
            echo "--opt muon --lr 1e-3 --muon_lr_factor 1e-2 --weight_decay 0.1 --scheduler cos --beta1 0.8 --beta2 0.999 --momentum 0.95 --nesterov True --muon_ns_steps 5"
            ;;
        softeq-k2000-muon)
            echo "--opt softeq-k2000-muon --lr 1e-3 --muon_lr_factor 1e-2 --weight_decay 0.1 --scheduler cos --beta1 0.8 --beta2 0.999 --momentum 0.95"
            ;;
        sophiag)
            echo "--opt sophiag --lr 1e-3 --weight_decay 0.1 --scheduler cos --beta1 0.9 --beta2 0.999"
            ;;
        newton-muon)
            echo "--opt newton-muon --lr 1e-3 --muon_lr_factor 1e-2 --weight_decay 0.1 --scheduler cos --beta1 0.8 --beta2 0.999 --momentum 0.95 --nesterov True --muon_ns_steps 5 --newton_muon_precond_every 32 --newton_muon_precond_ewma 0.95 --newton_muon_precond_init_diag 1e-3 --newton_muon_precond_ridge_mult 0.2 --newton_muon_precond_eps 1e-8"
            ;;
        *)
            echo "ERROR: unknown optimizer '$1'" >&2
            return 2
            ;;
    esac
}

capacity_batches() {
    local model="$1"
    local seq="$2"
    if [[ -n "${BATCH_LIST}" ]]; then
        echo "${BATCH_LIST}"
        return 0
    fi
    if [[ "${PROBE_MODE}" == "adaptive" ]]; then
        case "${model}:${seq}" in
            124m:512) echo "48 40 32 24 16 8" ;;
            124m:1024) echo "32 24 16 8" ;;
            210m:512) echo "40 32 24 16 8" ;;
            210m:1024) echo "24 16 8" ;;
            720m:512) echo "32 24 16 8" ;;
            720m:1024) echo "16 8" ;;
            *) echo "32 24 16 8" ;;
        esac
        return 0
    fi
    local batches=()
    local batch="${START_BATCH}"
    while [[ "${batch}" -le "${MAX_BATCH}" ]]; do
        batches+=("${batch}")
        batch=$((batch + BATCH_STEP))
    done
    echo "${batches[*]}"
}

for model in ${MODEL_PRESETS}; do
    for seq in ${SEQUENCE_LENGTHS}; do
        for opt in ${OPTS}; do
            for batch in $(capacity_batches "${model}" "${seq}"); do
                run_name="gpu-${GPU_LABEL}_capacity_model-${model}_seq-${seq}_effbs-${batch}_microbs-${batch}_acc-1_world-1_opt-${opt}_seed-${SEED}"
                log_path="${LOG_DIR}/${run_name}.log"
                if [[ "${SKIP_EXISTING}" == "1" && -f "${log_path}" ]]; then
                    echo "[skip] ${run_name}"
                    continue
                fi

                read -r -a model_argv <<< "$(model_args "${model}")"
                read -r -a opt_argv <<< "$(optimizer_args "${opt}")"
                cmd=(
                    python ./src/main.py
                    --config_format base
                    --model llama
                    --device cuda:0
                    "${model_argv[@]}"
                    --batch_size "${batch}"
                    --sequence_length "${seq}"
                    --acc_steps 1
                    --dataset fineweb
                    --datasets_dir "${DATASETS_DIR}"
                    --results_base_folder "${RESULTS_BASE_FOLDER}"
                    --experiment_name "${run_name}"
                    --dropout 0.0
                    --warmup_steps "${WARMUP_STEPS}"
                    --grad_clip 0.5
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
                RUN_NAME="${run_name}" LOG_DIR="${LOG_DIR}" GPU_IDS="${GPU_IDS}" \
                    bash scripts/gpu_compare/run_with_smi.sh -- "${cmd[@]}"
                rc=$?
                set -e

                if [[ "${rc}" -ne 0 ]]; then
                    if grep -Eiq "out of memory|CUDA out of memory|OutOfMemoryError" "${log_path}"; then
                        if [[ "${PROBE_MODE}" == "adaptive" ]]; then
                            echo "[oom] ${run_name}; adaptive mode will try the next smaller candidate"
                            continue
                        fi
                        echo "[oom] ${run_name}; stopping this model/seq/opt batch sweep"
                        break
                    fi
                    echo "[failed] ${run_name}; rc=${rc}; continuing to next batch" >&2
                fi

            done
        done
    done
done
