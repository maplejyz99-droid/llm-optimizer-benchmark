#!/bin/bash

set -euo pipefail

GPU_LABEL="${GPU_LABEL:-5090}"
REPO_DIR="${REPO_DIR:-$(pwd)}"
DATASETS_DIR="${DATASETS_DIR:-/root/autodl-tmp/llmopt/datasets/fineweb-30B}"
RESULTS_BASE_FOLDER="${RESULTS_BASE_FOLDER:-/root/autodl-tmp/llmopt/exps/5090-comparison-20260701}"
LOG_DIR="${LOG_DIR:-/root/work/llmopt-results/5090-comparison-20260701/logs}"
GPU_IDS="${GPU_IDS:-0}"
SEED="${SEED:-0}"
CASES="${CASES:-124m-small 124m-large 210m-main 720m-main}"
OPTS="${OPTS:-softeq-k2000-muon adamw muon sophiag newton-muon}"
ITERATIONS="${ITERATIONS:-20}"
WARMUP_STEPS="${WARMUP_STEPS:-2}"
ITERATIONS_720M="${ITERATIONS_720M:-20}"
WARMUP_STEPS_720M="${WARMUP_STEPS_720M:-2}"
LOG_INTERVAL="${LOG_INTERVAL:-5}"
FINAL_EVAL_BATCHES="${FINAL_EVAL_BATCHES:-1}"
ENABLE_FALLBACK="${ENABLE_FALLBACK:-1}"
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

case_model() {
    case "$1" in
        124m-small|124m-large) echo "124m" ;;
        210m-main) echo "210m" ;;
        720m-main) echo "720m" ;;
        *) echo "ERROR: unknown benchmark case '$1'" >&2; return 2 ;;
    esac
}

model_args() {
    case "$(case_model "$1")" in
        124m) echo "--n_embd 768 --n_head 12 --n_layer 12" ;;
        210m) echo "--n_embd 768 --n_head 12 --n_layer 24" ;;
        720m) echo "--n_embd 2048 --n_head 16 --n_layer 12" ;;
        *) return 2 ;;
    esac
}

grad_clip() {
    case "$(case_model "$1")" in
        720m) echo "0.1" ;;
        *) echo "0.5" ;;
    esac
}

case_iterations() {
    case "$(case_model "$1")" in
        720m) echo "${ITERATIONS_720M} ${WARMUP_STEPS_720M}" ;;
        *) echo "${ITERATIONS} ${WARMUP_STEPS}" ;;
    esac
}

micro_acc_exact() {
    local case_name="$1"
    local opt="$2"
    case "${case_name}" in
        124m-small) echo "32 1" ;;
        124m-large|210m-main)
            if [[ "${opt}" == "sophiag" ]]; then echo "32 8"; else echo "64 4"; fi
            ;;
        720m-main)
            if [[ "${opt}" == "sophiag" ]]; then echo "32 62"; else echo "62 32"; fi
            ;;
        *) echo "ERROR: unknown benchmark case '${case_name}'" >&2; return 2 ;;
    esac
}

micro_acc_fallback() {
    local case_name="$1"
    local opt="$2"
    case "${case_name}" in
        124m-small)
            if [[ "${opt}" == "sophiag" ]]; then echo "16 2"; else echo "32 1"; fi
            ;;
        124m-large|210m-main)
            if [[ "${opt}" == "sophiag" ]]; then echo "16 16"; else echo "32 8"; fi
            ;;
        720m-main)
            if [[ "${opt}" == "sophiag" ]]; then echo "8 248"; else echo "16 124"; fi
            ;;
        *) echo "ERROR: unknown benchmark case '${case_name}'" >&2; return 2 ;;
    esac
}

optimizer_args() {
    local model="$1"
    local opt="$2"
    case "${opt}" in
        adamw)
            if [[ "${model}" == "124m" ]]; then
                echo "--opt adamw --lr 1e-3 --weight_decay 0.1 --scheduler cos --beta1 0.8 --beta2 0.999"
            else
                echo "--opt adamw --lr 1e-3 --weight_decay 0.1 --scheduler cos --beta1 0.9 --beta2 0.999"
            fi
            ;;
        muon)
            echo "--opt muon --lr 1e-3 --muon_lr_factor 1e-2 --weight_decay 0.1 --scheduler cos --beta1 0.8 --beta2 0.999 --momentum 0.95 --nesterov True --muon_ns_steps 5"
            ;;
        softeq-k2000-muon)
            echo "--opt softeq-k2000-muon --lr 1e-3 --muon_lr_factor 1e-2 --weight_decay 0.1 --scheduler cos --beta1 0.8 --beta2 0.999 --momentum 0.95"
            ;;
        sophiag)
            if [[ "${model}" == "720m" ]]; then
                echo "--opt sophiag --lr 5e-4 --weight_decay 0.1 --scheduler cos --beta1 0.95 --beta2 0.999"
            else
                echo "--opt sophiag --lr 1e-3 --weight_decay 0.1 --scheduler cos --beta1 0.9 --beta2 0.999"
            fi
            ;;
        newton-muon)
            echo "--opt newton-muon --lr 1e-3 --muon_lr_factor 1e-2 --weight_decay 0.1 --scheduler cos --beta1 0.8 --beta2 0.999 --momentum 0.95 --nesterov True --muon_ns_steps 5 --newton_muon_precond_every 32 --newton_muon_precond_ewma 0.95 --newton_muon_precond_init_diag 1e-3 --newton_muon_precond_ridge_mult 0.2 --newton_muon_precond_eps 1e-8"
            ;;
        *) echo "ERROR: unknown optimizer '${opt}'" >&2; return 2 ;;
    esac
}

run_probe() {
    local case_name="$1"
    local opt="$2"
    local microbs="$3"
    local acc_steps="$4"
    local suffix="$5"
    local model seq run_iterations run_warmup effbs run_name log_path
    model="$(case_model "${case_name}")"
    seq="512"
    read -r run_iterations run_warmup <<< "$(case_iterations "${case_name}")"
    effbs=$((microbs * acc_steps))
    run_name="gpu-${GPU_LABEL}_benchmark-probe_case-${case_name}_model-${model}_seq-${seq}_effbs-${effbs}_microbs-${microbs}_acc-${acc_steps}_world-1_opt-${opt}_seed-${SEED}_steps-${run_iterations}${suffix}"
    log_path="${LOG_DIR}/${run_name}.log"
    if [[ "${SKIP_EXISTING}" == "1" && -f "${log_path}" ]]; then
        if grep -q "# returncode: 0" "${log_path}"; then
            echo "[skip] ${run_name}"
            return 0
        fi
        echo "[rerun-failed] ${run_name}"
    fi
    if (( run_warmup >= run_iterations || run_warmup < 2 || run_iterations - run_warmup < 2 )); then
        echo "ERROR: invalid warmup/iterations for ${case_name}: warmup=${run_warmup}, iterations=${run_iterations}" >&2
        return 2
    fi

    read -r -a model_argv <<< "$(model_args "${case_name}")"
    read -r -a opt_argv <<< "$(optimizer_args "${model}" "${opt}")"
    cmd=(
        python ./src/main.py
        --config_format base
        --model llama
        --device cuda:0
        "${model_argv[@]}"
        --batch_size "${microbs}"
        --sequence_length "${seq}"
        --acc_steps "${acc_steps}"
        --dataset fineweb
        --datasets_dir "${DATASETS_DIR}"
        --results_base_folder "${RESULTS_BASE_FOLDER}"
        --experiment_name "${run_name}"
        --dropout 0.0
        --warmup_steps "${run_warmup}"
        --grad_clip "$(grad_clip "${case_name}")"
        --seed "${SEED}"
        --dtype bfloat16
        --iterations "${run_iterations}"
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

for case_name in ${CASES}; do
    for opt in ${OPTS}; do
        read -r microbs acc_steps <<< "$(micro_acc_exact "${case_name}" "${opt}")"
        set +e
        run_probe "${case_name}" "${opt}" "${microbs}" "${acc_steps}" ""
        rc=$?
        set -e
        if [[ "${rc}" -ne 0 && "${ENABLE_FALLBACK}" == "1" ]]; then
            read -r fb_micro fb_acc <<< "$(micro_acc_fallback "${case_name}" "${opt}")"
            if [[ "${fb_micro}" != "${microbs}" || "${fb_acc}" != "${acc_steps}" ]]; then
                fb_eff=$((fb_micro * fb_acc))
                echo "[fallback] ${case_name}/${opt}: exact failed, trying microbs=${fb_micro}, acc=${fb_acc}, effbs=${fb_eff}"
                set +e
                run_probe "${case_name}" "${opt}" "${fb_micro}" "${fb_acc}" "_fallback-effbs-${fb_eff}"
                set -e
            fi
        fi
    done
done
