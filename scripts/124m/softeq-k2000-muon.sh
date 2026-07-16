#!/bin/bash

set -euo pipefail

REPO_DIR="${REPO_DIR:-/root/work/llm-optimizer-benchmark}"
DATASETS_DIR="${DATASETS_DIR:-/root/autodl-tmp/llmopt/datasets/fineweb-30B}"
RESULTS_BASE_FOLDER="${RESULTS_BASE_FOLDER:-/root/autodl-tmp/llmopt/exps/5090-comparison-20260701}"
EXPERIMENT_NAME="${EXPERIMENT_NAME:-124m-softeq-k2000-muon-fineweb-probe}"
ITERATIONS="${ITERATIONS:-20}"
WARMUP_STEPS="${WARMUP_STEPS:-2}"
BATCH_SIZE="${BATCH_SIZE:-64}"
ACC_STEPS="${ACC_STEPS:-4}"
ENABLE_WANDB="${ENABLE_WANDB:-0}"
WANDB_PROJECT="${WANDB_PROJECT:-}"
WANDB_ENTITY="${WANDB_ENTITY:-}"

cd "${REPO_DIR}"
export LLMOPT_FINEWEB_NO_DOWNLOAD=1
python scripts/gpu_compare/check_fineweb.py --datasets-dir "${DATASETS_DIR}"

cmd=(
    python ./src/main.py
    --config_format base
    --model llama
    --device cuda:0
    --n_embd 768 --n_head 12 --n_layer 12
    --batch_size "${BATCH_SIZE}"
    --sequence_length 512
    --acc_steps "${ACC_STEPS}"
    --dataset fineweb
    --datasets_dir "${DATASETS_DIR}"
    --results_base_folder "${RESULTS_BASE_FOLDER}"
    --experiment_name "${EXPERIMENT_NAME}"
    --dropout 0.0
    --warmup_steps "${WARMUP_STEPS}"
    --grad_clip 0.5
    --seed 0
    --dtype bfloat16
    --iterations "${ITERATIONS}"
    --eval_interval 1000000
    --eval_batches 1
    --final_eval_batches 1
    --latest_ckpt_interval 0
    --permanent_ckpt_interval 0
    --log_interval 5
    --opt softeq-k2000-muon
    --lr 1e-3
    --muon_lr_factor 1e-2
    --weight_decay 0.1
    --scheduler cos
    --beta1 0.8 --beta2 0.999
    --momentum 0.95
)

if [[ "${ENABLE_WANDB}" == "1" ]]; then
    cmd+=(--wandb --wandb_project "${WANDB_PROJECT}" --wandb_entity "${WANDB_ENTITY}")
fi

"${cmd[@]}"
