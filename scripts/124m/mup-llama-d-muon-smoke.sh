#!/bin/bash
set -euo pipefail

torchrun --standalone --nproc_per_node=1 ./src/main.py --config_format base --model mup_llama \
    --n_embd 128 --n_head 2 --n_layer 2 \
    --batch_size 2 --sequence_length 128 --acc_steps 1 \
    --dataset slimpajama --iterations 20 \
    --dropout 0.0 --warmup_steps 2 --grad_clip 0.5 --seed 0 \
    --opt d-muon --lr 1e-3 --weight_decay 0.1 --scheduler cos \
    --beta1 0.8 --beta2 0.999 --momentum 0.95 --nesterov True --muon_ns_steps 5 \
    --scale_base_model 256 --scale_emb 10 --scale_depth 1.4 \
    --eval_interval 10 --latest_ckpt_interval 0 --log_interval 1 \
    --experiment_name mup_llama_d_muon_smoke --log_optimizer_groups
