#!/bin/bash

# Small-memory 124M dense Llama Newton-Muon entry.
# This is a single-card script; do not add torchrun or --distributed_backend.
python ./src/main.py --config_format base --model llama --device cuda:0 \
    --n_embd 768 --n_head 12 --n_layer 12 \
    --batch_size 16 --sequence_length 512 --acc_steps 2 \
    --dataset fineweb --iterations 128000 \
    --dropout 0.0 --warmup_steps 2000 --grad_clip 0.5 --seed 0 \
    --opt newton-muon --lr 1e-3 --muon_lr_factor 1e-2 --weight_decay 0.1 --scheduler cos \
    --beta1 0.8 --beta2 0.999 --momentum 0.95 --nesterov True --muon_ns_steps 5 \
    --newton_muon_precond_every 32 --newton_muon_precond_ewma 0.95 \
    --newton_muon_precond_init_diag 1e-3 --newton_muon_precond_ridge_mult 0.2 \
    --newton_muon_precond_eps 1e-8 \
    --wandb --wandb_project YOUR_WANDB-PROJECT --wandb_entity YOUR-WANDB-ENTITY \
    --eval_interval 115 --latest_ckpt_interval 1000
