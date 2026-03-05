#!/bin/bash

torchrun --nproc_per_node=1 ./src/main.py --config_format base --model llama --distributed_backend nccl \
    --n_embd 768 --n_head 12 --n_layer 12 \
    --batch_size 64 --sequence_length 512 --acc_steps 1 \
    --dataset fineweb --iterations 128000 \
    --dropout 0.0 --warmup_steps 2000 --grad_clip 0.5 --seed 0 \
    --opt gn-full --scheduler cos \
    --gn_inner_iters 4 --gn_inner_lr 1e-3 --gn_inner_b1 0.9 --gn_inner_b2 0.999 --gn_inner_wd 0.0 \
    --gn_linesearch --gn_ls_range 5 \
    --wandb --wandb_project YOUR_WANDB-PROJECT --wandb_entity YOUR-WANDB-ENTITY \
    --eval_interval 115 --latest_ckpt_interval 1000 \
