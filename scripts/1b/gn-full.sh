#!/bin/bash

python ./src/main.py --config_format base --model llama \
    --n_embd 1792 --n_head 14 --n_layer 24 \
    --batch_size 62 --sequence_length 512 --acc_steps 32 \
    --dataset fineweb --iterations 48000 \
    --dropout 0.0 --warmup_steps 2000 --grad_clip 0.1 --seed 0 \
    --opt gn-full --scheduler cos \
    --gn_inner_iters 4 --gn_inner_lr 1e-3 --gn_inner_b1 0.9 --gn_inner_b2 0.999 --gn_inner_wd 0.0 \
    --gn_linesearch --gn_ls_range 5 \
    --wandb --wandb_project YOUR_WANDB-PROJECT --wandb_entity YOUR-WANDB-ENTITY \
    --eval_interval 200 --permanent_ckpt_interval 2000 \
