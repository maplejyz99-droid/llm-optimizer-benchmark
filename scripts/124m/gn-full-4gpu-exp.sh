#!/bin/bash

# Experimental 4-GPU GN launch. The current GN DDP path does not explicitly
# all-reduce GN gradients, so use this as an exploratory run rather than a
# formal multi-GPU baseline.
torchrun --nproc_per_node=4 ./src/main.py --config_format base --model llama --distributed_backend nccl \
    --n_embd 768 --n_head 12 --n_layer 12 \
    --batch_size 32 --sequence_length 512 --acc_steps 1 \
    --dataset slimpajama --iterations 11445 \
    --dropout 0.0 --warmup_steps 375 --grad_clip 0.5 --seed 0 \
    --opt gn-full --scheduler cos \
    --gn_inner_iters 4 --gn_inner_lr 1e-3 --gn_inner_b1 0.9 --gn_inner_b2 0.999 --gn_inner_wd 0.0 \
    --gn_linesearch --gn_ls_range 5 \
    --dtype bfloat16 \
    --eval_interval 200 --log_interval 50 \
    --run_prefix gn_full_124m_1p5B_4gpu_exp
