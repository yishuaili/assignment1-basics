#!/bin/bash

python cs336_basics/train.py \
    --dataset_name='ts' \
    --context_length=256 \
    --batch_size=256 \
    --vocab_size=10000 \
    --d_model=768 \
    --d_ff=3072 \
    --num_layers=12 \
    --num_heads=12 \
    --lr_max=0.0005 \
    --total_iters=200 \
    --wandb_project='cs336_basics' \
    --wandb_run_name="leaderboard" \
    --wandb_logging=True