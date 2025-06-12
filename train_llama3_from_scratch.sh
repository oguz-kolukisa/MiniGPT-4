#!/bin/bash
# Train MiniGPT-4 with Llama 3 in two stages
# Usage: bash train_llama3_from_scratch.sh NUM_GPU

NUM_GPU=${1:-1}

# Stage 1 pretraining
torchrun --nproc-per-node $NUM_GPU train.py --cfg-path train_configs/minigpt4_llama3_stage1_pretrain.yaml

# Stage 2 finetuning
torchrun --nproc-per-node $NUM_GPU train.py --cfg-path train_configs/minigpt4_llama3_stage2_finetune.yaml

