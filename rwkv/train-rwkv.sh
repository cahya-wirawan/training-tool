#!/bin/bash
# A script to train RWKV 7B parameters (World model) on 8xA100 with 40GB GPU ram.
# Place this script under the directory RWKV-v4neo of https://github.com/BlinkDL/RWKV-LM

export TORCH_EXTENSIONS_DIR=$HOME/.cache/torch_extensions
export RWKV_JIT_ON=1
export RWKV_CUDA_ON=1

# The base RWKV 7B World model from https://huggingface.co/BlinkDL/rwkv-4-world/resolve/main/RWKV-4-World-7B-v1-20230626-ctx4096.pth
BASE_MODEL="$HOME/Work/models/rwkv/world/RWKV-4-World-7B-v1-20230626-ctx4096.pth"

# The checkpoint directory
OUTPUT_DIR="./checkpoint"

# The dataset in binidx format. Use this repo to convert jsonl dataset to binidx format: https://github.com/Abel2076/json2binidx_tool
DATA_FILE="$HOME/Work/data/wiki/indonesian-dataset-new_text_document"

# Wandb project name
WANDB_PROJECT="RWKV-World-7b"

mkdir -p $OUTPUT_DIR

if [ "$(ls -A $OUTPUT_DIR)" ]; then
	COUNTER=$(ls -lt $OUTPUT_DIR/rwkv*|head -1| sed -E 's/.+rwkv-(.+)\.pth/\1/')
	MODEL_NAME=$OUTPUT_DIR/rwkv-${COUNTER}.pth
else
	COUNTER=-1
	MODEL_NAME=$BASE_MODEL
fi


echo "Start with model ${MODEL_NAME}"
echo python train.py --load_model $MODEL_NAME --wandb "$WANDB_PROJECT" --proj_dir "${OUTPUT_DIR}" \
	--data_file "$DATA_FILE" --data_type "binidx" --vocab_size 65536 \
	--ctx_len 2048 --epoch_steps 1000 --epoch_count 200 --epoch_begin $((COUNTER+1)) --epoch_save 1 \
	--micro_bsz 1 --n_layer 32 --n_embd 4096 --pre_ffn 0 --head_qk 0 \
	--lr_init 1e-5 --lr_final 9e-6 --warmup_steps 200 --beta1 0.9 --beta2 0.999 --adam_eps 1e-8 \
	--accelerator gpu --devices 8 --precision bf16 --strategy deepspeed_stage_2 --grad_cp 1

