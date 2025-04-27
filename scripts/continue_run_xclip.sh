#!/bin/bash

# Set variables
EPOCH=2
DATA_PATH="/data/s3705609/VATEX"
job_name="xclip_vatex_run_vibranium_1"
CHECKPOINT_PATH="${DATA_PATH}/x-clip_checkpoints/${job_name}/pytorch_model.bin.${EPOCH}" # Replace {EPOCH} with actual epoch number

# Use torchrun instead of torch.distributed.launch (recommended in newer PyTorch)
python -m torch.distributed.run --nproc_per_node=1 main_xclip_work.py \
    --do_train \
    --num_thread_reader=8 \
    --lr 1e-4 \
    --batch_size=64 \
    --batch_size_val=64 \
    --epochs=5 \
    --n_display=100 \
    --data_path ${DATA_PATH} \
    --features_path ${DATA_PATH}/clips \
    --output_dir ${DATA_PATH}/x-clip_checkpoints/${job_name} \
    --resume_model ${CHECKPOINT_PATH} \
    --max_words 32 \
    --max_frames 12 \
    --datatype vatex \
    --expand_msrvtt_sentences \
    --feature_framerate 1 \
    --coef_lr 1e-3 \
    --freeze_layer_num 0 \
    --slice_framepos 2 \
    --loose_type \
    --linear_patch 2d \
    --sim_header seqTransf \
    --pretrained_clip_name ViT-B/32 \
    --use_wandb \
    --wandb_project x-clip \
    --wandb_name ${job_name}
