#!/bin/bash

# Set variables
DATA_PATH="/vol/home/s3705609/Desktop/data_vatex"
job_name="xclip_vatex_aug_run_test"

# Use torchrun instead of torch.distributed.launch (recommended in newer PyTorch)
python -m torch.distributed.run --nproc_per_node=1 main_xclip_aug.py \
    --do_train \
    --num_thread_reader=8 \
    --lr 1e-4 \
    --batch_size=8 \
    --batch_size_val=8 \
    --epochs=5 \
    --n_display=100 \
    --data_path ${DATA_PATH} \
    --features_path ${DATA_PATH}/clips \
    --output_dir ${DATA_PATH}/x-clip_checkpoints/${job_name} \
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
    --loss_func fineGrained