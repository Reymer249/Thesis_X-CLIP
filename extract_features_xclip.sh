#!/bin/bash

# Set variables
DATA_PATH="/vol/home/s3705609/Desktop/data_vatex"
job_name="xclip_vatex_run_lu_pc_2"

# Use torchrun instead of torch.distributed.launch (recommended in newer PyTorch)
python -m torch.distributed.run --nproc_per_node=1 test_x-clip_vatex.py \
    --do_eval \
    --num_thread_reader=8 \
    --lr 1e-4 \
    --batch_size=64 \
    --batch_size_val=64 \
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
    --models_path /vol/home/s3705609/Desktop/data_vatex/x-clip_checkpoints/xclip_vatex_run \
    --num_epochs=5 \
    --changed_sentences_jsons_path /vol/home/s3705609/Desktop/data_vatex/splits_txt
