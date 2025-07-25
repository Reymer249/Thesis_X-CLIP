#!/bin/bash

# Set variables
DATA_PATH="/vol/home/s3705609/Desktop/data_vatex"
job_name="xclip_vatex_run"

# Use torchrun instead of torch.distributed.launch (recommended in newer PyTorch)
python -m torch.distributed.run --nproc_per_node=1 main_xclip.py \
    --do_train \
    --num_thread_reader=8 \
    --lr 1e-4 \
    --batch_size=64 \
    --batch_size_val=64 \
    --epochs=5 \
    --n_display=100 \
    --data_path ${DATA_PATH} \
    --features_path ${DATA_PATH}/clips \
    --output_dir ${DATA_PATH}/${job_name}_test_run \
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
    --train_path_from_data_folder splits_txt/vatex_train_avail.txt \
    --val_path_from_data_folder splits_txt/vatex_val_avail.txt \
    --test_path_from_data_folder splits_txt/vatex_test_avail.txt \
    --captions_path_from_data_folder splits_txt/captions_avail_formatted.json \

# vatex_[train/val/test]_avail.txt are the txt files with the video_ids on each line. It is just a "repository" of all
# video ids we have or consider. captions_avail_formatted.json is the file with captions for each video
