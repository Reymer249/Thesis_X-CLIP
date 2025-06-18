#!/bin/bash

# Set variables
DATA_PATH="/data/s3705609/VATEX"
job_name=$1

# Use torchrun instead of torch.distributed.launch (recommended in newer PyTorch)
python -m torch.distributed.run --nproc_per_node=1 main_xclip.py \
    --do_eval \
    --init_model ${DATA_PATH}/x-clip_checkpoints/${job_name}/pytorch_model.bin.4 \
    --num_thread_reader=8 \
    --lr 1e-4 \
    --batch_size=64 \
    --batch_size_val=64 \
    --n_display=100 \
    --data_path ${DATA_PATH} \
    --features_path ${DATA_PATH}/clips \
    --output_dir ${DATA_PATH}/x-clip_checkpoints/${job_name}/test_results \
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
    --train_path_from_data_folder splits_txt/vatex_val_avail.txt \
    --val_path_from_data_folder splits_txt/vatex_val_avail.txt \
    --test_path_from_data_folder splits_txt/vatex_val_avail.txt \
    --captions_path_from_data_folder splits_txt/captions_avail_formatted.json \

# I mapped everything to the validation set just to be sure :)
