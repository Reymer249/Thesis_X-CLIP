#!/bin/bash

# Set variables
DATA_PATH="/vol/home/s3705609/Desktop/data_vatex"
job_name="xclip_vatex_aug_run_batch16_num_neg8"

# Use torchrun instead of torch.distributed.launch (recommended in newer PyTorch)
python -m torch.distributed.run --nproc_per_node=1 main_xclip_aug.py \
    --do_train \
    --num_thread_reader=16 \
    --lr 1e-4 \
    --batch_size=16 \
    --batch_size_val=16 \
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
    --loss_func maxcol_word \
    --do_neg_aug True \
    --neg_aug_num_sentences 8 \
    --train_path_from_data_folder splits_txt/vatex_train_avail_020.txt \
    --val_path_from_data_folder splits_txt/vatex_val_avail_020.txt \
    --test_path_from_data_folder splits_txt/vatex_test_avail_020.txt \
    --captions_path_from_data_folder splits_txt/captions_avail_formatted.json \
    --hard_negatives_json_path ${DATA_PATH}/splits_txt/hard_negatives_all_pos.json \
    --use_wandb \
    --wandb_project x-clip \
    --wandb_name ${job_name}