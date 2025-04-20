#!/bin/bash

# Set variables
DATA_PATH="/vol/home/s3705609/Desktop/data_vatex"
job_name="xclip_vatex_features_extraction_batch_16"

# Use torchrun instead of torch.distributed.launch (recommended in newer PyTorch)
python get_features.py \
    --model_file /vol/home/s3705609/Desktop/data_vatex/x-clip_checkpoints/xclip_vatex_run/pytorch_model.bin.4 \
    --max_words 32 \
    --max_frames 12 \
    --datatype vatex \
    --sim_header seqTransf \
    --data_path ${DATA_PATH} \
    --features_path ${DATA_PATH}/clips \
    --feature_framerate 1 \
    --output_dir ${DATA_PATH}/features \

