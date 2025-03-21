#!/bin/bash
###########################
# Settings for slurm
###########################
#SBATCH --job-name=run_xclip
#SBATCH --out=%x_%j.out
#SBATCH --mail-user="007litovka@gmail.com"
#SBATCH --mail-type="ALL"
#SBATCH --partition=gpu-short
#SBATCH --time=10:00
#SBATCH --ntasks=2
#SBATCH --ntasks-per-node=2
#SBATCH --cpus-per-task=1
#SBATCH --mem=16G
#SBATCH --gres=gpu:2
###########################
# Set up your software environment
###########################
module purge
module load ALICE/default
module load slurm
module load CUDA/12.1.1
module load PyTorch/2.1.2-foss-2023a-CUDA-12.1.1
module load torchvision/0.16.0-foss-2023a-CUDA-12.1.1
source /home/s3705609/data1/venv_clip/bin/activate
nvcc --version
g++ --version
echo "## Available CUDA devices: $CUDA_VISIBLE_DEVICES"
echo "## Checking status of CUDA device with nvidia-smi"
nvidia-smi
###########################
# Execute tasks
###########################
job_name="xclip_vatex_vit16"
DATA_PATH="/home/s3705609/data1"

# Explicitly set distributed environment variables
export MASTER_ADDR="localhost"
export MASTER_PORT=$(($RANDOM + 10000))

# Print detailed environment information for debugging
echo "CUDA_VISIBLE_DEVICES: $CUDA_VISIBLE_DEVICES"
echo "MASTER_ADDR: $MASTER_ADDR"
echo "MASTER_PORT: $MASTER_PORT"

# Explicitly set CUDA_VISIBLE_DEVICES=0,1 to make sure PyTorch sees both GPUs
export CUDA_VISIBLE_DEVICES=0,1

# Important: Use Python's distributed launcher with explicit node_rank and nnodes
python -m torch.distributed.run \
    --nproc_per_node=2 \
    --master_addr="$MASTER_ADDR" \
    --master_port=$MASTER_PORT \
    main_xclip_work.py --do_train --num_thread_reader=2 \
    --epochs=5 --batch_size=16 --n_display=1 \
    --data_path ${DATA_PATH}/VATEX \
    --features_path ${DATA_PATH}/VATEX/clips \
    --output_dir ckpts3/${job_name} \
    --lr 1e-4 --max_words 32 --max_frames 12 --batch_size_val 64 \
    --datatype vatex \
    --feature_framerate 1 --coef_lr 1e-3 \
    --freeze_layer_num 0 --slice_framepos 2 \
    --loose_type --linear_patch 2d --sim_header seqTransf \
    --n_gpu 2 --fp16 --fp16_opt_level O2 \
    --pretrained_clip_name ViT-B/16 2>&1 | tee -a /home/s3705609/data1/log/${job_name}