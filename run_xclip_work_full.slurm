#!/bin/bash
###########################
# Settings for slurm
###########################
#SBATCH --job-name=run_xclip
#SBATCH --out=%x_%j.out
#SBATCH --mail-user="007litovka@gmail.com"
#SBATCH --mail-type="ALL"
#SBATCH --partition=gpu-short
#SBATCH --time=4:00:00
#SBATCH --ntasks=2
#SBATCH --ntasks-per-node=2
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
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

# Function to run the main task
run_main_task() {
    # Explicitly set distributed environment variables
    export MASTER_ADDR="localhost"
    export MASTER_PORT=$(($RANDOM + 10000))
    
    # Print detailed environment information for debugging
    echo "CUDA_VISIBLE_DEVICES: $CUDA_VISIBLE_DEVICES"
    echo "MASTER_ADDR: $MASTER_ADDR"
    echo "MASTER_PORT: $MASTER_PORT"
    
    # Explicitly set CUDA_VISIBLE_DEVICES=0,1 to make sure PyTorch sees both GPUs
    export CUDA_VISIBLE_DEVICES=0,1
    
    # Run the main python command
    python -m torch.distributed.run \
     --nproc_per_node=2 \
     --master_addr="$MASTER_ADDR" \
     --master_port=$MASTER_PORT \
     main_xclip_work.py --do_train --num_thread_reader=8 \
     --lr 1e-4 --batch_size=16 --batch_size_val=16 \
     --epochs=5 --n_display=100 \
     --data_path ${DATA_PATH}/VATEX \
     --features_path ${DATA_PATH}/VATEX/clips \
     --output_dir ${DATA_PATH}/VATEX/x-clip_checkpoints/${job_name} \
     --max_words 32 --max_frames 12 \
     --datatype vatex --expand_msrvtt_sentences \
     --feature_framerate 1 --coef_lr 1e-3 \
     --freeze_layer_num 0 --slice_framepos 2 \
     --loose_type --linear_patch 2d --sim_header seqTransf \
     --n_gpu 2 \
     --pretrained_clip_name ViT-B/32
}

# Auto-restart logic
MAX_RETRIES=3
RETRY_COUNT=0
TIMEOUT=600  # 10 minutes in seconds

while [ $RETRY_COUNT -lt $MAX_RETRIES ]; do
    echo "Starting job attempt $((RETRY_COUNT+1)) of $MAX_RETRIES"
    START_TIME=$(date +%s)
    
    # Run the task and capture its exit code
    run_main_task 2>&1 | tee -a /home/s3705609/data1/VATEX/log/${job_name}_attempt$((RETRY_COUNT+1))
    EXIT_CODE=${PIPESTATUS[0]}
    
    END_TIME=$(date +%s)
    DURATION=$((END_TIME - START_TIME))
    
    if [ $EXIT_CODE -eq 0 ]; then
        echo "Job completed successfully on attempt $((RETRY_COUNT+1))"
        break
    else
        echo "Job failed with exit code $EXIT_CODE on attempt $((RETRY_COUNT+1))"
        
        # Check if the job ran for less than the timeout
        if [ $DURATION -lt $TIMEOUT ]; then
            echo "Job ran for $DURATION seconds, which is less than the timeout of $TIMEOUT seconds"
            echo "This might indicate a CUDA or OOM error. Cleaning up and restarting..."
            
            # Optional: Add cleanup commands here if needed
            # For example, you might want to clear GPU memory or remove temporary files
            nvidia-smi --gpu-reset 2>/dev/null || echo "Could not reset GPUs (this is normal if not supported)"
            sleep 30  # Give system some time to recover
            
            RETRY_COUNT=$((RETRY_COUNT+1))
            echo "Retry $RETRY_COUNT of $MAX_RETRIES"
        else
            echo "Job ran for $DURATION seconds, which is more than the timeout of $TIMEOUT seconds"
            echo "This suggests a different issue than a quick CUDA/OOM error. Not restarting."
            break
        fi
    fi
    
    if [ $RETRY_COUNT -eq $MAX_RETRIES ]; then
        echo "Maximum number of retries ($MAX_RETRIES) reached. Giving up."
    fi
done

echo "Job execution finished with final exit status: $EXIT_CODE"