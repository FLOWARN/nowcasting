#!/bin/bash
#SBATCH --job-name=ldm_csi
#SBATCH --nodes=1                # Single node only
#SBATCH --ntasks=1               # Single task
#SBATCH --gpus-per-node=4        # Request 4 GPUs
#SBATCH --cpus-per-task=40       # CPU cores per task
#SBATCH --mem=200G               # Memory per node
#SBATCH --partition=gpu1         # Partition name (adjust to your cluster)
#SBATCH --output=logs/ldm_csi_%j.out
#SBATCH --error=logs/ldm_csi_%j.err

# Load conda
source /home1/ppatel2025/miniconda3/etc/profile.d/conda.sh

# Activate environment
conda activate tito_env

# Print system info
echo "Job ID: $SLURM_JOB_ID"
echo "Node: $(hostname)"
echo "GPUs available:"
nvidia-smi --query-gpu=name,memory.total --format=csv

echo "Starting single-node multi-GPU training..."
# Example: resuming from checkpoint: checkpoints/ldm_epoch_5.pt
# Run training (automatically detects and uses all available GPUs)
python train.py --lr 5e-5 --loss-type mse --batch-size 1 --epochs 100
# python calculate_csi_rapsd.py --save-predictions --predictions-file csi_predictions_latent_diff.h5
# python evaluate_model.py --model-path checkpoints/ldm_epoch_0.pt

echo "Training completed!"