#!/bin/bash
#SBATCH --job-name=vae_generative_crps
#SBATCH --partition=gpu1
#SBATCH --gres=gpu:4
#SBATCH --mem=200G
#SBATCH --cpus-per-task=40
#SBATCH --output=/home1/ppatel2025/ppworktp/encoder/encoder_diff_trial_crps_2_fine_tuning/logs/generative_crps_%j.out
#SBATCH --error=/home1/ppatel2025/ppworktp/encoder/encoder_diff_trial_crps_2_fine_tuning/logs/generative_crps_%j.err

# Create logs directory
mkdir -p logs


# Clear conflicting library paths
unset LD_LIBRARY_PATH

# Setup CUDA paths
export CUDA_HOME=/usr/local/cuda-12.9
export PATH=$CUDA_HOME/bin:$PATH
# export LD_LIBRARY_PATH=$CUDA_HOME/lib64:$CONDA_PREFIX/lib:$LD_LIBRARY_PATH
export LD_LIBRARY_PATH=$CUDA_HOME/lib64:$LD_LIBRARY_PATH
# Activate conda
source /home1/ppatel2025/miniconda3/etc/profile.d/conda.sh
conda activate tito_env

# Verify environment
echo "=== CUDA Version ==="
nvcc --version
echo "=== cuDNN Version ==="
python -c "import torch; print(f'cuDNN: {torch.backends.cudnn.version() if torch.cuda.is_available() else \"CUDA not available\"}')"
echo "=== GPU Info ==="
nvidia-smi

#

# Find the best pretrained DETERMINISTIC model
PRETRAINED_MODEL="/home1/ppatel2025/ppworktp/encoder/encoder_diff_trial_crps_2_fine_tuning/experiments/vae_diff_training_20250709_131353/checkpoints/checkpoint_epoch_99.pth"

# Check if pretrained model exists
if [ ! -f "$PRETRAINED_MODEL" ]; then
    echo "Error: Pretrained DETERMINISTIC model not found at $PRETRAINED_MODEL"
    echo "Please check the path and update the script accordingly"
    exit 1
fi

echo "🚀 Starting deterministic-to-generative VAE conversion..."
echo "📋 Using pretrained DETERMINISTIC model: $PRETRAINED_MODEL"
echo "⏰ Time: $(date)"

# Run generative CRPS fine-tuning
python train_generative_crps.py \
    --pretrained "$PRETRAINED_MODEL" \
    --epochs 30 \
    --batch_size 1 \
    --lr 1e-5 \
    --crps_samples 9

echo "✅ Generative conversion completed at: $(date)"
echo "🎉 Your deterministic VAE is now GENERATIVE!"
