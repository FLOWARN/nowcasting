#!/bin/bash
#SBATCH --job-name=IR_train_model
#SBATCH --output=/home1/ppatel2025/ppworktp/encoder/encoder_decoder_ir_trial_1/logs/final_train_%j.out
#SBATCH --error=/home1/ppatel2025/ppworktp/encoder/encoder_decoder_ir_trial_1/logs/final_train_%j.err
#SBATCH --partition=gpu1
#SBATCH --mem=240G
#SBATCH --cpus-per-task=90

# Clear conflicting library paths
unset LD_LIBRARY_PATH

# Setup CUDA paths
export CUDA_HOME=/usr/local/cuda-12.9
export PATH=$CUDA_HOME/bin:$PATH
# export LD_LIBRARY_PATH=$CUDA_HOME/lib64:$CONDA_PREFIX/lib:$LD_LIBRARY_PATH
export LD_LIBRARY_PATH=$CUDA_HOME/lib64:$LD_LIBRARY_PATH
# Activate conda
source /home1/ppatel2025/miniconda3/etc/profile.d/conda.sh
conda activate ml-cuda114

# Verify environment
echo "=== CUDA Version ==="
nvcc --version
echo "=== cuDNN Version ==="
python -c "import torch; print(f'cuDNN: {torch.backends.cudnn.version() if torch.cuda.is_available() else \"CUDA not available\"}')"
echo "=== GPU Info ==="
nvidia-smi

# Navigate to project directory
cd /home1/ppatel2025/ppworktp/encoder/encoder_decoder_ir_trial_1

# Run training with explicit library control
LD_PRELOAD=$CONDA_PREFIX/lib/libcudnn_cnn_infer.so.8 python train.py