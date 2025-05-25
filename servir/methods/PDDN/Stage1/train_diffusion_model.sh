#!/bin/bash
#SBATCH --job-name PDDN_IR
#SBATCH --partition=gpu
#SBATCH --cpus-per-task=1            # Cores per task (>1 if multithread tasks)
#SBATCH --gres=gpu:1                # Request number of GPUs
#SBATCH --mem=150G                # Real memory (RAM) required (MB)
#SBATCH --time=00:10:00              # Total run time limit (HH:MM:SS)
#SBATCH --error=/home/aa3328/slurm_outputs/testjob.%J.err
#SBATCH --output=/home/aa3328/slurm_outputs/testjob.%J.out
conda init
conda activate env
nvidia-smi
python3 PDDN.py