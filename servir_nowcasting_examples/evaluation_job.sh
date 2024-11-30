#!/bin/bash


#SBATCH --job-name dgmr_evaluation



#SBATCH --partition=SOE_main
#SBATCH --cpus-per-task=1            # Cores per task (>1 if multithread tasks)
#SBATCH --mem=56000                  # Real memory (RAM) required (MB)
#SBATCH --time=4:00:00              # Total run time limit (HH:MM:SS)

#SBATCH --error=/volume/NFS/aa3328/slurm_outputs/testjob.%J.err 
#SBATCH --output=/volume/NFS/aa3328/slurm_outputs/testjob.%J.out

# export CUDA_VISIBLE_DEVICES=4

conda init
conda activate tito_env

nvidia-smi
# This script contains predictions of the USA events shown in the paper. The output folders '0' to '9' correspond to Fig. 2 in the main text, Extended Data Fig. 2-6 and Supplementary Fig.2-5 in order.
python3 evaluate_results_gpu.py