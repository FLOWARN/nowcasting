#!/bin/bash

#SBATCH --job-name nowcast_job


#SBATCH --partition=SOE_main
#SBATCH --cpus-per-task=1            # Cores per task (>1 if multithread tasks)
#SBATCH --mem=128000                  # Real memory (RAM) required (MB)
#SBATCH --time=00:05:00              # Total run time limit (HH:MM:SS)

#SBATCH --error=/vol_efthymios/NFS07/en279/SERVIR/TITO_test3/ML/slurm_outputs/testjob.%J.err 

#SBATCH --output=/vol_efthymios/NFS07/en279/SERVIR/TITO_test3/ML/slurm_outputs/testjob.%J.out

# export CUDA_VISIBLE_DEVICES=4

conda init
conda activate pysteps

nvidia-smi
# This script contains predictions of the USA events shown in the paper. The output folders '0' to '9' correspond to Fig. 2 in the main text, Extended Data Fig. 2-6 and Supplementary Fig.2-5 in order.
python3 PDDN.py