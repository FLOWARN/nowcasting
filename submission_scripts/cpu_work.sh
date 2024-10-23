#!/bin/bash

#SBATCH --job-name ir_utils


#SBATCH --nodes 1
#SBATCH --ntasks 1
#SBATCH --mem=100GB
#SBATCH --partition=gpu1

#SBATCH --error=/home1/zhang2012/nowcasting/runs/job.%J.err 
#SBATCH --output=/home1/zhang2012/nowcasting/runs/job.%J.out


# python /home1/zhang2012/nowcasting/servir/datasets/dataLoader_wa_IR.py

export CUDA_VISIBLE_DEVICES=3
python /home1/zhang2012/nowcasting/servir/utils/nc_images_utils.py
