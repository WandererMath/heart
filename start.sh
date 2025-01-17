#!/bin/bash
#SBATCH --time=04:00:00
#SBATCH --account=PAS2927
#SBATCH --nodes=1
#SBATCH --gpus-per-node=1
#SBATCH --mem=20g

source ~/.bashrc

module load cuda/11.8.0
conda activate ml

python interpret2.py 
conda deactivate
