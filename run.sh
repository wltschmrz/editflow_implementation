#!/bin/bash
#SBATCH --partition=yss
#SBATCH --nodelist=beren
#SBATCH --cpus-per-task=16
#SBATCH --mem=24G
#SBATCH --gres=gpu:1
#SBATCH --time=11:00:00
#SBATCH --job-name=my_job
#SBATCH --output=logs/output_%j.log

# Load any required modules
# module load python/3.9
# source activate your_env
# mamba activate plm

# Your commands here
echo "Job started on $(hostname) at $(date)"
echo "GPU info:"
nvidia-smi

# Add your actual commands here
mamba run -n plm python main.py

echo "Job completed at $(date)"
