#!/bin/bash

#SBATCH --job-name=vanilla_validation_experiments
#SBATCH --output=vanilla.out
#SBATCH --error=vanilla.err
#SBATCH --time=36:00:00
#SBATCH --mem=16G
#SBATCH --nodes=1
#SBATCH --partition=gpu2
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1

echo "My SLURM_ARRAY_TASK_I: " $SLURM_ARRAY_TASK_ID

module load Anaconda3/5.3.0
source activate embedding-base

module load cuda/10.0

python ./encode_validation__vanillaA.py
python ./encode_validation__vanillaB.py
python ./encode_validation__vanillaC.py