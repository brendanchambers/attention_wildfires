#!/bin/bash

#SBATCH --job-name=scibert_validation_experiments
#SBATCH --output=scibert.out
#SBATCH --error=scibert.err
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

python ./encode_validation__scibertA.py
python ./encode_validation__scibertB.py
python ./encode_validation__scibertC.py