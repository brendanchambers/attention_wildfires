#!/bin/bash

#SBATCH --job-name=array_gpu_eval_sbatch
#SBATCH --output=scibert_%a.out
#SBATCH --error=scibert_%a.err
#SBATCH --array=1-2
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

python ./my_array_gpu_evaluation_script_nodb.py $SLURM_ARRAY_TASK_ID
