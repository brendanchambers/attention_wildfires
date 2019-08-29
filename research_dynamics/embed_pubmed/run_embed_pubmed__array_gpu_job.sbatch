#!/bin/bash

#SBATCH --job-name=embed_pubmed_job_array
#SBATCH --output=scibert_%a.out
#SBATCH --error=scibert_%a.err
#SBATCH --array=1-49
#SBATCH --time=36:00:00
#SBATCH --mem=32G
#SBATCH --nodes=1
#SBATCH --partition=gpu2
#SBATCH --ntasks=4
#SBATCH --gres=gpu:1

echo "My SLURM_ARRAY_TASK_I: " $SLURM_ARRAY_TASK_ID

module load Anaconda3/5.3.0
source activate embedding-base

module load cuda/10.0

python ./embed_pubmed__array_gpu_job.py $SLURM_ARRAY_TASK_ID
