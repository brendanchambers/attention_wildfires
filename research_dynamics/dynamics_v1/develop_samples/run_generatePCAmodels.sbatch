#!/bin/bash

#SBATCH --job-name=generate_pca_models
#SBATCH --output=masspca.out
#SBATCH --error=masspca.err
#SBATCH --time=36:00:00
#SBATCH --mem=128G
#SBATCH --nodes=1
#SBATCH --partition=bigmem2
#SBATCH --ntasks=24

module load mysql
module load Anaconda3/5.3.0
source activate embedding-base

python ./generate_dimensionality_reduction_models.py