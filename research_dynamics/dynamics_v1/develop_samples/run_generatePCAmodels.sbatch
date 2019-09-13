#!/bin/bash

#SBATCH --job-name=generate_pca_models
#SBATCH --output=masspca.out
#SBATCH --error=masspca.err
#SBATCH --time=36:00:00
#SBATCH --mem=128G
#SBATCH --nodes=1
#SBATCH --partition=bigmem2
#SBATCH --ntasks=14

module load mysql
module load Anaconda3/5.3.0
source activate embedding-base

module load cuda/10.0

python ./encode_validation__scibertA.py
python ./encode_validation__scibertB.py
python ./encode_validation__scibertC.py