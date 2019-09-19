#!/bin/bash

#SBATCH --job-name=quickplot
#SBATCH --output=quickplot.out
#SBATCH --error=quickplot.err
#SBATCH --time=36:00:00
#SBATCH --exclusive
#SBATCH --partition=broadwl

module load Anaconda3/5.3.0
source activate embedding-base

python ./quickplot.py
