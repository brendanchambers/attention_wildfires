#!/bin/bash

#SBATCH --job-name=process_year_state
#SBATCH --output=yearstate.out
#SBATCH --error=yearstate.err
#SBATCH --time=36:00:00
#SBATCH --mem=256G
#SBATCH --nodes=1
#SBATCH --partition=bigmem2
#SBATCH --ntasks=24

module load mysql
mysqld --defaults-file=~/.my.cnf &

module load Anaconda3/5.3.0
source activate embedding-base

python ./process_year_state.py
