#!/bin/sh

#SBATCH --account=def-sutton
#SBATCH --mail-type=ALL
#SBATCH --mail-user=zhimin3@ualberta.ca
#SBATCH --error=slurm-%j-%n-%a.err
#SBATCH --output=slurm-%j-%n-%a.out
#SBATCH --cpus-per-task=32
#SBATCH --mem=32000M
#SBATCH --time=0-03:00

source activate lambda-greedy
./src/run.sh "$@"
