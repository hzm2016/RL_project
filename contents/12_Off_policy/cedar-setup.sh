#!/bin/sh

#SBATCH --account=def-sutton
#SBATCH --mail-type=ALL
#SBATCH --mail-user=zhimin3@ualberta.ca
#SBATCH --error=slurm-%j-%n-%a.err
#SBATCH --output=slurm-%j-%n-%a.out
#SBATCH --cpus-per-task=1
#SBATCH --mem=4000M
#SBATCH --time=0-00:10

conda create -n lambda-greedy python=3.6 pip
source activate lambda-greedy
pip install ipython jupyter matplotlib numpy pyserial seaborn visdom
