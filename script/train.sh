#!/bin/bash
#SBATCH --time=04:00:00
#SBATCH --qos=regular
#SBATCH --constraint=gpu
#SBATCH --nodes=1
#SBATCH --gpus=4
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=12

##SBATCH --mail-type=ALL
#SBATCH --output=%j-%x.%u.out
#SBATCH --job-name=default
#SBATCH --account=m3691

#SBATCH --signal=SIGUSR1@90

export NCCL_DEBUG=INFO
export PYTHONFAULTHANDLER=1
export CONDA_PYTHON=/global/homes/z/zangwei/.conda/envs/pcss

conda activate $CONDA_PYTHON

srun python -m project.train $@
