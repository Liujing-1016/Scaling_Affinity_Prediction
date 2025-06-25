#!/bin/bash

# ==============================
# MoleTrain Model Training Script
# 
# This script submits a SLURM job to train the MoE model using 
# a fine-tuning approach. It sets up the environment, loads the required 
# dataset, and runs the training process with specified hyperparameters.
#
# Usage:
#   sbatch this_script.sh
#
# Enviroment: 
# - `myenv` is a Conda virtual environment that satisfies all dependencies for MolE.
# - Python version: 3.10.16
#
# Outputs:
# - Logs: stored in record2/mole_0309_<job_id>.log
# - Errors: stored in record2/mole_0309_<job_id>.err
# - Trained Model stored in /home/x_lchen/proj/mole_public
#
# Author: Liujing
# Date: 2025-03-11
# ==============================

#SBATCH --job-name=mole_train      
#SBATCH --output=/home/x_lchen/proj/mole_public/record2/mole_0309_%j.log 
#SBATCH --error=/home/x_lchen/proj/mole_public/record2/mole_0309_%j.err
#SBATCH -t 2-00:00:00
#SBATCH --gpus=4

# Activate the Conda environment
cd /home/x_lchen/proj/mole_public
source ~/miniconda3/etc/profile.d/conda.sh
conda activate myenv

python --version


# MolE train 
mole_train model=finetune \
  data_file='/proj/berzelius-pharmbio/users/x_lchen/train_set.parquet' \
  checkpoint_path=null \
  dropout=0.2 \
  lr=1.0e-4 \
  task=regression \
  num_tasks=6142 \
  model.name='MolE_Finetune_Regression' \
  model.hyperparameters.datamodule.validation_data='/proj/berzelius-pharmbio/users/x_lchen/val.parquet' \
  model.data.trainer.logger=true