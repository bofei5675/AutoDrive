#!/bin/bash
#SBATCH --job-name=eval_auto_drive
#SBATCH --nodes=1
#SBATCH --cpus-per-task=4
#SBATCH --time=24:30:00
#SBATCH --mem=16GB
#SBATCH --gres=gpu:1

source activate capstone
cd ../
python make_prediction.py -lm /scratch/bz1030/auto_drive/run/model_HG2_stack_2_features_256_MSE_2020-01-05_00-21-52/model_10.pth