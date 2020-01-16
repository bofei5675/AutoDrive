#!/bin/bash
#SBATCH --job-name=eval_auto_drive
#SBATCH --nodes=1
#SBATCH --cpus-per-task=4
#SBATCH --time=8:30:00
#SBATCH --mem=16GB
#SBATCH --gres=gpu:1

source activate capstone
cd ../
python make_prediction.py -lm /scratch/bz1030/auto_drive/run/model_aug_LHG_stack_2_feat_256_g_50.0_FL_pt_2020-01-15_13-46-22/model_21.pth -t -0.75
