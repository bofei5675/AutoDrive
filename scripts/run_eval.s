#!/bin/bash
#SBATCH --job-name=eval_auto_drive
#SBATCH --nodes=1
#SBATCH --cpus-per-task=4
#SBATCH --time=8:30:00
#SBATCH --mem=16GB
#SBATCH --gres=gpu:1

source activate capstone
cd ../
python make_prediction.py -lm /scratch/bz1030/auto_drive/run/model_aug_HG2_stack_4_feat_256_g_40.0_FL_2020-01-11_04-14-07/model_27.pth -t 0
