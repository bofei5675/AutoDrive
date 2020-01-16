#!/bin/bash
#SBATCH --job-name=vis_auto_drive
#SBATCH --nodes=1
#SBATCH --cpus-per-task=4
#SBATCH --time=00:02:00
#SBATCH --mem=16GB
#SBATCH --gres=gpu:1

source activate capstone
cd ../
python pred_visualization.py -lm /scratch/bz1030/auto_drive/resnet/model_aug_res_50_stack_2_feat_256_g_40.0_FL_pt_2020-01-14_02-30-38/model_32.pth