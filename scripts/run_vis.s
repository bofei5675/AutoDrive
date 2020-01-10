#!/bin/bash
#SBATCH --job-name=vis_auto_drive
#SBATCH --nodes=1
#SBATCH --cpus-per-task=4
#SBATCH --time=00:01:00
#SBATCH --mem=16GB
#SBATCH --gres=gpu:1

source activate capstone
cd ../
python pred_visualization.py -lm /scratch/bz1030/auto_drive/cbam/model_aug_HG2_stack_4_feat_256_g_1.0_FL_cbam_2020-01-09_07-09-04/model_2.pth