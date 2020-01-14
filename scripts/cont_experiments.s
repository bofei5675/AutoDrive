#!/bin/bash
#SBATCH --job-name=auto_drive
#SBATCH --nodes=1
#SBATCH --cpus-per-task=4
#SBATCH --time=48:30:00
#SBATCH --mem=16GB
#SBATCH --gres=gpu:1

source activate capstone
cd ../
python main.py -cp /scratch/bz1030/auto_drive/run/model_aug_HG2_stack_4_feat_256_g_40.0_FL_Score_0.082\
 -nc 8\
 -ns 4 -nf 256 -lf FL -s 1 -bs 2\
 -db no -tp 0.5 -pt yes -vs 0.2 -g 40 -uc no\
 -uns 0 -norm no -e 45