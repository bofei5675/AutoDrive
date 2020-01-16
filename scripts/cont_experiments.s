#!/bin/bash
#SBATCH --job-name=auto_drive
#SBATCH --nodes=1
#SBATCH --cpus-per-task=4
#SBATCH --time=48:30:00
#SBATCH --mem=16GB
#SBATCH --gres=gpu:k80:1

source activate capstone
cd ../
python main.py -cp /scratch/bz1030/auto_drive/resnet/model_aug_res_152_stack_2_feat_256_g_40.0_FL_pt_2020-01-14_06-38-12\
 -nc 8 -m res_152\
 -ns 2 -nf 256 -lf FL -s 1 -bs 2\
 -db no -tp 0.5 -pt yes -vs 0.2 -g 40 -uc no\
 -uns 0 -norm no -e 45