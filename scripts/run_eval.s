#!/bin/bash
#SBATCH --job-name=auto_drive
#SBATCH --nodes=1
#SBATCH --cpus-per-task=4
#SBATCH --time=24:30:00
#SBATCH --mem=16GB
#SBATCH --gres=gpu:1

source activate capstone
cd ../
python make_prediction.py