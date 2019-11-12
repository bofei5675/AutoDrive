#!/bin/bash
#SBATCH --job-name=auto_drive
#SBATCH --nodes=1
#SBATCH --cpus-per-task=2
#SBATCH --time=12:30:00
#SBATCH --mem=16GB
#SBATCH --gres=gpu:1

source activate capstone
cd ../
python center_net.py