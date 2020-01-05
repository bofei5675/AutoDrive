#!/bin/bash
#SBATCH --job-name=auto_drive
#SBATCH --nodes=1
#SBATCH --cpus-per-task=4
#SBATCH --time=24:30:00
#SBATCH --mem=16GB
#SBATCH --gres=gpu:k80:2

source activate capstone
cd ../
python main.py -m HG2 -sd run/ -nc 8 -ns 8 -nf 256 -lf MSE -s 1 -db no
