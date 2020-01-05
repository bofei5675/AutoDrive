#!/bin/bash
#SBATCH --job-name=auto_drive
#SBATCH --nodes=1
#SBATCH --cpus-per-task=4
#SBATCH --time=8:30:00
#SBATCH --mem=16GB
#SBATCH --gres=gpu:1

source activate capstone
cd ../
python main.py -m HG2 -sd run_test/ -nc 8 -ns 2 -nf 256 -lf MSE -s 1 -bs 2 -db no -a 2 -b 4
