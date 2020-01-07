#!/bin/bash
#SBATCH --job-name=auto_drive
#SBATCH --nodes=1
#SBATCH --cpus-per-task=4
#SBATCH --time=24:30:00
#SBATCH --mem=16GB
#SBATCH --gres=gpu:k80:1

source activate capstone
cd ../
python main.py -m HG2 -sd run2/ -nc 8 -ns 4 -nf 256 -lf FL -s 5 -db no -tp 0.2 -pt yes -vs 0.2 -g 1
