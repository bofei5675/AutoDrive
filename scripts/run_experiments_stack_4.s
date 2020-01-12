#!/bin/bash
#SBATCH --job-name=auto_drive
#SBATCH --nodes=1
#SBATCH --cpus-per-task=4
#SBATCH --time=60:30:00
#SBATCH --mem=16GB
#SBATCH --gres=gpu:k80:2

source activate capstone
cd ../
python main.py -m HG2 -sd run/ -nc 8\
 -ns 4 -nf 256 -lf FL -s 1 -bs 2\
 -db no -tp 0.5 -pt yes -vs 0.2 -g 50 -uc no\
 -uns 0 -norm no -e 45