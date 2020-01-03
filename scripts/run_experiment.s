#!/bin/bash
#SBATCH --job-name=auto_drive
#SBATCH --nodes=1
<<<<<<< HEAD
#SBATCH --cpus-per-task=4
#SBATCH --time=24:30:00
=======
#SBATCH --cpus-per-task=2
#SBATCH --time=12:30:00
>>>>>>> 958c3ed67d74887374148c8170fdf9cba8f46999
#SBATCH --mem=16GB
#SBATCH --gres=gpu:1

source activate capstone
cd ../
<<<<<<< HEAD
python main.py -m HG2 -sd run/ -nc 8 -ns 2 -nf 256
=======
python center_net.py
>>>>>>> 958c3ed67d74887374148c8170fdf9cba8f46999
