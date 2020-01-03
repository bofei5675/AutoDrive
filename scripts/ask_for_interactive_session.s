# request a gpu
srun -t2:30:00 --mem=8000 --gres=gpu:k80:1 --pty /bin/bash

# request 4 cpu
srun -c4 -t2:00:00 --mem=8000 --pty /bin/bash