# request 2 gpu
srun -t2:30:00 --mem=8000 --gres=gpu:2 -c2 --pty /bin/bash

# request 1 gpu
srun -t2:30:00 --mem=8000 --gres=gpu:1 --pty /bin/bash

# request 4 cpu
srun -c4 -t2:00:00 --mem=8000 --pty /bin/bash