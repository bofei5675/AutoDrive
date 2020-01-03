# request a gpu
<<<<<<< HEAD
srun -t2:30:00 --mem=8000 --gres=gpu:k80:1 --pty /bin/bash
=======
srun -t1:30:00 --mem=8000 --gres=gpu:1 --pty /bin/bash
>>>>>>> 958c3ed67d74887374148c8170fdf9cba8f46999

# request 4 cpu
srun -c4 -t2:00:00 --mem=8000 --pty /bin/bash