# srun -A chip -p chip-gpu --mem=128G --gres=gpu:NVIDIA_A100:1 --pty /bin/bash
# srun -A chip -p bch-gpu --mem=128G --gres=gpu:NVIDIA_A100:1 --pty /bin/bash
srun -A bch -p bch-gpu --mem=128G --gres=gpu:NVIDIA_A100:1 --pty /bin/bash
