#!/bin/bash
# Sample batchscript to run a pytorch job on HPC
#SBATCH --partition=chip-gpu                           # queue to be used
#SBATCH --account=chip
#SBATCH --time=119:59:59                         # Running time (in hours-minutes-seconds)
#SBATCH --job-name=lora1_env_test                       # Job name
#SBATCH --mail-type=BEGIN,END,FAIL              # send and email when the job begins, ends or fails
#SBATCH --mail-user=eli.goldner@childrens.harvard.edu            # Email address to send the job status
#SBATCH --output=output_%j.txt                         # Name of the output file
#SBATCH --nodes=1                               # Number of gpu nodes
#SBATCH --gres=gpu:TITAN_RTX:1                          # Number of gpu devices on one gpu node
#SBATCH --mem=4GB

# module load anaconda3
# source activate /home/ch231037/miniconda3/envs/lora1

source /home/ch231037/.bashrc
conda activate lora1

python -c "import transformers; print('NAME JEFF')"
