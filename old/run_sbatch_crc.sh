#!/bin/bash
# Sample batchscript to run a pytorch job on HPC
#SBATCH --partition=bch-gpu                           # queue to be used
#SBATCH --account=bch
#SBATCH --time=119:59:59                         # Running time (in hours-minutes-seconds)
#SBATCH --job-name=lm2_crc                       # Job name
#SBATCH --mail-type=BEGIN,END,FAIL              # send and email when the job begins, ends or fails
#SBATCH --mail-user=eli.goldner@childrens.harvard.edu            # Email address to send the job status
#SBATCH --output=output_%j.txt                         # Name of the output file
#SBATCH --nodes=1                               # Number of gpu nodes
#SBATCH --gres=gpu:NVIDIA_A100:1                          # Number of gpu devices on one gpu node
#SBATCH --mem=128GB

module load anaconda3
source activate /home/ch231037/miniconda3/envs/lora1

input_dir=../crc_llama2_testcase/
output_dir=.
cancer_type=crc
# input_filename=${cancer_type}_dev_for_llm.json
input_filename=sample.json

python ./fewshot_llama2.py --input_dir $input_dir \
       --input_filename $input_filename \
       --cancer_type $cancer_type \
       --output_dir $output_dir \
       --few_shot_example_path timenorm_examples.txt > \
       llama2_test_case.txt
