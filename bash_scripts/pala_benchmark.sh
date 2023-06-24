#!/bin/bash

#SBATCH --job-name="pala_benchmark"
#SBATCH --time=15:00:00

#SBATCH --mail-user=christopher.hahne@unibe.ch
#SBATCH --mail-type=none
#SBATCH --cpus-per-task=4
#SBATCH --mem-per-cpu=8G
#SBATCH --qos=job_gpu
#SBATCH --account=ws_00000
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --array=1-10%1

module load Python/3.8.6-GCCcore-10.2.0

source ~/23_culminate/stofnet/venv/bin/activate

python -c "import torch; print(torch.cuda.is_available())"

param_store=./bash_scripts/array_params.txt

model=$(cat $param_store | awk -v var=$SLURM_ARRAY_TASK_ID 'NR==var {print $1}')
model_file=$(cat $param_store | awk -v var=$SLURM_ARRAY_TASK_ID 'NR==var {print $2}')
threshold=$(cat $param_store | awk -v var=$SLURM_ARRAY_TASK_ID 'NR==var {print $3}')

cd ~/23_culminate/stofnet/

python -c "import torch; print(torch.cuda.is_available())"

echo "Model: ${model}, Model File: ${model_file}"

python ./main.py model=${model} model_file=${model_file} th=${threshold} evaluate=True ch_gap=1 sequences=[1,2,3,4,5,6,7,8,9,10,11,12,13,14,15] data_dir=storage/homefs/ch21z139/PALA_dataset/PALA_data_InSilicoFlow

