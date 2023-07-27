#!/bin/bash

#SBATCH --job-name="pala_benchmark"
#SBATCH --time=08:00:00

#SBATCH --mail-user=christopher.hahne@unibe.ch
#SBATCH --mail-type=none
#SBATCH --cpus-per-task=4
#SBATCH --mem-per-cpu=16G
#SBATCH --qos=job_gpu
#SBATCH --account=ws_00000
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --array=1-7%1

module load Python/3.8.6-GCCcore-10.2.0

source ~/23_culminate/stofnet/venv/bin/activate

python -c "import torch; print(torch.cuda.is_available())"

param_store=~/23_culminate/stofnet/bash_scripts/array_pala_params.txt

model=$(cat $param_store | awk -v var=$SLURM_ARRAY_TASK_ID 'NR==var {print $1}')
model_file=$(cat $param_store | awk -v var=$SLURM_ARRAY_TASK_ID 'NR==var {print $2}')
threshold=$(cat $param_store | awk -v var=$SLURM_ARRAY_TASK_ID 'NR==var {print $3}')

cd ~/23_culminate/

python -c "import torch; print(torch.cuda.is_available())"

echo "Model: ${model}, Model File: ${model_file}"

python ./stofnet/main.py model=${model} model_file=${model_file} th=${threshold} evaluate=True batch_size=1 etol=1 data_dir=/storage/workspaces/artorg_aimi/ws_00000/chris/PALA_data_InSilicoFlow/ logging=pala_array ch_gap=1

python ./stofnet/utils/load_table_contents.py pala_array
