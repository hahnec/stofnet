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

module load Python/3.8.6-GCCcore-10.2.0

source ~/23_culminate/stofnet/venv/bin/activate

python -c "import torch; print(torch.cuda.is_available())"

cd ~/23_culminate/

python -c "import torch; print(torch.cuda.is_available())"

python ./stofnet/main.py model=${model} model_file=${model_file} th=${threshold} evaluate=True batch_size=1 etol=1 data_dir=./datasets/stof_chirp101_dataset logging=chirp_single rf_scale_factor=10
