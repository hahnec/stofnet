#!/bin/bash

#SBATCH --job-name="stofnet"
#SBATCH --time=11:30:00

#SBATCH --mail-user=christopher.hahne@unibe.ch
#SBATCH --mail-type=none
#SBATCH --cpus-per-task=4
#SBATCH --mem-per-cpu=8G
#SBATCH --qos=job_gpu
#SBATCH --account=ws_00000
#SBATCH --partition=gpu
#SBATCH --gres=gpu:rtx2080ti:1

module load Python/3.8.6-GCCcore-10.2.0

cd ~/stofnet

python -m venv venv

source ~/stofnet/venv/bin/activate

python -m pip install -r requirements.txt
python -m pip install -r datasets/pala_dataset/requirements.txt

python -c "import torch; print(torch.cuda.is_available())"

mkdir -p ckpts

python main.py evaluate=False logging=train
