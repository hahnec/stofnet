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
#module load CUDA/11.3.0-GCC-10.2.0
#module load cuDNN/8.2.0.53-CUDA-11.3.0
#module load Workspace

cd ~/23_culminate/stofnet

python -m venv venv

source ~/23_culminate/stofnet/venv/bin/activate

python -m pip install -r requirements.txt
python3 -m pip install git+https://github.com/CyberZHG/torch-same-pad.git

python -c "import torch; print(torch.cuda.is_available())"

cd ..

python ./stofnet/main.py evaluate=False logging=train data_dir=./datasets/stof_chirp101_dataset th=Null rf_scale_factor=10
