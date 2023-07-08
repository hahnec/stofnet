
#!/bin/bash

#SBATCH --job-name="stofnet"
#SBATCH --time=11:30:00

#SBATCH --mail-user=christopher.hahne@unibe.ch
#SBATCH --mail-type=none
#SBATCH --cpus-per-task=4
#SBATCH --mem-per-cpu=8G
#SBATCH --qos=job_gpu_sznitman
#SBATCH --account=ws_00000
#SBATCH --partition=gpu-invest
#SBATCH --gres=gpu:rtx3090:1

module load Python/3.8.6-GCCcore-10.2.0

cd ~/23_culminate/stofnet

python -m venv venv

source ~/23_culminate/stofnet/venv/bin/activate

python -m pip install -r requirements.txt
python3 -m pip install git+https://github.com/CyberZHG/torch-same-pad.git

python -c "import torch; print(torch.cuda.is_available())"

cd ..

python ./stofnet/main.py evaluate=False logging=train
