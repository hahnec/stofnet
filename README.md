# StofNet - Super-resolution time of flight Network

[![arXiv paper link](https://img.shields.io/badge/paper-arXiv:2308.12009-red)](https://arxiv.org/pdf/2308.12009.pdf)

<img src="https://github.com/hahnec/stofnet/blob/master/docs/stofnet_arch.svg" width="750" scale="100%">

<br>
<br>

## Installation

`$ python3 -m venv venv`

`$ source venv/bin/activate`

`$ python3 -m pip install -r requirements.txt`

`$ unzip datasets/stof_chirp101_dataset.zip -d datasets/`

## Training

`$ python3 main.py evaluate=False logging=train model=stofnet data_dir=./datasets/stof_chirp101_dataset th=Null rf_scale_factor=10` 


## Inference

`$ python3 main.py evaluate=True batch_size=1 etol=1 data_dir=./datasets/stof_chirp101_dataset logging=chirp_single rf_scale_factor=10`

**Note**: More information on commands and settings are found in [config.yaml](config.yaml) or [bash_scripts](bash_scripts).

## Results

<img src="https://github.com/hahnec/stofnet/blob/master/docs/chirp_plot.svg" width="750" scale="100%">
<br>
<br>
<img src="https://github.com/hahnec/stofnet/blob/master/docs/pala_plot.svg" width="750" scale="100%">
<br>
<br>

If you use this project for your work, please cite the original [paper](https://arxiv.org/pdf/2308.12009.pdf):

```
@inproceedings{stofnet,
 title={StofNet: Super-resolution Time of Flight Network}, 
 author={Christopher Hahne and Michel Hayoz and Raphael Sznitman},
 booktitle={Proceedings of the IEEE International Conference on Acoustics, Speech and Signal Processing (ICASSP)},
 year={2024},
}
```
