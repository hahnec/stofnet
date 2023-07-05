import torch
import torch.nn as nn
import torch.nn.init as init
import torch.nn.functional as F

from utils.hilbert import hilbert_transform, HilbertTransform
from utils.sample_shuffle import SampleShuffle1D


class StofNet(nn.Module):

    def __init__(self, upsample_factor, hilbert_opt=True, concat_oscil=True):
        super(StofNet, self).__init__()
    
        # input signal handling
        self.hilbert = HilbertTransform(concat_oscil=concat_oscil) if hilbert_opt else None
        in_channels = 2 if hilbert_opt and concat_oscil else 1

        # init first and last layer
        self.conv1 = nn.Conv1d(in_channels, 64, 9, 1, 4)
        self.conv13 = nn.Conv1d(64, upsample_factor, 3, 1, 1)

        # init remaining layers
        for i in range(2, 13):
            setattr(self, f'conv{i}', nn.Conv1d(64, 64, 7, 1, 'same'))

        # shuffle feature channels to high resolution output
        self.sample_shuffle = SampleShuffle1D(upsample_factor)

        # indices of layers where residual connections are added and ReLU is not used
        self.residual_layers = [3, 5, 7, 9, 11, 12, 13]

    def forward(self, x):
        
        # input signal handling
        x = self.hilbert(x) if self.hilbert is not None else x
        
        # first layer
        x = F.relu(self.conv1(x))
        res, res1 = x, x

        # iterate through convolutional layers
        for i in range(2, 12):
            # get corresponding convolutional layer
            conv = getattr(self, f'conv{i}')
            # pass data to layer considering residual connection
            x = torch.add(res, conv(x)) if i in self.residual_layers else F.relu(conv(x))
            # store residuals
            if i in self.residual_layers: res = x

        # second last layer
        x = torch.add(res1, self.conv12(x))

        # last layer and shuffling
        x = self.sample_shuffle(self.conv13(x))
        
        return x

    def _initialize_weights(self):

        for i in range(1, 14):
            if i not in self.residual_layers:
                init.orthogonal(getattr(self, f'conv{i}').weight, init.calculate_gain('relu'))
            else:
                init.orthogonal(getattr(self, f'conv{i}').weight)
