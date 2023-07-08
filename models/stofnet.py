import torch
import torch.nn as nn
import torch.nn.init as init
import torch.nn.functional as F

from utils.hilbert import hilbert_transform, HilbertTransform
from utils.sample_shuffle import SampleShuffle1D


class StofNet(nn.Module):

    def __init__(self, upsample_factor, feat_channels=64, fs=None, hilbert_opt=True, concat_oscil=True):
        super(StofNet, self).__init__()
    
        # input signal handling
        if fs is not None:
            from models.sincnet import SincConv_fast
            self.sinc_filter = SincConv_fast(1, kernel_size=129, sample_rate=fs, in_channels=1, padding=64)
        self.hilbert = HilbertTransform(concat_oscil=concat_oscil) if hilbert_opt else None
        in_channels = 2 if hilbert_opt and concat_oscil else 1

        # init first and last layer
        self.conv1 = nn.Conv1d(in_channels, feat_channels, 9, 1, 4)
        self.conv13 = nn.Conv1d(feat_channels, upsample_factor, 3, 1, 1)

        # init semi-global block
        self.semi_global_block = SemiGlobalBlock(feat_channels, feat_channels, 80)

        # init remaining layers
        for i in range(2, 13):
            setattr(self, f'conv{i}', nn.Conv1d(feat_channels, feat_channels, 7, 1, padding='same'))

        # shuffle feature channels to high resolution output
        self.sample_shuffle = SampleShuffle1D(upsample_factor)

        # indices of layers where residual connections are added and ReLU is not used
        self.residual_layers = [3, 5, 7, 9, 11, 12, 13]

    def forward(self, x):
        
        # input signal handling
        x = self.sinc_filter(x) if self.sinc_filter else x
        x = self.hilbert(x) if self.hilbert is not None else x
        
        # first layer
        x = F.relu(self.conv1(x))

        # semi-global block
        x = self.semi_global_block(x)

        # iterate through convolutional layers
        res, res1 = x, x
        for i in range(2, 12):
            # get corresponding convolutional layer
            conv = getattr(self, f'conv{i}')
            # pass data to layer considering residual connection
            x = torch.add(res, conv(x)) if i in self.residual_layers else F.leaky_relu(conv(x))
            # store residuals
            if i in self.residual_layers: res = x

        # second last layer
        x = torch.add(res1, self.conv12(x))

        # last layer and shuffling
        x = self.sample_shuffle(self.conv13(x))
        
        return x

    def _initialize_weights(self):

        return None

        for i in range(1, 14):
            if i not in self.residual_layers:
                init.orthogonal(getattr(self, f'conv{i}').weight, init.calculate_gain('relu'))
            else:
                init.orthogonal(getattr(self, f'conv{i}').weight)


class SemiGlobalBlock(nn.Module):
    def __init__(self, in_channels, out_channels, sample_scale=2):
        super(SemiGlobalBlock, self).__init__()

        self.sample_scale = sample_scale
        self.feat_scale = max(1, sample_scale//10)
        
        # Contracting path
        self.contract_conv = nn.Conv1d(in_channels, self.feat_scale*out_channels, kernel_size=5, stride=1, padding=1)
        self.contract_relu = nn.LeakyReLU()
        self.contract_pool = nn.MaxPool1d(kernel_size=sample_scale, stride=sample_scale)
        
        # Expanding path
        self.expand_conv = nn.Conv1d(self.feat_scale*out_channels, out_channels, kernel_size=5, stride=1, padding=1)
        self.expand_relu = nn.LeakyReLU()
        self.expand_upsample = nn.Upsample(scale_factor=sample_scale, mode='nearest')
        
    def forward(self, x):
        # Contracting path
        x_scale = self.contract_conv(x)
        x_scale = self.contract_relu(x_scale)
        x_scale = self.contract_pool(x_scale)
        
        # Expanding path
        x_scale = self.expand_conv(x_scale)
        x_scale = self.expand_relu(x_scale)
        x_scale = self.expand_upsample(x_scale)
        
        # Adjust padding for correct output size
        padding = max(0, x.size(-1) - x_scale.size(-1))
        x_scale = nn.functional.pad(x_scale, (padding // 2, padding // 2))
        
        # Skip connection via concatenation
        #x = torch.cat((x, x_scale), dim=1)
        x = torch.add(x, x_scale)
        
        return x

