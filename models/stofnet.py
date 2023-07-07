import torch
import torch.nn as nn
import torch.nn.init as init
import torch.nn.functional as F

from utils.hilbert import hilbert_transform, HilbertTransform
from utils.sample_shuffle import SampleShuffle1D
from models.sincnet import SincConv_fast


class StofNet(nn.Module):

    def __init__(self, fs, upsample_factor, hilbert_opt=True, concat_oscil=True):
        super(StofNet, self).__init__()
    
        # input signal handling
        self.hilbert = HilbertTransform(concat_oscil=concat_oscil) if hilbert_opt else None
        in_channels = 2 if hilbert_opt and concat_oscil else 1

        # init first and last layer
        self.conv1 = nn.Conv1d(in_channels, 64, 9, 1, 4)
        self.conv13 = nn.Conv1d(64, upsample_factor, 3, 1, 1)

        self.sinc_layer = SincConv_fast(1, kernel_size=129, sample_rate=fs, in_channels=1, padding=64)
        self.semi_global_block = SemiGlobalBlock(64, 64, 80)

        # init remaining layers
        for i in range(2, 13):
            setattr(self, f'conv{i}', nn.Conv1d(64, 64, 7, 1, padding='same'))

        # shuffle feature channels to high resolution output
        self.sample_shuffle = SampleShuffle1D(upsample_factor)

        # indices of layers where residual connections are added and ReLU is not used
        self.residual_layers = [3, 5, 7, 9, 11, 12, 13]

    def forward(self, x):
        
        # input signal handling
        x = self.sinc_layer(x)
        x = self.hilbert(x) if self.hilbert is not None else x
        

        # first layer
        x = F.relu(self.conv1(x))
        res, res1 = x, x
        
        x = self.semi_global_block(x)

        # iterate through convolutional layers
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

        for i in range(1, 14):
            if i not in self.residual_layers:
                init.orthogonal(getattr(self, f'conv{i}').weight, init.calculate_gain('relu'))
            else:
                init.orthogonal(getattr(self, f'conv{i}').weight)


class SemiGlobalBlock(nn.Module):
    def __init__(self, in_channels, out_channels, sample_scale=2):
        super(SemiGlobalBlock, self).__init__()

        self.sample_scale = sample_scale
        self.feat_scale = sample_scale//10
        
        # Contracting path
        self.contract_conv = nn.Conv1d(in_channels, self.feat_scale*out_channels, kernel_size=3, stride=1, padding=1)
        self.contract_relu = nn.LeakyReLU()
        self.contract_pool = nn.MaxPool1d(kernel_size=sample_scale, stride=sample_scale)
        
        # Expanding path
        self.expand_conv = nn.Conv1d(self.feat_scale*out_channels, out_channels, kernel_size=3, stride=1, padding=1)
        self.expand_relu = nn.LeakyReLU()
        self.expand_upsample = nn.Upsample(scale_factor=sample_scale, mode='nearest')
        
    def forward(self, x):
        # Contracting path
        x_contract = self.contract_conv(x)
        x_contract = self.contract_relu(x_contract)
        x_contract_pooled = self.contract_pool(x_contract)
        
        # Expanding path
        x_expand = self.expand_conv(x_contract_pooled)
        x_expand = self.expand_relu(x_expand)
        x_expand = self.expand_upsample(x_expand)
        
        # Adjust padding for correct output size
        padding = max(0, x.size(-1) - x_expand.size(-1))
        x_expand = nn.functional.pad(x_expand, (padding // 2, padding // 2))
        
        # Skip connection via concatenation
        #x_out = torch.cat((x_padded, x_expand), dim=1)
        x_out = torch.add(x, x_expand)
        
        return x_out
