import torch
import torch.nn as nn
import torch.nn.init as init
import torch.nn.functional as F

from utils.sample_shuffle import SampleShuffle1D


class StofNet(nn.Module):

    def __init__(self, upsample_factor=4, num_features=64, num_blocks=13, kernel_sizes=[9, 7, 3], in_channels=1, semi_global_scale=80, weights_init=False):
        super(StofNet, self).__init__()

        # dimensions
        self.num_blocks = num_blocks
        self.in_channels = in_channels
        self.num_features = num_features
        self.kernel_sizes = kernel_sizes
        self.upsample_factor = upsample_factor
        self.semi_global_scale = semi_global_scale

        # init first and last layer
        self.conv1 = nn.Conv1d(self.in_channels, self.num_features, self.kernel_sizes[0], 1, 4)
        self.conv_last = nn.Conv1d(self.num_features, self.upsample_factor, self.kernel_sizes[-1], 1, 1)

        # init semi-global block
        self.semi_global_block = SemiGlobalBlock(self.num_features, self.num_features, self.semi_global_scale) if self.semi_global_scale != 1 else None

        # init remaining layers
        for i in range(2, self.num_blocks):
            setattr(self, f'conv{i}', nn.Conv1d(self.num_features, self.num_features, self.kernel_sizes[1], 1, padding='same'))

        # shuffle feature channels to high resolution output
        self.sample_shuffle = SampleShuffle1D(self.upsample_factor)

        # indices of layers where residual connections are added and ReLU is not used
        self.residual_layers = list(range(3, self.num_blocks-1, 2))+[self.num_blocks-1, self.num_blocks]

        # initialize weights
        if weights_init: self._initialize_weights()

    def forward(self, x):

        # first layer
        x = F.relu(self.conv1(x))

        # semi-global block
        x = self.semi_global_block(x) if self.semi_global_block is not None else x

        # iterate through convolutional layers
        res, res1 = x, x
        for i in range(2, self.num_blocks-1):
            # get corresponding convolutional layer
            conv = getattr(self, f'conv{i}')
            # pass data to layer considering residual connection
            x = torch.add(res, conv(x)) if i in self.residual_layers else F.leaky_relu(conv(x))
            # store residuals
            if i in self.residual_layers: res = x

        # second last layer
        conv = getattr(self, f'conv{i+1}')
        x = torch.add(res1, conv(x))

        # last layer and shuffling
        x = self.sample_shuffle(self.conv_last(x))

        return x

    def _initialize_weights(self):

        for i in range(1, self.num_blocks):
            if i not in self.residual_layers:
                init.orthogonal(getattr(self, f'conv{i}').weight, init.calculate_gain('relu'))
            else:
                init.orthogonal(getattr(self, f'conv{i}').weight)
        
        init.orthogonal(self.conv_last.weight)


class SemiGlobalBlock(nn.Module):
    def __init__(self, in_channels, out_channels, sample_scale=2, kernel_size=5):
        super(SemiGlobalBlock, self).__init__()

        self.sample_scale = sample_scale
        self.feat_scale = max(1, sample_scale//10)

        # Contracting path
        self.contract_conv = nn.Conv1d(in_channels, self.feat_scale*out_channels, kernel_size=kernel_size, stride=1, padding=kernel_size//2)
        self.contract_relu = nn.LeakyReLU()
        #self.attention = AttentionBlock(2578//2-1, 2578//2-1)
        self.contract_pool = nn.MaxPool1d(kernel_size=sample_scale, stride=sample_scale)

        # Expanding path
        self.expand_conv = nn.Conv1d(self.feat_scale*out_channels, out_channels, kernel_size=kernel_size, stride=1, padding=kernel_size//2)
        self.expand_relu = nn.LeakyReLU()
        self.expand_upsample = nn.Upsample(scale_factor=sample_scale, mode='nearest')

    def forward(self, x):
        # Contracting path
        x_scale = self.contract_conv(x)
        x_scale = self.contract_relu(x_scale)
        #x_scale = self.attention(x_scale)  # Apply attention
        x_scale = self.contract_pool(x_scale)

        # Expanding path
        x_scale = self.expand_conv(x_scale)
        x_scale = self.expand_relu(x_scale)
        x_scale = self.expand_upsample(x_scale)

        # Adjust padding for correct output size
        padding = max(0, x.size(-1) - x_scale.size(-1))
        x_scale = nn.functional.pad(x_scale, (padding // 2, padding // 2))

        # Skip connection via addition
        x = torch.add(x, x_scale)
        
        return x
