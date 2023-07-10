import torch
import torch.nn as nn

from utils.sample_shuffle import SampleShuffle1D


class ResidualBlock(nn.Module):
    def __init__(self, channels):
        super(ResidualBlock, self).__init__()
        self.conv1 = nn.Conv1d(channels, channels, kernel_size=3, stride=1, padding=1)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv1d(channels, channels, kernel_size=3, stride=1, padding=1)
        
    def forward(self, x):
        residual = x
        out = self.relu(self.conv1(x))
        out = self.conv2(out)
        out += residual
        return out


class EDSR_1D(nn.Module):
    def __init__(self, num_channels=1, num_features=64, num_blocks=8, upscale_factor=4):
        super(EDSR_1D, self).__init__()
        self.conv_input = nn.Conv1d(num_channels, num_features, kernel_size=3, stride=1, padding=1)
        self.relu = nn.ReLU(inplace=True)
        self.residual_blocks = nn.ModuleList([ResidualBlock(num_features) for _ in range(num_blocks)])
        self.conv_mid = nn.Conv1d(num_features, num_features, kernel_size=3, stride=1, padding=1)
        self.upscale = SampleShuffle1D(upscale_factor)
        self.conv_output = nn.Conv1d(num_features//upscale_factor, num_channels, kernel_size=3, stride=1, padding=1)
        
    def forward(self, x):
        out = self.relu(self.conv_input(x))
        residual = out
        
        for block in self.residual_blocks:
            out = block(out)
        
        out = self.conv_mid(out)
        out += residual
        
        out = self.upscale(out)
        out = self.conv_output(out)
        
        return out
