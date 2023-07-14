import torch
import torch.nn as nn
import torch.nn.functional as F

from utils.sample_shuffle import SampleShuffle1D


class ESPCN_1D(nn.Module):
    def __init__(self, upscale_factor):
        super(ESPCN_1D, self).__init__()

        self.conv1 = nn.Conv1d(1, 64, 5, 1, 2)
        self.conv2 = nn.Conv1d(64, 32, 3, 1, 1)
        self.conv3 = nn.Conv1d(32, upscale_factor, 3, 1, 1)
        self.sample_shuffle = SampleShuffle1D(upscale_factor)

        # initial model weights
        for module in self.modules():
            if isinstance(module, nn.Conv1d):
                if module.in_channels == 32:
                    nn.init.normal_(module.weight.data,
                                    0.0,
                                    0.001)
                    nn.init.zeros_(module.bias.data)
                else:
                    nn.init.normal_(module.weight.data,
                                    0.0,
                                    (2 / (module.out_channels * module.weight.data[0][0].numel()))**.5)
                    nn.init.zeros_(module.bias.data)

    def forward(self, x):
        x = F.tanh(self.conv1(x))
        x = F.tanh(self.conv2(x))
        x = F.sigmoid(self.sample_shuffle(self.conv3(x)))

        return x
