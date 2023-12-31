import torch
import torch.nn as nn


class SampleShuffle1D(nn.Module):
    def __init__(self, upsample_factor):
        super(SampleShuffle1D, self).__init__()
        self.upsample_factor = upsample_factor

    def forward(self, x):

        #batch_size, channels, sequence_length = x.size()
        #assert channels % self.upsample_factor == 0, "Input channels must be divisible by upsample_factor."
        #unfolded = x.unfold(2, self.upsample_factor, self.upsample_factor).permute(0, 2, 1, 3)
        #unfolded = unfolded.contiguous().view(batch_size, sequence_length * self.upsample_factor, channels // self.upsample_factor)
        #x = unfolded.permute(0, 2, 1)
        
        # https://github.com/serkansulun/pytorch-pixelshuffle1d/blob/master/pixelshuffle1d.py
        batch_size, short_channel_len, short_width = x.shape

        long_channel_len = short_channel_len // self.upsample_factor
        long_width = self.upsample_factor * short_width

        x = x.contiguous().view([batch_size, self.upsample_factor, long_channel_len, short_width])
        x = x.permute(0, 2, 3, 1).contiguous()
        x = x.view(batch_size, long_channel_len, long_width)

        return x
    