import torch
import torch.nn as nn


def hilbert_transform(y):
    ''' inspired by https://stackoverflow.com/questions/56380536/hilbert-transform-in-python#56381976 '''

    # fft length
    n = y.shape[-1]
    # forward Fourier transform
    f = torch.fft.fft(y, dim=-1)
    # number of elements to zero out
    m = n - n//2 - 1
    # zero out negative frequency components
    f[..., n//2+1:] = torch.zeros(m, device=f.device, dtype=f.dtype)
    # double fft energy except for dc component
    f[..., 1:n//2] *= 2
    # inverse Fourier transform
    v = torch.fft.ifft(f, dim=-1)

    return v


class HilbertTransform(nn.Module):
    def __init__(self, upsample_factor):
        super(HilbertTransform, self).__init__()

    def forward(self, x):
        return hilbert_transform(x)