import torch
from multimodal_emg.util.peak_detect import grad_peak_detect

from utils.hilbert import hilbert_transform


def toa_detect(frame, threshold=1, rescale_factor=1, echo_max=float('inf')):

    # extract ToAs
    hilbert_data = abs(hilbert_transform(frame))
    echoes = grad_peak_detect(hilbert_data, grad_step=rescale_factor//6*5, ival_smin=rescale_factor, ival_smax=50*rescale_factor, threshold=threshold)
    echo_num = echoes.shape[1]

    # optional: use consistent number of echoes for deterministic computation time in subsequent processes
    if echo_num > echo_max:
        # sort by maximum amplitude
        idcs = torch.argsort(echoes[..., -1], descending=True, dim=1)
        # select echoes with larger amplitude
        echoes = torch.gather(echoes, dim=1, index=idcs[..., None].repeat(1,1,3))[:, :echo_max]
        # sort by time of arrival
        idcs = torch.argsort(echoes[..., 1], descending=False, dim=1)
        echoes = torch.gather(echoes, dim=1, index=idcs[..., None].repeat(1,1,3))

    return echoes


class GradPeak(torch.nn.Module):
    def __init__(self, threshold=1, rescale_factor=1, echo_max=float('inf')):
        super(GradPeak, self).__init__()

        self._fun = lambda x: toa_detect(x, threshold=threshold, rescale_factor=rescale_factor, echo_max=echo_max)

    def forward(self, x):
        
        # get echoes as start, peak and amplitude in that order of last dimension
        echoes = self._fun(x.squeeze())
        
        # return peaks
        return echoes[..., 1]
