import torch
from torch.nn.functional import conv1d
from torch.distributions import Normal

from utils.hilbert import hilbert_transform


def grad_peak_detect(data, grad_step: int=None, threshold: float=None, ival_smin: int=None, ival_smax: int=None):

    batch_size = data.shape[0]

    # hilbert data preparation
    grad_step = grad_step if grad_step is not None else 2    
    grad_data = torch.gradient(data, spacing=grad_step, dim=-1)[0]
    grad_data = gaussian_filter_1d(grad_data, sigma=(grad_step*2-1)/6)

    # parameter init (defaults are heuristics)
    thres_pos = threshold if threshold is not None else (grad_data.std()**16)*1.2e13
    thres_neg = -thres_pos/4
    ival_list = [ival_smin, ival_smax] if ival_smin is not None and ival_smax is not None else [grad_step//2, grad_step*3]

    # gradient analysis
    grad_plus = grad_data > thres_pos
    grad_minu = grad_data < thres_neg

    # get potential echo positions
    peak_plus = torch.diff((grad_plus==1).int(), axis=-1)
    peak_minu = torch.diff((grad_minu==1).int(), axis=-1)
    args_plus = torch.argwhere(peak_plus==1)#.flatten()
    args_minu = torch.argwhere(peak_minu==1)#.flatten()

    # hysteresis via interval analysis from differences between start and stop indices
    peak_list = []
    max_len = 0
    for i in range(batch_size):
        ap = args_plus[args_plus[:, 0]==i, 1].unsqueeze(1)#.float()
        am = args_minu[args_minu[:, 0]==i, 1].unsqueeze(0)#.float()
        if ap.numel() == 0 or am.numel() == 0:
            peak_list.append(torch.tensor([], device=data.device))
            continue

        dmat = am - ap.repeat((1, am.shape[1]))
        dmat[dmat<0] = 2**32 # constraint that only differences for ap occuring before am are valid
        echo_peak_idcs = torch.argmin(abs(dmat), dim=0)
        candidates = torch.hstack([ap[echo_peak_idcs], am.T])

        # constraint that only differences for ap occuring before am are valid
        gaps = candidates.diff(1).squeeze()
        candidates = candidates[(gaps>ival_list[0]) & (gaps<ival_list[1]), :]

        # ensure candidates consists of 2 dimensions
        candidates = candidates.squeeze(0) if len(candidates.shape) > 2 else candidates

        if candidates.numel() == 0:
            return torch.tensor([[], [], []])
        
        # gradient peak uniqueness constraint
        apu, uniq_idcs = torch.unique(candidates[:, 0], return_inverse=True)
        amu = candidates[torch.diff(uniq_idcs.flatten(), prepend=torch.tensor([-1], device=apu.device))>0, 1]
        peaks = torch.stack([apu.flatten(), amu.flatten()]).T

        peak_list.append(peaks)
        if len(peaks) > max_len: max_len = len(peaks)

    # convert list to tensor: batch_size x echo_num x (xy)-coordinates
    batch_peaks = torch.tensor([torch.hstack([echoes, data[i, echoes[:, 1][:, None]]]).tolist()+[[0,0,0],]*(max_len-len(echoes)) if len(echoes) > 0 else [[0,0,0],]*max_len for i, echoes in enumerate(peak_list)], dtype=data.dtype, device=data.device)

    return batch_peaks


def gaussian_kernel_1d(sigma: float, num_sigmas: float = 3.) -> torch.Tensor:
    
    radius = int(num_sigmas * sigma)+1 # ceil
    support = torch.arange(-radius, radius + 1, dtype=torch.float64)
    kernel = Normal(loc=0, scale=sigma).log_prob(support).exp_()
    return kernel.mul_(1 / kernel.sum())


def gaussian_fn(M, std):

    n = torch.arange(0, M) - (M - 1.0) / 2.0
    sig2 = 2 * std * std
    w = torch.exp(-n ** 2 / sig2)
    w /= w.sum()

    return w


def gaussian_filter_1d(data: torch.Tensor, sigma: float) -> torch.Tensor:
    
    kernel_1d = gaussian_kernel_1d(sigma).to(data.device, dtype=data.dtype)
    padding = len(kernel_1d) // 2
    data = data.unsqueeze(1) if len(data.shape) == 2 else data
    data = conv1d(data, weight=kernel_1d.view(1, 1, -1), padding=padding)

    return data.squeeze(1)


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
