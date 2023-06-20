import torch
from torch.nn import functional as F


def nms_1d(scores, window_size):

    max_scores = torch.nn.functional.max_pool1d(scores, window_size, stride=1, padding=(window_size - 1) // 2).squeeze(0)
    suppressed = (scores == max_scores).float() * scores
    
    return suppressed


def get_maxima_positions(scores, window_size):

    suppressed = nms_1d(scores, window_size)
    indices = torch.nonzero(suppressed.squeeze(), as_tuple=False).long()

    return indices


def samples2nested_list(scores, window_size, upsample_factor=1):
    ''' caution: computationally expensive '''

    indices = get_maxima_positions(scores, window_size)

    helper_list = []
    nested_list = []
    for bidx in range(max(indices[:, 0])+1):
        for cidx in range(max(indices[:, 1])+1):
            samples = indices[(indices[:, 0]==bidx) & (indices[:, 1]==cidx), -1] / upsample_factor
            helper_list.append(samples.cpu().numpy())
        nested_list.append(helper_list)
        helper_list = []

    return nested_list


def batch_samples2coords(scores, window_size, threshold=.5, upsample_factor=1):

    scores[scores<threshold] = 0

    indices = get_maxima_positions(scores, window_size)

    # catch case where no maxima is found
    if indices.numel() == 0:
        return torch.zeros((scores.shape[0], scores.shape[1], 1), device=scores.device)

    b_max = int(max(indices[:, 0])) + 1
    c_max = int(max(indices[:, 1])) + 1
    samples = indices[:, 2].float() / upsample_factor

    # Compute the flattened indices for gather operation
    flattened_indices_2d = indices[:, 0] * c_max + indices[:, 1]
    unique_indices_2d, counts = torch.unique(flattened_indices_2d, return_counts=True)
    max_samples_per_channel = int(max(counts))

    cnts_idx = torch.vstack([torch.cat((torch.arange(0, count)+i*max_samples_per_channel, -1*torch.ones(max_samples_per_channel-count)), dim=-1) for i, count in enumerate(counts)]).long().to(scores.device)
    flattened_indices_3d = cnts_idx.flatten()[cnts_idx.flatten()>=0]

    # Create a tensor with zeros and assign the samples to their respective indices
    coords = torch.zeros((b_max, c_max, max_samples_per_channel), device=scores.device)
    coords.view(-1)[flattened_indices_3d] = samples

    return coords


def samples2coords(scores, window_size, threshold=.5, upsample_factor=1, echo_max=None):

    scores[scores<threshold] = 0
    
    indices = get_maxima_positions(scores, window_size)

    # catch case where no maxima is found
    if indices.numel() == 0:
        return torch.zeros((scores.shape[0], scores.shape[1], 1), device=scores.device)

    # Compute the flattened indices for gather operation
    c_max = int(max(indices[:, 0])) + 1
    flattened_indices_2d = indices[:, 0]
    unique_indices_2d, counts = torch.unique(flattened_indices_2d, return_counts=True)
    max_samples_per_channel = int(max(counts))

    cnts_idx = torch.vstack([torch.cat((torch.arange(0, count)+i*max_samples_per_channel, -1*torch.ones(max_samples_per_channel-count)), dim=-1) for i, count in enumerate(counts)]).long().to(scores.device)
    flattened_indices_3d = cnts_idx.flatten()[cnts_idx.flatten()>=0]

    # Create a tensor with zeros and assign the samples to their respective indices
    coords = torch.zeros((c_max, max_samples_per_channel), device=scores.device)
    coords.view(-1)[flattened_indices_3d] = indices[:, 1].float()

    # reduce number of echoes based on score amplitudes
    if echo_max and echo_max < coords.shape[-1]:
        amplitudes = get_amplitudes(scores, coords)
        coords = reduce_echoes(torch.dstack([coords, amplitudes]), echo_max=echo_max)[..., 0]

    coords /= upsample_factor

    return coords


def reduce_echoes(samples_and_amps, echo_max=100):

    echo_num = samples_and_amps.shape[1]
    channel_num = samples_and_amps.shape[-1]

    # optional: use consistent number of echoes for deterministic computation time in subsequent processes
    if echo_num > echo_max:
        # sort by maximum amplitude
        idcs = torch.argsort(samples_and_amps[..., 1], descending=True, dim=1)
        # select echoes with larger amplitude
        echoes = torch.gather(samples_and_amps, dim=1, index=idcs[..., None].repeat(1,1,channel_num))[:, :echo_max]
        # sort by time of arrival
        idcs = torch.argsort(echoes[..., 0], descending=False, dim=1)
        echoes = torch.gather(echoes, dim=1, index=idcs[..., None].repeat(1,1,channel_num))

    return echoes


def get_amplitudes(frames, samples):
    return torch.gather(frames.squeeze(), -1, torch.round(samples).long())


def samples2mask(samples, ref):
    
    empty_mask = torch.zeros_like(ref)
    samples[samples<0] = 0
    empty_mask.scatter_(2, samples, 1)
    
    # skip first sample as it serves as placeholder for NaNs
    empty_mask[..., :1] = 0

    return empty_mask
