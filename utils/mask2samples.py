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


def samples2coords(scores, window_size, upsample_factor=1):

    indices = get_maxima_positions(scores, window_size)
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
    nested_tensor = torch.zeros((b_max, c_max, max_samples_per_channel), device=scores.device)
    nested_tensor.view(-1)[flattened_indices_3d] = samples

    return nested_tensor

def samples2mask(samples, ref):
    
    empty_mask = torch.zeros_like(ref)
    samples[samples<0] = 0
    empty_mask.scatter_(2, samples, 1)
    
    # skip first sample as it serves as placeholder for NaNs
    empty_mask[..., :1] = 0

    return empty_mask
