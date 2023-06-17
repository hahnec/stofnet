import torch


def nms_1d(scores, window_size):

    max_scores = torch.nn.functional.max_pool1d(scores, window_size, stride=1, padding=(window_size - 1) // 2).squeeze(0)
    suppressed = (scores == max_scores).float() * scores
    
    return suppressed


def get_maxima_positions(scores, window_size):

    suppressed = nms_1d(scores, window_size)
    indices = torch.nonzero(suppressed.squeeze(), as_tuple=False).long()

    return indices


def samples2nested_list(scores, window_size, upsample_factor=1):

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

def samples2mask(samples, ref):
    
    empty_mask = torch.zeros_like(ref)
    samples[samples<0] = 0
    empty_mask.scatter_(2, samples, 1)
    
    # skip first sample as it serves as placeholder for NaNs
    empty_mask[..., :1] = 0

    return empty_mask
