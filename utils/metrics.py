import torch
import numpy as np


jaccard_index = lambda tp, fn, fp: tp/(fn+tp+fp)*100
remove_invalids = lambda x: x[(x!=0) & (~torch.isnan(x)) & (~torch.isinf(x))]


def toa_rmse(gt_samples, es_samples, tol=1):

    # initialize output variables
    batch_dim = gt_samples.shape[0]
    mes, tps, fps, fns = [torch.zeros(batch_dim, device=gt_samples.device) for _ in range(4)]

    # iterate through batch dimension (samples are of unequal size per batch)
    for batch_idx in range(batch_dim):

        # remove Zeros and NaNs and Infs
        gt_valid = remove_invalids(gt_samples[batch_idx])
        es_valid = remove_invalids(es_samples[batch_idx])

        # replicate estimates to get distances
        distances = abs(gt_valid[:, None] - es_valid[None, :].repeat(gt_valid.shape[-1], 1))
        mins, args = distances.min(-1)

        # compute metrics
        mes[batch_idx] = torch.mean(mins[mins<=tol], dim=-1)
        tps[batch_idx] = (mins <=tol).sum(-1).float()
        fns[batch_idx] = (mins > tol).sum(-1).float()
        fps[batch_idx] = es_valid.shape[-1] - tps[batch_idx]

    # compute metrics (vectorized)
    jaccards = jaccard_index(tps, fns, fps)
    precisions = tps/(fps+tps) * 100
    recalls = tps/(fns+tps) * 100

    return mes, precisions, recalls, jaccards, tps, fps, fns
