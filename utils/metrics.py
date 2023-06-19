import torch
import numpy as np


jaccard_index = lambda tp, fn, fp: tp/(fn+tp+fp)*100


def toa_rmse(gt_samples, es_samples, tol=1):

    distances = abs(gt_samples[..., None] - es_samples[:, :, None, :].repeat(1, 1, gt_samples.shape[-1], 1))
    mins, args = distances.min(-1)

    tps = (mins <=tol).sum(-1)
    fns = (mins > tol).sum(-1)
    fps = (es_samples > 0).sum(-1) - tps
    jaccards = jaccard_index(tps, fns, fps)
    precisions = tps/(fps+tps) * 100
    recalls = tps/(fns+tps) * 100

    mins[mins>tol] = torch.nan
    mes = torch.nanmean(mins, dim=-1)

    return mes, precisions, recalls, jaccards, tps.float(), fps.float(), fns.float()
