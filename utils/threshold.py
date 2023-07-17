import torch
import numpy as np
from sklearn.metrics import roc_curve

from utils.mask2samples import nms_1d


def find_threshold(masks_pred, masks_true, window_size, norm_opt=False):

    masks_norm = masks_pred.clone().detach()

    # normalize prediction
    if norm_opt:
        masks_norm -= masks_norm.min()
        masks_norm /= masks_norm.max()
    else:
        masks_norm = masks_pred

    mask_norm = nms_1d(mask_norm, window_size)
    
    # true label value
    max_val = float(masks_true.max()) if float(masks_true.max()) != 0 else 1

    # compute ROC curve results
    fpr, tpr, thresholds = roc_curve(masks_true.float().numpy().flatten(), masks_norm.flatten().float().numpy(), pos_label=max_val)

    # calculate the g-mean for each threshold
    gmeans = (tpr * (1-fpr))**.5
    th_idx = np.argmax(gmeans)
    threshold = thresholds[th_idx]

    return threshold
