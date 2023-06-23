import torch
import numpy as np
from sklearn.metrics import roc_curve

from utils.mask2samples import nms_1d


def find_threshold(masks_pred, masks_true):

    # normalize prediction
    masks_norm = masks_pred.detach().clone()
    masks_norm -= masks_norm.min()
    masks_norm /= masks_norm.max()
    
    # compute ROC curve results
    fpr, tpr, thresholds = roc_curve(masks_true.float().numpy().flatten(), masks_norm.flatten().float().numpy())

    # calculate the g-mean for each threshold
    gmeans = (tpr * (1-fpr))**.5
    th_idx = np.argmax(gmeans)
    threshold = thresholds[th_idx]

    return threshold