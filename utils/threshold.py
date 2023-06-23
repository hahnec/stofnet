import torch
import numpy as np
from sklearn.metrics import roc_curve


def find_threshold(masks_pred, masks_true, window_size=None):

    if window_size:
        window_size = window_size//2*2+1
        max_scores = torch.nn.functional.max_pool1d(masks_pred, window_size, stride=1, padding=(window_size - 1) // 2).squeeze(0)
        suppressed = (masks_pred == max_scores).float() * masks_pred
    else:
        suppressed = masks_pred

    fpr, tpr, thresholds = roc_curve(masks_true.float().numpy().flatten(), suppressed.flatten().float().numpy())

    # calculate the g-mean for each threshold
    gmeans = (tpr * (1-fpr))**.5
    th_idx = np.argmax(gmeans)
    threshold = thresholds[th_idx]

    return threshold