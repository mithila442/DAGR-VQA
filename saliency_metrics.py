import numpy as np
from sklearn.metrics import roc_auc_score

def compute_auc_judd(saliency_map, fixation_map):
    saliency = saliency_map.flatten()
    labels = (fixation_map.flatten() > 0).astype(np.uint8)
    if labels.sum() == 0:
        return 0.0
    return roc_auc_score(labels, saliency)

def compute_nss(saliency_map, fixation_map):
    if fixation_map.sum() == 0:
        return 0.0
    saliency_norm = (saliency_map - saliency_map.mean()) / (saliency_map.std() + 1e-8)
    return np.mean(saliency_norm[fixation_map == 1])

def compute_cc(saliency_map, gt_map):
    sal = saliency_map - np.mean(saliency_map)
    gt = gt_map - np.mean(gt_map)
    numerator = np.sum(sal * gt)
    denominator = np.sqrt(np.sum(sal ** 2) * np.sum(gt ** 2))
    return numerator / (denominator + 1e-8)

def compute_kldiv(saliency_map, gt_map):
    sal = saliency_map / (np.sum(saliency_map) + 1e-8)
    gt = gt_map / (np.sum(gt_map) + 1e-8)
    return np.sum(gt * np.log((gt + 1e-8) / (sal + 1e-8)))
