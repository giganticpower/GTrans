from sklearn.metrics import auc
from sklearn.metrics import roc_auc_score, average_precision_score
from skimage import measure
import pandas as pd
import numpy as np


def auc_pro_curve(masks, scores, expect_fpr=0.3, max_step=5000):
    # binary masks
    # masks[masks <= 0.5] = 0
    # masks[masks > 0.5] = 1
    masks = masks.astype(np.bool)

    # per region overlap and per image iou
    max_th = scores.max()
    min_th = scores.min()
    delta = (max_th - min_th) / max_step

    pros_mean = []
    pros_std = []
    threds = []
    fprs = []
    binary_score_maps = np.zeros_like(scores, dtype=np.bool)
    for step in range(max_step):
        thred = max_th - step * delta
        # segmentation
        binary_score_maps[scores <= thred] = 0
        binary_score_maps[scores > thred] = 1

        pro = []  # per region overlap
        iou = []  # per image iou
        # pro: find each connected gt region, compute the overlapped pixels between the gt region and predicted region
        # iou: for each image, compute the ratio, i.e. intersection/union between the gt and predicted binary map
        for i in range(len(binary_score_maps)):  # for i th image
            # pro (per region level)
            label_map = measure.label(masks[i], connectivity=2)
            props = measure.regionprops(label_map)
            for prop in props:
                x_min, y_min, x_max, y_max = prop.bbox  # find the bounding box of an anomaly region
                cropped_pred_label = binary_score_maps[i][x_min:x_max, y_min:y_max]
                cropped_mask = masks[i][x_min:x_max, y_min:y_max]
                intersection = np.logical_and(cropped_pred_label, cropped_mask).astype(np.float32).sum()
                pro.append(intersection / prop.area)
        # against steps and average metrics on the testing data

        pros_mean.append(np.array(pro).mean())
        pros_std.append(np.array(pro).std())
        # fpr for pro-auc
        masks_neg = ~masks
        fpr = np.logical_and(masks_neg, binary_score_maps).sum() / masks_neg.sum()
        fprs.append(fpr)
        threds.append(thred)

    # as array
    threds = np.array(threds)
    pros_mean = np.array(pros_mean)
    pros_std = np.array(pros_std)
    fprs = np.array(fprs)

    # default 30% fpr vs pro, pro_auc
    idx = fprs <= expect_fpr  # find the indexs of fprs that is less than expect_fpr (default 0.3)
    fprs_selected = fprs[idx]
    # rescale fpr [0,0.3] -> [0, 1]
    fprs_selected = (fprs_selected - fprs_selected.min()) / (fprs_selected.max() - fprs_selected.min())
    pros_mean_selected = pros_mean[idx]
    pro_auc_score = auc(fprs_selected, pros_mean_selected)
    return pro_auc_score
