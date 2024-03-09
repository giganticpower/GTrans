import torch
import torch.nn.functional as F
import numpy as np
from scipy.ndimage import gaussian_filter
import math


def cal_loss(fs_list, ft_list):   # 原来的cal_loss
    t_loss = 0
    N = len(fs_list)
    for i in range(N):
        fs = fs_list[i]
        ft = ft_list[i]
        _, _, h, w = fs.shape
        fs_norm = F.normalize(fs, p=2)
        ft_norm = F.normalize(ft, p=2)

        f_loss = 0.5 * (ft_norm - fs_norm)**2
        f_loss = f_loss.sum() / (h*w)
        t_loss += f_loss

    return t_loss / N

def cal_loss_cos(fs_list, ft_list):   # 原来的cal_loss
    t_loss = 0
    N = len(fs_list)
    similarity_loss = torch.nn.CosineSimilarity()
    criterion = torch.nn.MSELoss()
    for i in range(N):
        fs = fs_list[i]
        ft = ft_list[i]
        _, _, h, w = fs.shape
        fs_norm = F.normalize(fs, p=2)
        ft_norm = F.normalize(ft, p=2)
        loss_cr = criterion(fs_norm, ft_norm)
        index_cr = loss_cr * 100
        loss_cos = torch.mean(1 - similarity_loss(ft.view(ft.shape[1], -1), fs.view(fs.shape[1], -1)))
        index = 2 * (loss_cos * 10) * index_cr / (loss_cos * 10 + index_cr)
        # print(index)
        f_loss = index * (ft_norm - fs_norm)**2
        f_loss = f_loss.sum() / (h*w)
        t_loss += f_loss

    return t_loss / N


def cal_anomaly_maps(fs_list, ft_list, out_size):
    anomaly_map = 0
    map_list = []
    similarity_loss = torch.nn.CosineSimilarity()
    criterion = torch.nn.MSELoss()
    for i in range(len(ft_list)):
        fs = fs_list[i]
        ft = ft_list[i]

        fs_norm = F.normalize(fs, p=2)
        ft_norm = F.normalize(ft, p=2)
        _, _, h, w = fs.shape
        loss_cr = criterion(fs_norm, ft_norm)
        loss_cos = torch.mean(1 - similarity_loss(ft.view(ft.shape[1], -1), fs.view(fs.shape[1], -1)))
        index_cr = loss_cr * 100
        index_cos = loss_cos * 10
        # index = index_cr * index_cos
        index = 2 * index_cos * index_cr / (index_cos + index_cr)
        # index = loss_cr * 10 + loss_cos
        # print(index_cr, index_cos, index)
        a_map = (index * (ft_norm - fs_norm)**2) / (h*w)
        a_map = a_map.sum(1, keepdim=True)

        a_map = F.interpolate(a_map, size=out_size, mode='bilinear', align_corners=False)
        map_list.append(a_map)

    anomaly_map, map_list = map_select(map_list)

    for i, map in enumerate(map_list):
        if i == 0:
            inter_map = map
        else:
            inter_map = torch.cat([inter_map, map], dim=1)

    anomaly_map = anomaly_map.squeeze().cpu().numpy()
    inter_map = inter_map.squeeze().cpu().numpy()
    for i in range(anomaly_map.shape[0]):
        anomaly_map[i] = gaussian_filter(anomaly_map[i], sigma=4)

    return anomaly_map, inter_map

def map_select(map_list):

    map_12 = 2 * map_list[1] * map_list[2]
    map_02 = 2 * map_list[0] * map_list[2]
    map_01 = 2 * map_list[0] * map_list[1]
    map_add = map_list[0] + map_list[1] + map_list[2]
    map_list.append(map_01)
    map_list.append(map_02)
    map_list.append(map_12)
    # anomaly_map =  map_02 + map_12
    # anomaly_map = map_add

    # base
    # anomaly_map = map_list[0] + map_list[1] + map_list[2]

    # 02*12
    anomaly_map = map_02 * map_12

    # 0*1*2  01,02,12
    # anomaly_map = map_add * (map_02 + map_12)

    # anomaly_map = map_01 * map_12 * map_12 * map_02

    return anomaly_map, map_list


