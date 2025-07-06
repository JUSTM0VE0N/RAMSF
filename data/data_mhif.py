#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time: 2025/6/23 22:09
# @Author: laxiojustmoveon


# here put the import lib
import torch.utils.data as data
import torch
import h5py
import cv2
import numpy as np

def get_edge(data):  # for training
    rs = np.zeros_like(data)
    N = data.shape[0]
    for i in range(N):
        if len(data.shape) == 3:
            rs[i, :, :] = data[i, :, :] - cv2.boxFilter(data[i, :, :], -1, (5, 5))
        else:
            rs[i, :, :, :] = data[i, :, :, :] - cv2.boxFilter(data[i, :, :, :], -1, (5, 5))
    return rs

class Load_Trainset(data.Dataset):
    def __init__(self, file_path):
        super(Load_Trainset, self).__init__()
        dataset = h5py.File(file_path, mode='r')

        self.gt = dataset.get("GT")  # NxCxHxW
        self.ms = dataset.get("HSI_up")
        self.lms = dataset.get("LRHSI")
        self.pan = dataset.get("RGB")

    def __getitem__(self, index):
        gt = torch.from_numpy(self.gt[index, :, :, :] / 2047).float()
        lms = torch.from_numpy(self.lms[index, :, :, :] / 2047).float()
        ms = torch.from_numpy(self.ms[index, :, :, :] / 2047).float()
        pan = torch.from_numpy(self.pan[index, :, :, :] / 2047).float()
        return gt, lms, ms, pan

    def __len__(self):
        return self.gt.shape[0]

# load MHIF testing set
class Load_RRTset(data.Dataset):
    def __init__(self, file_path):
        super(Load_RRTset, self).__init__()
        dataset = h5py.File(file_path, mode='r')

        self.gt = dataset.get("GT")  # NxCxHxW
        self.ms = dataset.get("HSI_up")
        self.lms = dataset.get("LRHSI")
        self.pan = dataset.get("RGB")

    def __getitem__(self, index):
        gt = torch.from_numpy(self.gt[index, :, :, :]).float()
        lms = torch.from_numpy(self.lms[index, :, :, :]).float()
        ms = torch.from_numpy(self.ms[index, :, :, :]).float()
        pan = torch.from_numpy(self.pan[index, :, :, :]).float()
        return gt, lms, ms, pan

    def __len__(self):
        return self.gt.shape[0]