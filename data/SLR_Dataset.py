# Definition for Dataset
# Auto divided into train/validation/test dataset (used function-"getThreeSet()")
# Easy change dataset wanted (used parameter-"mode")

# ATTENTION: MUST MODIFY CONSTRUCTOR WHILE CHANGING DATASET LOCATION OR FORMAT !!

import os
import cv2
from torch.utils.data import Dataset, Subset
import numpy


class SLR_Dataset(Dataset):
    def __init__(self, mode, server,tf):
        super(SLR_Dataset, self).__init__()
        self.mode = mode
        self.tf = tf
        #######################################
        # SLR dataset location
        # "Raw" outputs video path
        if server == "shanyx":
            if mode == "Raw":
                self.path = "/data/shanyx/hrh/sign/Crop"
            elif mode == "Resnet18":
                self.path = "/data/shanyx/hrh/sign/Resnet18"
            elif mode == "Resnet34":
                self.path = "/data/shanyx/hrh/sign/Resnet34"
            elif mode == "Resnet50":
                self.path = "/data/shanyx/hrh/sign/Resnet50"
            elif mode == "Resnet101":
                self.path = "/data/shanyx/hrh/sign/Resnet101"
            elif mode == "Resnet152":
                self.path = "/data/shanyx/hrh/sign/Resnet152"
        elif server == "hk1":
            if mode == "Raw":
                self.path = "/data/hk1/Goolo/hjj/SLR_Data/Crop"
            elif mode == "Resnet18":
                self.path = "/data/hk1/Goolo/hjj/SLR_Data/Resnet18"
            elif mode == "Resnet34":
                self.path = "/data/hk1/Goolo/hjj/SLR_Data/Resnet34"
            elif mode == "Resnet50":
                self.path = "/data/hk1/Goolo/hjj/SLR_Data/Resnet50"
            elif mode == "Resnet101":
                self.path = "/data/hk1/Goolo/hjj/SLR_Data/Resnet101"
            elif mode == "Resnet152":
                self.path = "/data/hk1/Goolo/hjj/SLR_Data/Resnet152"
        #######################################

    def __getitem__(self, item):
        label = item // 50
        index = item % 50
        #######################################
        # output ((Batch, dataPath)
        if self.mode == "Raw":
            data = os.path.join(self.path, str(label), str(index) + ".avi")
        # output (Batch, sampleCounts(30), Feature)
        elif self.mode in {"Resnet18", "Resnet34", "Resnet50", "Resnet101", "Resnet152"}:
            path = os.path.join(self.path, str(label), str(index) + ".npy")
            data = numpy.load(path)
        #######################################
        print(data)
        for i in range(len(data)):
            data[i] = self.tf(data[i])
        return data, label

    def __len__(self):
        return 500*50

    ################################
    # Train:Validation:Test = 7:2:1
    # output Dataset type
    def getThreeSet(self):
        train_set, validation_set, test_set = [], [], []
        for ii in range(500):
            for jj in range(35):
                train_set.append(ii*50+jj)
            for jj in range(35, 45):
                validation_set.append(ii*50+jj)
            for jj in range(45, 50):
                test_set.append(ii*50+jj)
        return Subset(self, train_set), Subset(self, validation_set), Subset(self, test_set)
    ################################