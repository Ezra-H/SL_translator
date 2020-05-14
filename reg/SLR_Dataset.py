# Definition for Dataset
# Auto divided into train/validation/test dataset (used function-"getThreeSet()")
# Easy change dataset wanted (used parameter-"mode")

# ATTENTION: MUST MODIFY CONSTRUCTOR WHILE CHANGING DATASET LOCATION OR FORMAT !!

import os
import cv2
from torch.utils.data import Dataset, Subset
import torchvision
import numpy
import torch


class SLR_Dataset(Dataset):
    def __init__(self, mode, server):
        super(SLR_Dataset, self).__init__()
        self.mode = mode
        #######################################
        # SLR dataset location
        # "Raw" outputs video path
        if server == "shanyx":
            if mode == "Raw":
                self.path = "/data/shanyx/SLR/Crop"
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
            elif mode == "Efficientnet3":
                self.path = "/data/shanyx/SLR/Efficientnet3"
            elif mode == "Process_Face_Hand":
                self.path = "/data/shanyx/SLR/Process_Face"
            elif mode == "Process_Resnet18":
                self.path = "/data/shanyx/SLR/Process_Resnet18"
            elif mode == "Process_Resnet34":
                self.path = "/data/shanyx/SLR/Process_Resnet34"
            elif mode == "Process_Resnet50":
                self.path = "/data/shanyx/SLR/Process_Resnet50"
            elif mode == "Process_Resnet101":
                self.path = "/data/shanyx/SLR/Process_Resnet101"
            elif mode == "Process_Resnet152":
                self.path = "/data/shanyx/SLR/Process_Resnet152"
            elif mode == "Process_Efficientnet3":
                self.path = "/data/shanyx/SLR/Process_Efficientnet3"
            elif mode == "Hand_Resnet18":
                self.path = "/data/shanyx/SLR/Hand_Resnet18"
            elif mode == "Hand_Resnet34":
                self.path = "/data/shanyx/SLR/Hand_Resnet34"
            elif mode == "Hand_Resnet50":
                self.path = "/data/shanyx/SLR/Hand_Resnet50"
            elif mode == "Hand_Resnet101":
                self.path = "/data/shanyx/SLR/Hand_Resnet101"
            elif mode == "Hand_Resnet152":
                self.path = "/data/shanyx/SLR/Hand_Resnet152"
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
        elif server == "will":
            if mode == "Hand":
                self.path = "C:/Users/Will/Desktop/SLR_Data/SLR_Data_New"
            elif mode == "Resnet18":
                self.path = "C:/Users/Will/Desktop/SLR_Data/Hand_Resnet18"
            elif mode == "Resnet34":
                self.path = "C:/Users/Will/Desktop/SLR_Data/Hand_Resnet34"
            elif mode == "Resnet50":
                self.path = "C:/Users/Will/Desktop/SLR_Data/Hand_Resnet50"
            elif mode == "Resnet101":
                self.path = "C:/Users/Will/Desktop/SLR_Data/Hand_Resnet101"
            elif mode == "Resnet152":
                self.path = "C:/Users/Will/Desktop/SLR_Data/Hand_Resnet152"
            elif mode == "Process_Face_Hand":
                self.path = "D:/SLR_Data/Process_Face"

        #######################################

    def __getitem__(self, item):
        label = item // 50
        index = item % 50
        #######################################
        # output ((Batch, dataPath)
        if self.mode in {"Raw", "Process_Face_Hand"}:
            data = os.path.join(self.path, str(label), str(index) + ".avi")
        # output (Batch, sampleCounts(30), Feature)
        elif "Hand" in self.mode:
            path = os.path.join(self.path, str(label), str(index) + "LR.npy")
            data = numpy.load(path)
        elif "Resnet" in self.mode:
            path = os.path.join(self.path, str(label), str(index) + ".npy")
            data = numpy.load(path)
        elif "Efficientnet" in self.mode:
            path = os.path.join(self.path, str(label), str(index) + ".npy")
            data = numpy.load(path)

            # video_path = os.path.join(self.path, str(label), str(index) + ".avi")
            # cap = cv2.VideoCapture(video_path)
            # frame_counts = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            # data = torch.empty(size=(frame_counts, 3, 570, 570))
            # for ii in range(frame_counts):
            #     _, frame = cap.read()
            #     temp = torchvision.transforms.ToTensor()(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
            #     data[ii] = torchvision.transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])(temp)
        #######################################
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