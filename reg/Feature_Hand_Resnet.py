# Definition for Feature_Resnet_Get&Save
# import raw data, process and save in format-npy
# have tuned mean and std for ResnetPreTrainningModel || Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])

import os
import sys
import cv2
import time
import torch
import numpy
import torchvision
import SLR_Dataset
from pathlib import Path
from torch.utils.data import DataLoader, Subset


def collate_fn1(data):
    data_list, label, person_index, videoLength = [], [], [], []
    for index, item in enumerate(data):
        data_path = item[0]
        label.append(item[1])
        person_index.append(data_path[0].split("/")[-2])
        videoL, videoR = cv2.VideoCapture(data_path[0]), cv2.VideoCapture(data_path[1])
        #print(data_path[0])
        videoLength.append(int(videoL.get(7)))
        data_temp = torch.empty(size=(2, videoLength[index], 3, 150, 150))
        for ii in range(videoLength[index]):
            _, frame = videoL.read()
            frame = cv2.resize(frame, (456, 456), cv2.INTER_AREA)
            temp = torchvision.transforms.ToTensor()(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
            data_temp[0, ii] = torchvision.transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])(temp)

            _, frame = videoR.read()
            temp = torchvision.transforms.ToTensor()(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
            data_temp[1, ii] = torchvision.transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])(temp)

        data_list.append(data_temp)

    data_finalL, data_finalR = torch.empty(size=(sum(videoLength), 3, 150, 150)), torch.empty(size=(sum(videoLength), 3, 150, 150))
    ss = 0
    for index, item in enumerate(videoLength):
        data_finalL[ss:ss + item], data_finalR[ss:ss + item] = data_list[index][0], data_list[index][1]
        ss += item
    return data_finalL, data_finalR, label, person_index, videoLength


class Model(torch.nn.Module):
    def __init__(self, model_choose):
        super(Model, self).__init__()
        if model_choose == "18":
            self.feature = torch.nn.Sequential(*list(torchvision.models.resnet18(pretrained=True).children())[:-1])
        elif model_choose == "34":
            self.feature = torch.nn.Sequential(*list(torchvision.models.resnet34(pretrained=True).children())[:-1])
        elif model_choose == "50":
            self.feature = torch.nn.Sequential(*list(torchvision.models.resnet50(pretrained=True).children())[:-1])
        elif model_choose == "101":
            self.feature = torch.nn.Sequential(*list(torchvision.models.resnet101(pretrained=True).children())[:-1])
        elif model_choose == "152":
            self.feature = torch.nn.Sequential(*list(torchvision.models.resnet152(pretrained=True).children())[:-1])

    def forward(self, data):
        return self.feature(data)


if __name__ == '__main__':
    with torch.no_grad():
        # python Feature_Resnet.py will 0 152 C:/Users/Will/Desktop/SLR_Data/Hand_Resnet152
        # python Feature_Hand_Resnet.py shanyx 0 152 /data/shanyx/SLR/Hand_Resnet152 0
        server = sys.argv[1]
        st = int(sys.argv[2])
        model_choose = sys.argv[3]
        save_dir = sys.argv[4]
        cuda_index = int(sys.argv[5])
        torch.cuda.set_device(cuda_index)
        num_workers = 8
        batch_size = 1

        print("server: ", server)
        print("st: ", st)
        print("save_dir: ", save_dir)
        print("num_workers: ", num_workers)
        print("batch_size: ", batch_size)
        print("model_choose: ", model_choose)
        print("cuda_index: ", cuda_index)

        # batch_size must be 1 or modify collate_fn()
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        feature = Model(model_choose)
        #feature = torch.nn.DataParallel(Model(model_choose), device_ids=[0, 1])
        feature = feature.to(device).eval()
        datadata = SLR_Dataset.SLR_Dataset(mode="Hand", server=server)
        datadata = Subset(datadata, list(range(st, 25000)))
        dataloader = DataLoader(dataset=datadata, batch_size=batch_size, collate_fn=collate_fn1, shuffle=False, num_workers=num_workers)

        t1, t2, t3 = time.time(), time.time(), time.time()

        print("Processing......")

        for ITER, (dataLeft, dataRight, label, person_index, videoLength) in enumerate(dataloader):
            t1 = time.time()
            print("TIME T1：%f" % (t1 - t3))
            dataLeft = feature(dataLeft.to(device)).to("cpu").detach().numpy()
            dataRight = feature(dataRight.to(device)).to("cpu").detach().numpy()
            dataLeft = dataLeft.reshape(dataLeft.shape[0], -1)
            dataRight = dataRight.reshape(dataRight.shape[0], -1)
            #data = (dataLeft + dataRight)/2
            data = numpy.hstack((dataLeft, dataRight))
            t2 = time.time()
            print("TIME T2：%f" % (t2 - t1))
            ss = 0
            for index, item in enumerate(videoLength):
                numpy.save(os.path.join(save_dir, str(label[index]), person_index[index]+"LR.npy"), data[ss:ss+item])
                ss += item
            t3 = time.time()
            print("TIME T3：%f" % (t3 - t2))
            #if ITER % 100 == 0:
            print("ITERATION: %d/%d RUNNING......"% (ITER+1, len(dataloader)))
