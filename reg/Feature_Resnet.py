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
from efficientnet import EfficientNet


def collate_fn1(data):
    data_list, label, person_index, videoLength = [], [], [], []
    for index, item in enumerate(data):
        data_path = item[0]
        label.append(item[1])
        person_index.append(Path(data_path).stem)
        video = cv2.VideoCapture(data_path)
        video.read()
        videoLength.append(int(video.get(7)) - 1)
        # sample = numpy.linspace(0, frameCounts-1, sampleCounts, dtype=numpy.int)

        data_temp = torch.empty(size=(videoLength[index], 3, 570, 570))
        for ii in range(videoLength[index]):
            _, frame = video.read()
            temp = torchvision.transforms.ToTensor()(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
            data_temp[ii] = torchvision.transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])(temp)

        data_list.append(data_temp)

    data_final = torch.empty(size=(sum(videoLength), 3, 570, 570))
    ss = 0
    for index, item in enumerate(videoLength):
        data_final[ss:ss + item] = data_list[index]
        ss += item
    return data_final, label, person_index, videoLength


class Model(torch.nn.Module):
    def __init__(self, model_choose):
        super(Model, self).__init__()
        if model_choose == "3":
            self.feature = EfficientNet.from_pretrained("efficientnet-b3")
        elif model_choose == "5":
            self.feature = EfficientNet.from_pretrained("efficientnet-b5")
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
        # cd /data/hk1/Goolo/hjj/Sign\ Language\ Translation/
        # CUDA_VISIBLE_DEVICES=1,3 python Feature_Resnet.py 0 50 /data/hk1/Goolo/hjj/SLR_Data/Resnet50

        # python Feature_Resnet.py shanyx 0 50 /data/shanyx/hrh/sign/Resnet50
        # python Feature_Resnet.py will 0 152 D:/SLR_Data/Process_Resnet152
        # python Feature_Resnet.py shanyx 0 5 /data/shanyx/SLR/Process_Efficientnet5
        # torch.cuda.set_device(3)
        server = sys.argv[1]
        st = int(sys.argv[2])
        model_choose = sys.argv[3]
        # save directory must named by ResnetLayerCounts && sampleCounts
        save_dir = sys.argv[4]
        num_workers = 8
        batch_size = 1
        # one more hyperparameter below (ResnetLayerCounts)

        print("server: ", sys.argv[1])
        print("st: ", sys.argv[2])
        print("save_dir: ", save_dir)
        print("num_workers: ", num_workers)
        print("batch_size: ", batch_size)
        print("model_choose: ", model_choose)

        # batch_size must be 1 or modify collate_fn()
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        feature = Model(model_choose)
        feature = torch.nn.DataParallel(feature, device_ids=[0, 1, 3])
        #feature = torch.nn.DataParallel(Model(model_choose), device_ids=[0, 1])
        feature = feature.to(device).eval()
        datadata = SLR_Dataset.SLR_Dataset(mode="Process_Face_Hand", server=server)
        datadata = Subset(datadata, list(range(st, 25000)))
        dataloader = DataLoader(dataset=datadata, batch_size=batch_size, collate_fn=collate_fn1, shuffle=False, num_workers=num_workers)

        t1, t2, t3 = time.time(), time.time(), time.time()

        print("Processing......")

        for ITER, (data, label, person_index, videoLength) in enumerate(dataloader):
            t1 = time.time()
            print("TIME T1：%f" % (t1 - t3))
            data = feature(data.to(device)).to("cpu").detach().numpy()
            t2 = time.time()
            print("TIME T2：%f" % (t2 - t1))
            ss = 0
            for index, item in enumerate(videoLength):
                numpy.save(os.path.join(save_dir, str(label[index]), person_index[index]+".npy"), data[ss:ss+item].reshape(item, -1))
                ss += item
            t3 = time.time()
            print("TIME T3：%f" % (t3 - t2))
            #if ITER % 100 == 0:
            print("ITERATION: %d/%d RUNNING......"% (ITER+1, len(dataloader)))
