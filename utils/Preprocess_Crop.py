# preprocess the raw data
# crop and cut in centrally
# trun into square(3, 570(W/px), 570(H/px))
import os
import cv2
import json
import pathlib
import SLR_Dataset
from torch.utils.data import DataLoader

import cv2
import numpy as np

save_dir = "H:/Crop"


def get_middle(img_raw):
    #cv2.imshow("1", img_raw)
    if img_raw.ndim > 2 and img_raw.shape[2] == 3:
        img = cv2.cvtColor(img_raw, cv2.COLOR_RGB2GRAY)
    else:
        img = img_raw
    img = cv2.Canny(img_raw, 50, 250)
    #cv2.imshow("2", img)
    temp = np.argwhere(img[690] == 255)
    temp = temp[temp > 100]
    middle = (temp[-1] + temp[0]) // 2
    #cv2.line(img_raw, (middle, 0), (middle, img_raw.shape[0]), (0, 0, 255), 5)
    #cv2.imshow("3", img_raw)
    #cv2.waitKey(0)
    top = np.argwhere(img[:, middle] == 255).flatten()
    top = top[0]-20
    if top+570 > 720:
        top = top - (top+570-720)
    return middle, top


def collate_fn1(data):
    return data[0]


dataloader = DataLoader(dataset=SLR_Dataset.SLR_Dataset(mode="Raw"), batch_size=1, shuffle=False, collate_fn=collate_fn1)
for index, (data_path, label) in enumerate(dataloader):
    label = 58
    data_path = "H:/SLRT/SLR_dataset/xf500_color_video/58/20.avi"
    person_index = int(pathlib.Path(data_path).stem)
    print(index, label, person_index)
    #if person_index != 20 and label != 488:
    #    continue
    saveVideo_dir = os.path.join(save_dir, str(label), pathlib.Path(data_path).name)

    videoFile = cv2.VideoCapture(data_path)
    fourcc = cv2.VideoWriter_fourcc(*'MPEG')
    fps = videoFile.get(5)
    frameCounts = int(videoFile.get(7)-1)
    out = cv2.VideoWriter(saveVideo_dir, fourcc, fps, (570, 570))

    _, _ = videoFile.read()
    for ii in range(frameCounts):
        check, frame = videoFile.read()
        if ii == 0:
            middle, top = get_middle(frame)
        temp = frame[top:top+570, middle-285:middle+285, :]
        out.write(temp)
    cv2.imwrite(saveVideo_dir[:-4]+".jpg", temp)
    out.release()

    if (index+1) % 100 == 0:
        print ("%d / %d RUNNING..." % (index+1, len(dataloader)))