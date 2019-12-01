import os
import torch
import numpy as np
import pandas as pd
import PIL.Image as Image
import random
from torch.utils.data import Dataset,DataLoader
from torchvision import transforms, utils

class WeatherSignDataset(Dataset): 
    def __init__(self, root, csv_file,mode='train',num_frames_per_clip=8,skip_rate=4,transform=None): 
        super().__init__() 
        self.num_frames_per_clip=num_frames_per_clip
        self.tf = transform
        self.skip_rate = skip_rate
        self.root = root

        data=pd.read_csv(csv_file)
        self.dirs = data['names'].to_list()
        self.words = data['seq'].to_list()
        self.words = [i.split(' ') for i in self.words]

        # add tokens
        begin_token = ['<s>']
        end_token = ['</s>']
        add = []
        for i in self.words:
            add.append(begin_token+ i +end_token)
        self.words = add

    def __getitem__(self, index): 
        # return the N clips video, sequence
        path = self.dirs[index]
        imgs_path = os.listdir(path)
        imgs_path = [os.path.join(path,i) for i in imgs_path]

        img_clips = self.get_frames_data(imgs_path,num_frames_per_clip=8)
        return img_clips.permute(0,2,1,3,4),self.words[index]

    def __len__(self): 
        return len(self.dirs)

    def get_frames_data(self,filenames,num_frames_per_clip=8):
        imgs = []
        for i in filenames:
            img = Image.open(i)
            if self.tf:
                img = self.tf(img)
            else:
                img = transforms.ToTensor()(img)
            imgs.append(img)
        img_clips = []
        for i in range(0,len(imgs)-num_frames_per_clip+1,self.skip_rate):
            img_clips.append(torch.stack(imgs[i:i+num_frames_per_clip]))
        img_clips = torch.stack(img_clips)
        return img_clips