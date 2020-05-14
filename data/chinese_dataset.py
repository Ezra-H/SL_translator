import os
import torch
import numpy as np
import pandas as pd
import PIL.Image as Image
import random
from torch.utils.data import Dataset,DataLoader
from torchvision import transforms, utils
import re
class ChineseSignDataset(Dataset): 
    def __init__(self, root, csv_file, num_frames_per_clip = 8,skip_rate=4,transform=None, mode='translate',flip=1): 
        super().__init__() 
        self.num_frames_per_clip=num_frames_per_clip
        self.flip = flip
        self.tf = transform
        self.skip_rate = num_frames_per_clip//2 if num_frames_per_clip>1 else 1
        self.root = root
        self.mode = mode
        self.data=pd.read_csv(csv_file)
        self.dirs = ['%06d'%i for i in self.data['dir'].tolist()]
        self.label = self.data['label'].tolist()
        self.words = self.data['words'].tolist()
        self.words = [i.split(' ') for i in self.words]

        # add tokens
        begin_token = ['<s>']
        end_token = ['</s>']
        add = []
        for i in self.words:
            add.append(begin_token+ i +end_token)
        self.words = add
        
        # update dirs, label, words, flatten
        new_dirs = []
        new_label = []
        new_words = []
        for i in range(len(self.dirs)):
            files = os.listdir(os.path.join(root,self.dirs[i]))
            new_dirs += [os.path.join(root,self.dirs[i],j) for j in files]
            new_label += [self.label[i]]*len(files)
            new_words += [self.words[i]]*len(files)
        self.seq_dict = {l:int(j) for l,j in zip(self.label,self.dirs)}
        self.dirs = new_dirs
        self.label = new_label
        self.words = new_words

    def path_sort(self,x1,x2):
        y1 = x1

    def __getitem__(self, index:int): 
        '''
        功能： 得到一个视频的连续序列。读取的是已经裁剪的图像。
            直接按照序列读取，然后裁剪输出。
        输出： 4维图片。
        '''
        # return the N clips video, sequence
        path = self.dirs[index]
        imgs_path = os.listdir(path)
        imgs_path.sort(key = lambda i:int(re.match(r'(\d+)',i).group()))
        imgs_path = [os.path.join(path,i) for i in imgs_path]
        # print(imgs_path)
        # print(imgs_path[0],self.seq_dict[self.label[index]])
        img_clips = self.get_frames_data(imgs_path,num_frames_per_clip=self.num_frames_per_clip)
        if self.mode=='translate':
            return img_clips.permute(0,2,1,3,4),self.words[index]
        elif self.mode=='pretrain':
            # print(self.label[index],self.seq_dict[self.label[index]])
            return img_clips.permute(0,2,1,3,4),self.seq_dict[self.label[index]]
    def __len__(self): 
        return len(self.dirs)
    

    def get_frames_data(self,filenames,num_frames_per_clip=8):
        imgs = []
        flip_flag = 0

        if random.random()<0.5:
            flip_flag=1
        for i in filenames:
            img = Image.open(i)
            if flip_flag and self.flip:
                img = transforms.RandomHorizontalFlip(1)(img)
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

if __name__  == "__main__":
    csv_file = "/data/shanyx/hrh/sign/ccsl/corpus.csv"
    root_dir = "/data/shanyx/hrh/sign/ccsl/picture/"
    tf = transforms.Compose([transforms.ToTensor()])

    train_dataset = ChineseSignDataset(root_dir,csv_file,8,tf)
    train_loader = DataLoader(dataset=train_dataset,
                            batch_size=1,   #现在只能load一个batch，但是一个batch中有一组图片，对应一个翻译
                            shuffle=True)
    
    for step, data in enumerate(train_loader):
        # step: the index of data_loader
        # data: the return item of Class Dataset ,etc. (img_datas,img_label)
        print(data[0].size())