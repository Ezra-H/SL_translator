import os
import torch
import numpy as np
import pandas as pd
import PIL.Image as Image
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
from torchvision import transforms, utils
from skimage import io, transform

class SignDataset(Dataset): 
    def __len__(self) -> int: 
        return self.len
    
    def __init__(self, csv_file: str, root_dir: str, num_frames_per_clip = 16,transform=None) -> None: 
        super().__init__() 
        self.root_dir = root_dir
        #lines=list(lines["video"])
        self.datalist=pd.read_csv(csv_file,sep="|")
        self.len=len(self.datalist)
        self.num_frames_per_clip=num_frames_per_clip
        self.transform = transform
    def __getitem__(self, index:int): 
        data_dir=self.datalist["name"][index]
        img_label =self.datalist["translation"][index]
        img_datas=[]
        for parent, dirnames,filenames in os.walk(self.root_dir + data_dir ):
            filenames=sorted(filenames)
            for i in filenames:
                image_name = self.root_dir + data_dir + '/' + i
                img=Image.open(image_name)
                img_datas.append(np.array(img))
        if self.transform is not None: 
            img_datas = self.transform(img_datas)
        return np.array(img_datas),img_label 


def get_frames_data(filename,num_frames_per_clip=16,s_index=0):
    ret_arr=[]
    
    for parent, dirnames,filenames in os.walk(filename):
        if(len(filenames)<num_frames_per_clip):
            print("Get invaild data!")
            return [],s_index
        filenames=sorted(filenames)
        print(filenames)
        s_index=random.randint(0,len(filenames)-num_frames_per_clip) if s_index == 0 else s_index 
        for i in range(s_index,s_index+num_frames_per_clip):
            image_name=str(filename)+'/'+str(filenames[i])
            img=Image.open(image_name)
            img_data=np.array(img)
            ret_arr.append(img_data)
    return ret_arr,s_index

if __name__  == "__main__":
    csv_file = "/home/hrh/sign_language/PHOENIX-2014-T-release-v3/PHOENIX-2014-T/annotations/manual/PHOENIX-2014-T.dev.corpus.csv"
    root_dir = "/home/hrh/sign_language/PHOENIX-2014-T-release-v3/PHOENIX-2014-T/features/fullFrame-210x260px/dev/"
    dealDataset = SignDataset(csv_file,root_dir)
    train_dataset = SignDataset(csv_file,root_dir)
    train_loader = DataLoader(dataset=train_dataset,
                            batch_size=1,   #现在只能load一个batch，但是一个batch中有一组图片，对应一个翻译
                            shuffle=True)

    for step, data in enumerate(train_loader):
        # step: the index of data_loader
        # data: the return item of Class Dataset ,etc. (img_datas,img_label)
        print(data[0].size())