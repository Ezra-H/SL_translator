from p3d_encode import P3D64
import os
import sys
from typing import List, Tuple, Dict, Set, Union
import torch
import torch.nn as nn
import torch.nn.utils
import torch.nn.functional as F
from torch.nn.utils.rnn import pad_packed_sequence, pack_padded_sequence


class sign_language_model(p3d):
    def __init__(self):
        super(sign_language_model,self).__init__()
        self.encode_output_size = 2048  #layer=2
        
        self.encode = P3D64()
        self.decode = None
        self.fc1 = nn.Linear(2*encode_output_size,encode_output_size,bias=True)
        self.fc2 = nn.Linear()


    def forward(self,x):
        out,hn = self.encode(x)
        out,hn = self.decode(out,hn)
        pass

if __name__ == "__main__"
    pass
