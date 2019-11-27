import torch
import PIL.Image as Image
import numpy as np
from my_model import *


if __name__ == "__main__":
    torch.cuda.set_device(0)
    vocab_path="F:/学习资料/基金项目-手语/nslt-master/Data/phoenix2014T.vocab.gloss"
    source_path = 'F:/学习资料/基金项目-手语/Data/feature/train/01April_2010_Thursday_heute-6694/'
    source = []
    for i in os.walk(source_path):
        filenames = i[2]
    for i in filenames:
        img=Image.open(source_path + i)
        source.append(np.array(img))
    source = torch.from_numpy(np.array(source))
    print(source.size())

    target = "LIEB ZUSCHAUER ABEND|liebe zuschauer guten abend"
    
    vocab = read_vocab(vocab_path)
    model = sign_language_model(vocab)
    model = model.cuda()
    source=torch.autograd.Variable(torch.rand(2,3,16,260,210)).cuda()   # if modality=='Flow', please change the 2nd dimension 3==>2
    
    target = "test"
    #print(model)
    return_data = model(source,target)
    print(return_data[0].size(),return_data[1].size())

    pass
