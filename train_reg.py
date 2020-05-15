import os
import cv2
import sys
import numpy
import torch
import torch.nn as nn
import torchvision
from reg import *
from model import *
from torch.utils.data import DataLoader, Subset
from apex import amp
import argparse
parser = argparse.ArgumentParser(description='sign_language_recognition')

parser.add_argument(
    '--server',
    type=str,
    default="shanyx",
    help='which server we used now (default shanyx)')
parser.add_argument(
    '--gpus',
    type=str,
    default='0',
    help='the GPU device number (default 0)')
parser.add_argument(
    '--num_workers',
    type=int,
    default=8,
    help='using fp16 to shrink the occupation of GPU (default True)')
parser.add_argument(
    '--batch_size',
    type=int,
    default=4,
    help='The number of samples in a mini-batch(default 4)')
parser.add_argument(
    '--hidden_size',
    type=int,
    default=512,
    help='The hidden feature size (default 512)')
parser.add_argument(
    '--epoch_counts',
    type=float,
    default=250,
    help='the maximium epoch number(default 250)')
parser.add_argument(
    '--lr',
    type=float,
    default=5e-4,
    help='learning rate(default 5e-4)')
parser.add_argument(
    '--from_scratch',
    type=int,
    default=0,
    help='(default 0)')
parser.add_argument(
    '--feature_type',
    type=str,
    default="Raw",
    help='The type of input feature(default Raw)')
parser.add_argument(
    '--modelPath',
    type=str,
    default="/data/shanyx/SLR/model/all.pth",
    help='The model parameters path(default /data/shanyx/SLR/model/all.pth)')
parser.add_argument(
    '--num_layers',
    type=int,
    default=3,
    help='The num of rnn layers(default 3)')
parser.add_argument(
    '--fp16',
    type=bool,
    default=True,
    help='using fp16 to shrink the occupation of GPU (default True)')
parser.add_argument(
    '--number',
    type=int,
    default=50,
    help='the num of the dataset (default 50)')


class Sign_Recognition(torch.nn.Module):
    def __init__(self, feature_size, hidden_size, num_layers):
        super(Sign_Recognition, self).__init__()
        self.feature_size = feature_size
        self.gru = nn.GRU(input_size=feature_size, hidden_size=hidden_size, batch_first=True, num_layers=num_layers, bidirectional=True)
        self.linear = torch.nn.Linear(hidden_size*2, 500)
        self.efficientnet = EfficientNet.from_pretrained("efficientnet-b3")

    def forward(self, data_in, video_length):
        # device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        data = torch.empty(size=(torch.sum(video_length).item(), 3, 570, 570)).cuda()
        data_2 = torch.empty(size=(len(data_in), torch.max(video_length).item(), self.feature_size)).cuda()
        temp = 0
        for index, item in enumerate(data_in):
            data[temp:temp+video_length[index].item(), :] = item
            temp = temp + video_length[index].item()

        data = self.efficientnet(data)

        temp = 0
        for index in range(len(data_in)):
            item = video_length[index].item()
            data_2[index, 0:item, :] = data[temp:temp+item, :]
            temp = temp + item
        packed = nn.utils.rnn.pack_padded_sequence(data_2, video_length, batch_first=True, enforce_sorted=False)
        out, _ = self.gru(packed,)
        unpacked = torch.nn.utils.rnn.pad_packed_sequence(out, batch_first=True)
        last = unpacked[0][range(unpacked[0].shape[0]), unpacked[1]-1]
        last = self.linear(last)
        return last


if __name__ == '__main__':
    
    # python Recognition_Sample.py [cuda_index] [num_workers] [batch_size] [hidden_size]
    # [epoch_counts] [learning_rate] [from_scratch] [modelPath]
    # cd /home/shanyx/anaconda3/bin
    # conda activate hrh_pytorch
    # cd /data/shanyx/will/SignLanguageTranslation
    # python Recognition_GRU.py shanyx 3 8 32 2048 250 0.0001 0 Raw /data/shanyx/SLR/model/all.pth 3

    # server = sys.argv[1]
    # torch.cuda.set_device(int(sys.argv[2]))
    # num_workers = int(sys.argv[3])
    # batch_size = int(sys.argv[4])
    # hidden_size = int(sys.argv[5])
    # epoch_counts = int(sys.argv[6])
    # learning_rate = float(sys.argv[7])
    # from_scratch = bool(int(sys.argv[8]))
    # feature_type = sys.argv[9]
    # modelPath = sys.argv[10]
    # num_layers = int(sys.argv[11])
    args = parser.parse_args()
    os.environ['CUDA_VISIBLE_DEVICES'] = args.gpus
    print(os.environ['CUDA_VISIBLE_DEVICES'])
    server = args.server
    # torch.cuda.set_device(args.gpus)
    num_workers = args.num_workers
    batch_size = args.batch_size
    hidden_size = args.hidden_size
    epoch_counts = args.epoch_counts
    learning_rate = args.lr
    from_scratch = args.from_scratch
    feature_type = args.feature_type
    modelPath = args.modelPath
    num_layers = args.num_layers
    number = args.number
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    feature_size = 1536
    if "Efficientnet3" in feature_type:
        feature_size = 1536

    print("Checking existed model.....[from_scratch: %s]" % (from_scratch))
    epochStart, loss_history_train, acc_history_train, loss_history_valid, acc_history_valid = 1, [], [], [], []
    if os.path.exists(modelPath) and (not from_scratch):
        checkpoint = torch.load(modelPath)
        epochStart = checkpoint['epoch']
        loss_history_train = checkpoint['loss_history_train']
        acc_history_train = checkpoint['acc_history_train']
        loss_history_valid = checkpoint['loss_history_valid']
        acc_history_valid = checkpoint['acc_history_valid']
        learning_rate = checkpoint['learning_rate']
        batch_size = checkpoint['batch_size']
        feature_type = checkpoint['feature_type']
        hidden_size = checkpoint['hidden_size']
        num_layers = checkpoint['num_layers']
        model = Sign_Recognition(feature_size, hidden_size, num_layers)
        print("Model loaded from \""+str(modelPath)+"\"")
    else:
        model = Sign_Recognition(feature_size, hidden_size, num_layers)

    print("server: ", args.server)
    print("cuda_index: ", args.gpus)
    print("num_workers: ", num_workers)
    print("batch_size: ", batch_size)
    print("num_layers: ", num_layers)
    print("device: ", device)
    print("hidden_size: ", hidden_size)
    print("feature_type: ", feature_type)
    print("epoch_counts: ", epoch_counts)
    print("learning_rate: ", learning_rate)
    print("from_scratch: ", from_scratch)
    print("number:", number)
    print("modelPath:", modelPath)


    def collate_fn1(data):
        label = torch.empty(len(data), dtype=torch.long)
        videoLength = torch.empty(len(data), dtype=torch.long)
        data_all = []
        for index, item in enumerate(data):
            label[index] = item[1]
            data_path = item[0]
            video = cv2.VideoCapture(data_path)
            videoLength[index] = int(video.get(7))
            data_temp = torch.empty(size=(videoLength[index].item(), 3, 570, 570))
            for ii in range(videoLength[index].item()):
                _, frame = video.read()
                temp = torchvision.transforms.ToTensor()(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
                data_temp[ii] = torchvision.transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])(temp)
            data_all.append(data_temp)
        return data_all, videoLength, label


    model = model.cuda()
    # model = model.to(device)
    train_set, valid_set, _ = SLR_Dataset.SLR_Dataset(mode="Process_keyframe", server=server, number=number).getThreeSet()
    trainloader = DataLoader(dataset=train_set, batch_size=batch_size, shuffle=True, collate_fn=collate_fn1,
                             num_workers=num_workers)
    validloader = DataLoader(dataset=valid_set, batch_size=batch_size, shuffle=True, collate_fn=collate_fn1,
                             num_workers=num_workers)
    optim = torch.optim.Adam(model.parameters(), lr=learning_rate)
    # optim = torch.optim.SGD(model.parameters(), lr=learning_rate)
    criterion = torch.nn.CrossEntropyLoss()

    # appling fp16
    if args.fp16:
        model, optim = amp.initialize(model, optim, opt_level="O1") # 这里是“欧一”，不是“零一”
    model = torch.nn.DataParallel(model)

    if os.path.exists(modelPath) and (not from_scratch):
        checkpoint = torch.load(modelPath)
        model.load_state_dict(checkpoint['model_state_dict'])

    print("processing......")
    for epoch in range(epochStart, epoch_counts+1):
        sum_loss, sum_acc = 0, 0
        model.train()

        for iters, (data, video_length, label) in enumerate(trainloader):
            label = label.cuda()
            output = model(data, video_length)
            loss = criterion(output, label)
            optim.zero_grad()

            # fp16 backpropagation
            if args.fp16:
                with amp.scale_loss(loss, optim) as scaled_loss:
                    scaled_loss.backward()
            else:
                loss.backward()
            optim.step()

            _, prediction = output.max(1)
            acc = torch.sum(label == prediction).item()
            sum_acc += acc
            sum_loss += loss.item()

            if ((iters + 1) % 10*num_workers) == 0:
                print("Epoch: %d ITER: %d, loss: %f, acc: %f%%" % (
                    epoch, iters, loss.item(), acc/batch_size*100))

        loss_history_train.append(sum_loss)
        acc_history_train.append(sum_acc)
        print("************************************")
        print("************************************")
        print('(Train) EPOCH:[%d / %d] average loss: %f accurrate rate: %f%%'
              % (epoch, epoch_counts, sum_loss / len(trainloader), sum_acc / train_set.__len__() * 100))

        sum_loss, sum_acc = 0, 0
        model.eval()

        for iters, (data, video_length, label) in enumerate(validloader):
            label = label.to(device)
            output = model(data, video_length)
            sum_loss += criterion(output, label).item()
            _, prediction = output.max(1)
            sum_acc += torch.sum(label == prediction).item()

        loss_history_valid.append(sum_loss)
        acc_history_valid.append(sum_acc)
        print('(Validation) EPOCH:[%d / %d] average loss: %f accurrate rate: %f%%'
              % (epoch, epoch_counts, sum_loss / len(validloader), sum_acc / valid_set.__len__() * 100))

        torch.save({
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'loss_history_train': loss_history_train,
            'acc_history_train': acc_history_train,
            'loss_history_valid': loss_history_valid,
            'acc_history_valid': acc_history_valid,
            'learning_rate': learning_rate,
            'batch_size': batch_size,
            'hidden_size': hidden_size,
            'feature_type': feature_type,
            'num_layers': num_layers
        }, modelPath)
        print("Model Saved to \"" + str(modelPath) + "\"")

        print("************************************")
        print("************************************")