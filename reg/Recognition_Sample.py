import os
import sys
import numpy
import torch
# import SLR_Dataset
from torch.utils.data import DataLoader


class Neural_Classifier_sample(torch.nn.Module):
    def __init__(self, feature_size):
        super(Neural_Classifier_sample, self).__init__()
        self.linear = torch.nn.Linear(feature_size, 500)

    def forward(self, data):
        data = self.linear(data)
        return data


if __name__ == '__main__':
    # python Recognition_Sample.py [cuda_index] [num_workers] [batch_size] [sample_rate]
    # [epoch_counts] [learning_rate] [from_scratch] [modelPath]
    # cd /home/shanyx/anaconda3/bin
    # conda activate hrh_pytorch
    # cd /data/shanyx/will/Sign\ Language\ Translation
    # python Recognition_Sample.py shanyx 3 16 32 30 200 0.00001 0 Resnet34 /data/shanyx/hrh/sign/model/resnet34_sample30_0.00001.pth
    # python Recognition_Sample.py hk1 0 16 32 30 200 0.00001 0 Resnet50 /data/hk1/Goolo/hjj/SLR_Data/model/resnet50_sample30_0.00001.pth
    server = sys.argv[1]
    torch.cuda.set_device(int(sys.argv[2]))
    num_workers = int(sys.argv[3])
    batch_size = int(sys.argv[4])
    sample_rate = int(sys.argv[5])
    epoch_counts = int(sys.argv[6])
    learning_rate = float(sys.argv[7])
    from_scratch = bool(int(sys.argv[8]))
    feature_type = sys.argv[9]
    modelPath = sys.argv[10]
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    if feature_type == "Resnet18":
        feature_size = 512 * sample_rate
    elif feature_type == "Resnet34":
        feature_size = 512 * sample_rate
    elif feature_type == "Resnet50":
        feature_size = 2048 * sample_rate
    elif feature_type == "Resnet101":
        feature_size = 2048 * sample_rate
    elif feature_type == "Resnet152":
        feature_size = 2048 * sample_rate

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
        sample_rate = checkpoint['sample_rate']
        feature_type = checkpoint['feature_type']
        model = Neural_Classifier(feature_size)
        model.load_state_dict(checkpoint['model_state_dict'])
        print("Model loaded from \""+str(modelPath)+"\"")
    else:
        model = Neural_Classifier(feature_size)

    print("server: ", sys.argv[1])
    print("cuda_index: ", sys.argv[2])
    print("num_workers: ", num_workers)
    print("batch_size: ", batch_size)
    print("device: ", device)
    print("sample_rate: ", sample_rate)
    print("feature_type: ", feature_type)
    print("epoch_counts: ", epoch_counts)
    print("learning_rate: ", learning_rate)
    print("from_scratch: ", from_scratch)


    def collate_fn1(data_raw):
        label = torch.empty(len(data_raw), dtype=torch.long)
        data = torch.empty(size=(len(data_raw), feature_size), dtype=torch.float)
        for index, item in enumerate(data_raw):
            label[index] = item[1]
            data[index] = torch.from_numpy(
                item[0][numpy.linspace(0, item[0].shape[0] - 1, num=30, dtype=numpy.int)].flatten())
        return data, label


    model = model.to(device)
    train_set, valid_set, _ = SLR_Dataset.SLR_Dataset(mode=feature_type, server=server).getThreeSet()
    trainloader = DataLoader(dataset=train_set, batch_size=batch_size, shuffle=True, collate_fn=collate_fn1,
                             num_workers=num_workers)
    validloader = DataLoader(dataset=valid_set, batch_size=batch_size, shuffle=True, collate_fn=collate_fn1,
                             num_workers=num_workers)
    optim = torch.optim.Adam(model.parameters(), lr=learning_rate)
    criterion = torch.nn.CrossEntropyLoss()

    print("processing......")
    for epoch in range(epochStart, epoch_counts+1):
        sum_loss, sum_acc = 0, 0
        model.train()

        for iters, (data, label) in enumerate(trainloader):
            data, label = data.to(device), label.to(device)
            output = model(data)
            loss = criterion(output, label)
            optim.zero_grad()
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

        for iters, (data, label) in enumerate(validloader):
            data, label = data.to(device), label.long().to(device)
            output = model(data)
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
            'sample_rate': sample_rate,
            'feature_type': feature_type
        }, modelPath)
        print("Model Saved to \"" + str(modelPath) + "\"")

        print("************************************")
        print("************************************")