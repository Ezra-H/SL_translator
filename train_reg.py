
import os
import sys
import numpy
import torch
# import SLR_Dataset
from torch.utils.data import DataLoader
from reg import *
if __name__ == "__main__":
    server = sys.argv[1]
    torch.cuda.set_device(int(sys.argv[2]))
    num_workers = int(sys.argv[3])
    batch_size = int(sys.argv[4])
    hidden_size = int(sys.argv[5])
    epoch_counts = int(sys.argv[6])
    learning_rate = float(sys.argv[7])
    from_scratch = bool(int(sys.argv[8]))
    feature_type = sys.argv[9]
    modelPath = sys.argv[10]
    num_layers = int(sys.argv[11])
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    feature_size = 512
    if "Resnet18" in feature_type:
        feature_size = 512
    elif "Resnet34" in feature_type:
        feature_size = 512
    elif "Resnet50" in feature_type:
        feature_size = 2048
    elif "Resnet101" in feature_type:
        feature_size = 2048
    elif "Resnet152" in feature_type:
        feature_size = 2048
    elif "Efficientnet3" in feature_type:
        feature_size = 1536
    if "Hand" in feature_type:
        feature_size *= 2

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
        model = Neural_Classifier_sample(feature_size, hidden_size, num_layers)
        model.load_state_dict(checkpoint['model_state_dict'])
        print("Model loaded from \""+str(modelPath)+"\"")
    else:
        model = Neural_Classifier_sample(feature_size, hidden_size, num_layers)

    print("server: ", sys.argv[1])
    print("cuda_index: ", sys.argv[2])
    print("num_workers: ", num_workers)
    print("batch_size: ", batch_size)
    print("num_layers: ", num_layers)
    print("device: ", device)
    print("hidden_size: ", hidden_size)
    print("feature_type: ", feature_type)
    print("epoch_counts: ", epoch_counts)
    print("learning_rate: ", learning_rate)
    print("from_scratch: ", from_scratch)

    def collate_fn1(data_raw):
        batch_size_now = len(data_raw)
        video_length = torch.empty(batch_size_now, dtype=torch.long)
        for index, item in enumerate(data_raw):
            video_length[index] = item[0].shape[0]

        label = torch.empty(len(data_raw), dtype=torch.long)
        data = torch.empty(size=(batch_size_now, torch.max(video_length).item(), feature_size), dtype=torch.float)

        for index, item in enumerate(data_raw):
            label[index] = item[1]
            data[index, :video_length[index], :] = torch.from_numpy(item[0])

        return data, video_length, label


    #model = torch.nn.DataParallel(model, device_ids=[0, 2, 3])
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

        for iters, (data, video_length, label) in enumerate(trainloader):
            data, video_length, label = data.to(device), video_length.to(device), label.to(device)
            packed = nn.utils.rnn.pack_padded_sequence(data, video_length, batch_first=True, enforce_sorted=False)
            output = model(packed)
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

        for iters, (data, video_length, label) in enumerate(validloader):
            data, video_length, label = data.to(device), video_length.to(device), label.to(device)
            packed = nn.utils.rnn.pack_padded_sequence(data, video_length, batch_first=True, enforce_sorted=False)
            output = model(packed)
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