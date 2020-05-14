import torch
import matplotlib.pyplot as plt
def visualization(path):
    # "C:/Users/Administrator/Desktop/Effienctnet3_Gru_0.0001_3_512.pth"
    checkpoint18 = torch.load(path, map_location='cpu')
    #checkpoint101 = torch.load("C:/Users/Administrator/Desktop/resnet101_sample30_0.00001.pth", map_location='cpu')
    acc18 = checkpoint18["acc_history_valid"]
    for index, item in enumerate(acc18):
        acc18[index] = item/5000
    #acc101 = checkpoint101["acc_history_valid"]
    xx = list(range(len(acc18)))
    point18 = plt.plot(xx, acc18, c="g")
    #point101 = plt.scatter(xx, acc101, s=10, c="b")
    #plt.legend(handles=[point18, point101], labels=['Resnet18', 'Resnet101'])
    plt.show()