import torch
device= torch.device('cuda')

import numpy as np
import torchvision
import torchvision.transforms as trans

from matplotlib import pyplot as plt
from torchvision.utils import make_grid
import torch.nn as nn
import torch.nn.functional as f
#定义LeNet-5网络
class LeNet(nn.Module):
    """建立神经网络,需要继承nn.Module函数,需要定义两个函数:
    第一个:__init__初始化函数:采用nn.Sequential建立卷积神经网络
    第二个:forward(self,x):定义前馈学习,x为输入的像素矩阵
    """
    def __init__(self):
        super(LeNet,self).__init__()
        #构建卷积层
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels = 1,out_channels=6,kernel_size=5,stride=1,padding=0),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2,stride=2),
            nn.Conv2d(6,16,5),
            nn.ReLU(),
            nn.MaxPool2d(2,2)
        )
        #构建全连接层
        self.fc = nn.Sequential(
            nn.Linear(16*5*5,120),
            nn.ReLU(),
            nn.Linear(120,84),
            nn.ReLU(),
            nn.Linear(84,10)
        )#输出层没有用激活函数,后续损失函数包含了
    def forward(self,x):
        feature= self.conv(x)
        output = self.fc(feature.view(x.size(0),-1))
        return output

def accuracy(outputs, labels):
        preds = torch.max(outputs, dim=1)[1]
        return torch.sum(preds == labels).item()/len(preds)

def validate(model,test_loader):#######
    val_loss=0
    val_acc = 0
    model.eval()
    for inputs,labels in test_loader:
        inputs,labels = inputs.to(device),labels.to(device)
        outputs = model(inputs)
        loss = f.cross_entropy(outputs,labels)
        val_loss += loss.item()
        acc = accuracy(outputs,labels)
        val_acc += acc

    val_acc /= len(test_loader)
    val_loss /=len(test_loader)
    return val_loss,val_acc

def print_log(epoch,train_time,train_loss,train_acc,test_loss,test_acc,epochs):
    print(f"Epochs:[{epoch}/{epochs}],time:{train_time:.2f}s,train_loss:{train_loss:.4f},train_acc:{train_acc:.4f},"
          f"val_loss:{test_loss:.4f},val_acc:{test_acc:.4f}")

import time
def train(model,optimizer,trainloader,testloader,epochs,lr):
    train_losses=[]
    train_accs = []
    val_losses = []
    val_accs = []
    model.train()
    for epoch in range(epochs):
        train_loss= 0
        train_acc = 0
        start = time.time()
        for inputs,labels in trainloader:
            inputs,labels = inputs.to(device),labels.to(device)
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = f.cross_entropy(outputs,labels)
            acc = accuracy(outputs,labels)
            train_loss+=loss.item()
            train_acc += acc
            loss.backward()
            optimizer.step()
        end = time.time()
        train_time = end- start
        train_loss /= len(trainloader)
        train_acc/=len(trainloader)
        val_loss,val_acc = validate(model,testloader)
        train_losses.append(train_loss),train_accs.append(train_acc)
        val_losses.append(val_loss),val_accs.append(val_acc)
        print_log(epoch+1,train_time,train_loss,train_acc,val_loss,val_acc,epochs)
    return train_losses,train_accs,val_losses,val_accs


def main():
    transform = trans.Compose([
        trans.Resize((32, 32)),
        trans.ToTensor(),
        trans.Normalize((0.1307,), (0.3081,))
    ])  # 图片尺寸调整,转变为张量(像素值尺度范围调整和图片尺寸转职,并标准化,两个数字是数据提供方计算好的

    trainset = torchvision.datasets.MNIST(root='./mnist', train=True, download=True, transform=transform)
    testset = torchvision.datasets.MNIST(root='./mnist', train=False, download=True, transform=transform)

    train_loader = torch.utils.data.DataLoader(trainset, 128, shuffle=True, num_workers=8)
    test_loader = torch.utils.data.DataLoader(testset, 128, shuffle=False, num_workers=8)
    images, labels = next(iter(train_loader))#迭代器迭代,返回128个
    print(images.shape)
    print(labels.shape)
    fig,ax = plt.subplots(2,5,figsize=(12,5))#(生成2行共10个子图,fig是全图)
    ax = ax.flatten()#拉平成一维数组,便于后续循环引用
    for i in range(10):##制图
        im = images[labels==i][0].reshape(32,32)
        ax[i].imshow(im)#给每个子图赋值
    plt.show()#展示所有子图
    model = LeNet() #实例化一个LeNet模型
    model.to(device)#转变为GPU存储模式

    #查看模型具体信息
    from torchsummary import summary
    imagesize = 32
    summary(model,(1,32,32))
    #模型训练,指定迭代次数为30次,学习率指定为0.001
    epochs=30
    lr=1e-3
    optimizer = torch.optim.Adam(model.parameters(),lr=lr)
    train_losses,train_accs,val_losses,val_accs = train(model,optimizer,train_loader,test_loader,epochs,lr)
    #训练集的损失曲线和测试集的准确率曲线可视化
    plt.figure(figsize=(20,7))
    plt.subplot(1,2,1)
    plt.plot(train_losses)
    plt.xlabel("epoch")
    plt.ylabel("train_loss")
    plt.subplot(1,2,2)
    plt.plot(val_accs)
    plt.xlabel("epoch")
    plt.ylabel("val_accuracy")

    #第一层卷积核及其卷积特征图可视化
    plt.figure(figsize=(10,7))
    for i in range(6):
        plt.subplot(1,6,i+1)
        target = model.conv[0].weight.cpu()
        plt.imshow(target.data.numpy()[i,0,...])

    input_x_1 = testset[1][0].unsqueeze(0)
    feature_map = model.conv[1](model.conv[0].cpu()(input_x_1))
    input_x_2 = model.conv[2].cpu()(feature_map)
    feature_map2 = model.conv[4](model.conv[3].cpu()(input_x_2))
    plt.figure(figsize=(10,7))
    for i in range(6):
        plt.subplot(1,6,i+1)
        plt.imshow(feature_map[0,i,...].data.numpy())

    plt.figure(figsize=(48,18))##进行画图操作
    for i in range(6):
        for j in range(16):
            plt.subplot(6,16,i*16+j+1)
            plt.imshow(model.conv[3].weight.data.numpy()[j,i,...])


if __name__ == '__main__':
   main()
