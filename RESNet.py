import torch
import torchvision.datasets
import torchvision.transforms as transforms
device = torch.device('cuda')
from matplotlib import pyplot as plt
from torchvision.utils import make_grid
import torch.nn as nn
import torch.nn.functional as f
import numpy
from torchsummary import summary
def main():
    #第一步:读取数据，并进行转换，这里的操作是变形和转换成张量。
    transform = transforms.Compose([
        transforms.Resize((32,32)),
        transforms.ToTensor()
    ])
    train_set = torchvision.datasets.CIFAR10(root='/CIFAR', train=True, transform = transform, download = True)
    test_set = torchvision.datasets.CIFAR10(root='/CIFAR',train = False, transform = transform, download = True)
    #第二步:载入数据加载器，分配batch为64，并预览数据
    train_loader = torch.utils.data.DataLoader(train_set,batch_size=64,shuffle=True,num_workers=8)
    val_loader = torch.utils.data.DataLoader(test_set, batch_size=64, shuffle=False, num_workers=8)
    data_iter = next(iter(train_loader))
    images,labels = data_iter
    print(images.shape)
    print(labels.shape)
    plt.figure(figsize=(12,20))
    plt.axis('off')
    images_mak = make_grid(images,nrow = 8)
    images_permute = images_mak.permute(1,2,0)
    plt.imshow(images_permute.detach().numpy())
    model = ResNet(BasicBlock,[1,1,1,1],10)
    model.to(device)
    imagesize = 32
    summary(model,input_size = (3,32,32))
    epochs =30
    lr = 1e-3
    optimizer = torch.optim.Adam(model.parameters(),lr=lr)
    train_losses,train_accs,val_losses,val_accs=train(model,optimizer,train_loader,val_loader,epochs,lr)
    plt.figure(figsize=(20, 7))
    plt.subplot(1, 2, 1)
    plt.plot(train_losses)
    plt.xlabel("epoch")
    plt.ylabel("train_loss")
    plt.subplot(1, 2, 2)
    plt.plot(val_accs)
    plt.xlabel("epoch")
    plt.ylabel("val_accuracy")
#残差学习模块
class BasicBlock(nn.Module):
    def __init__(self,in_planes,out_planes,stride):
        super(BasicBlock,self).__init__()
        #残差fx部分
        self.conv1 = nn.Conv2d(in_planes,out_planes,kernel_size=3,stride=stride,padding=1,bias=False)
        self.bn1 = nn.BatchNorm2d(out_planes)
        self.conv2 = nn.Conv2d(out_planes,out_planes,kernel_size=3,stride=1,padding=1,bias=False)
        self.bn2 = nn.BatchNorm2d(out_planes)
        #恒等变换部分
        self.shortcut = nn.Sequential()
        if stride!=1:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_planes,out_planes,kernel_size=1,stride=stride,bias=False),
                nn.BatchNorm2d(out_planes)
            )
    def forward(self,x):
        out = f.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += self.shortcut(x)
        return f.relu(out)


class ResNet(nn.Module):
    def __init__(self,block,num_blocks,num_classes):
        super(ResNet,self).__init__()
        #残差前准备，增多通道数？
        self.in_planes = 64
        self.previous = nn.Sequential(
            nn.Conv2d(in_channels=3,out_channels=64,kernel_size=3,stride=1,padding=1,bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(True)
        )
        #残差模块
        self.layer1 = self._make_layer(block,64,num_blocks[0],stride=1)
        self.layer2 = self._make_layer(block,128,num_blocks[1],stride=2)
        self.layer3 = self._make_layer(block,256, num_blocks[2], stride=2)
        self.layer4 = self._make_layer(block, 512, num_blocks[3], stride=2)
        #全连接层
        self.linear = nn.Linear(512,num_classes)
    def _make_layer(self,block,out_planes,num_blocks,stride):
        """
        conv_*x均有基本的残差学校模块堆叠，该函数循环模块num_blacks
        :param block: 残差模块
        :param out_planes:  输出通道数
        :param num_blocks: 模块循环次数
        :param stride: 步长
        """
        strides = [stride] + [1]*(num_blocks-1)
        layers = []
        for stride in strides:
            layers.append(block(self.in_planes,out_planes,stride))
            self.in_planes = out_planes
        return nn.Sequential(*layers)

    def forward(self,x):
        out = self.previous(x)
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.layer4(out)
        out = f.avg_pool2d(out,4)
        out = out.view(out.size(0),-1)
        out = self.linear(out)
        return out

def accuracy(outputs, labels):
    preds = torch.max(outputs, dim=1)[1]
    return torch.sum(preds == labels).item() / len(preds)


def validate(model, test_loader):  #######
    val_loss = 0
    val_acc = 0
    model.eval()
    for inputs, labels in test_loader:
        inputs, labels = inputs.to(device), labels.to(device)
        outputs = model(inputs)
        loss = f.cross_entropy(outputs, labels)
        val_loss += loss.item()
        acc = accuracy(outputs, labels)
        val_acc += acc

    val_acc /= len(test_loader)
    val_loss /= len(test_loader)
    return val_loss, val_acc


def print_log(epoch, train_time, train_loss, train_acc, test_loss, test_acc, epochs):
    print(f"Epochs:[{epoch}/{epochs}],time:{train_time:.2f}s,train_loss:{train_loss:.4f},train_acc:{train_acc:.4f},"
          f"val_loss:{test_loss:.4f},val_acc:{test_acc:.4f}")


import time


def train(model, optimizer, trainloader, testloader, epochs, lr):
    train_losses = []
    train_accs = []
    val_losses = []
    val_accs = []
    model.train()
    for epoch in range(epochs):
        train_loss = 0
        train_acc = 0
        start = time.time()
        for inputs, labels in trainloader:
            inputs, labels = inputs.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = f.cross_entropy(outputs, labels)
            acc = accuracy(outputs, labels)
            train_loss += loss.item()
            train_acc += acc
            loss.backward()
            optimizer.step()
        end = time.time()
        train_time = end - start
        train_loss /= len(trainloader)
        train_acc /= len(trainloader)
        val_loss, val_acc = validate(model, testloader)
        train_losses.append(train_loss), train_accs.append(train_acc)
        val_losses.append(val_loss), val_accs.append(val_acc)
        print_log(epoch + 1, train_time, train_loss, train_acc, val_loss, val_acc, epochs)
    return train_losses, train_accs, val_losses, val_accs


if __name__ == '__main__':
    main()
