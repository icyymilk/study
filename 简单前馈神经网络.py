import torch
import pandas as pd
import matplotlib.pylab as plt
from PIL import Image  ## Pillow包 读取图片
import numpy as np  ## 数组处理包
from itertools import islice
from torchvision import datasets, transforms
from torch.utils.data import random_split
import time
import torch
import torch.nn as nn
from torchsummary import summary
from matplotlib import pyplot as plt
from torchvision.utils import make_grid

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
#图片读取器
def default_loader(path):
    return Image.open(path)
#定义Dataset类
class Dataset:
    def __init__(self,loader = default_loader,transform=None):
        with open('Facescore.csv','r') as f:  ##读取文件并将每行切分
            imgs=[]   #创建一个空集合
            for line in islice(f,1,None):  #f为文件，从第2行开始读到最后一行，None指定最后一行
                line = line.strip('\n')  #去除每行位的换行符，strip去除最后一位
                line = line.split(',')  #以，为分割依据分割每一行，CSV分割符为，
                im = 'SCUT-FBP5500_v2/images/' + line[0]   #im为链接
                imgs.append((im,float(line[1])))  ##最后imgs内含有的是链接+标注的二维数组
        self.imgs = imgs  #imgs分配给类
        self.loader = loader  #读取器分配给类，图片读取器
        self.transform = transform #转换分配给类
    def __len__(self):   #类的长度
        return len(self.imgs)
    def __getitem__(self,index): #类的调用
        images,labels=self.imgs[index]   #将img中的第index个的链接和标注调取出来，并进行读取和转换
        img=self.loader(images)
        img=self.transform(img)
        return img,labels


class LinearRegresson(nn.Module): ###线性回归部分 使用nn.Module来继承
    def __init__(self):    #初始化
        super(LinearRegresson,self).__init__()   #继承父类（nn.Module)必须进行的操作
        self.layer1 = nn.Linear(128*128*3,1)   #部署第一层神经网络,指定输入维度和输出维度
    def forward(self,x):#前馈神经网络
        x = x.reshape(-1,128*128*3)#将输入的张量拉直,-1表示自动适应batch的数量,此处为64
        x = self.layer1(x)  #输入第一层得到输出
        return x

def mse_metric(outputs,labels):  #计算均方误差,输入量为输出和原评分
    return torch.sum(pow(outputs.view(-1)-labels,2))/len(outputs.view(-1))
def validate(model,val_loader):  #验证集检验模型效果
    val_loss = 0  #初始化均方误差和损失函数
    val_mse = 0
    model.eval()
    for inputs,labels in val_loader:  #循环
        inputs,labels = inputs.to(device),labels.to(device) #先将输入和输出添加到GPU里,进行CUDA加速
        outputs = model(inputs)  #进入神经网络计算,得到输出结果
        loss = torch.nn.MSELoss()(outputs.view(-1),labels.to(torch.float32))  #使用内置的torch.nn.MSEloss()函数计算每一批(batch)损失函数,这里使用均方误差作为损失函数
        val_loss += loss.item()  #item可以将loss的数值导出来
        mse = mse_metric(outputs,labels)  #计算均方误差
        val_mse += mse
    val_loss /= len(val_loader)  #对总和取均值,计算平均损失和平均MSE
    val_mse /= len(val_loader)
    return val_loss,val_mse

def print_log(epoch,train_time,train_loss,train_mse,val_loss,val_mse,epochs=10):  #打印函数
    print(f"Epoch[{epoch}/{epochs}],time:{train_time:.2f}s,loss:{train_loss:.4f},mse:{train_mse:.4f},val_loss:{val_loss:.4f},val_mse:{val_mse:.4f}")

def train(model,optimizer,train_loader,val_loader,epochs):   #训练主体,初始化损失函数和均方误差
    train_losses = []
    train_mses=[]
    val_losses=[]
    val_mses=[]
    model.train()  #将模型调整至训练模式
    for epoch in range(epochs):  #epoch指迭代数量,这里一共十次迭代(训练包含前馈计算和反向传播)
        train_loss = 0
        train_mse = 0
        start = time.time()#记录初始时间
        for inputs,labels in train_loader:  #对加载器中的每一批样本进行循环
            inputs,labels = inputs.to(device),labels.to(device)   #导入GPU
            optimizer.zero_grad()  #梯度归零
            outputs = model(inputs)  #输入至神经网络得到输出
            loss = nn.MSELoss()(outputs.view(-1),labels.to(torch.float32))  #计算损失函数
            train_loss +=loss.item()
            mse = mse_metric(outputs,labels)
            train_mse+=mse
            loss.backward()   #损失函数反向传播
            optimizer.step()  #根据计算出的梯度进行迭代
        end = time.time()  #记录结束时间
        train_time = end-start  #计算每次迭代时间
        train_loss /= len(train_loader)  #计算平均训练损失和MSE
        train_mse /= len(train_loader)
        val_loss,val_mse=validate(model,val_loader)
        train_losses.append(train_loss);train_mses.append(train_mse)  #将每次迭代的训练和验证机损失,MSE都记录在一个list里
        val_losses.append(val_loss);val_mses.append(val_mse)
        print_log(epoch+1,train_time,train_loss,train_mse,val_loss,val_mse,epochs=epochs) #打印
    return train_losses,train_mses,val_losses,val_mses

def main():##主函数

    MasterFile = pd.read_csv('Facescore.csv')   #读入文件
    print(MasterFile.shape)
    MasterFile.hist()##描述成绩直方图
    transform = transforms.Compose([
        transforms.Resize((128,128)),
        transforms.ToTensor()
    ])   ##对转换进行定义 重新调整尺寸为128*128，并转变为张量
    full_data = Dataset(transform=transform)  #将full_data定义为Dataset类，这里类返回的实际是一个数组
    print("1")
    train_size = int(len(full_data)*0.8)  #划分训练集和测试集长度
    val_size = len(full_data)-train_size
    #随机划分训练集和测试集，函数random_spilt随机划分数组
    train_set,val_set=random_split(full_data,[train_size,val_size])


    batch_size = 64#设置每次抽取的样本为64个,调试训练集和测试集的加载器,训练集打乱,测试集不打乱,8核并行计算
    train_loader = torch.utils.data.DataLoader(train_set,batch_size=batch_size,shuffle=True,num_workers=8)
    val_loader = torch.utils.data.DataLoader(val_set,batch_size=batch_size,shuffle=False,num_workers=8)


    images,labels = next(iter(train_loader))  #设置一个基于train_loader的迭代器
    print(images.shape)
    print(labels.shape)
    plt.figure(figsize=(12,20))
    plt.axis('off')
    img_grid = make_grid(images, nrow=8)
    img_grid = img_grid.permute(1, 2, 0)
    # 显示图像
    plt.imshow(img_grid.detach().numpy())  # 转换为 NumPy 数组并显示
    plt.show()


    IMSIZE=128
    linear_regression_model = LinearRegresson().cuda()  #将神经网络调试至cuda加速
    summary(linear_regression_model,(3,IMSIZE,IMSIZE))   #对模型进行总结


    lr = 1e-3  #指定学习率和迭代次数
    epochs = 10
    optimizer = torch.optim.Adam(linear_regression_model.parameters(),lr=lr) #调试优化器
    history = train(linear_regression_model,optimizer,train_loader,val_loader,epochs)  #进行训练
    dataiter = iter(val_loader)
    images,labels=dataiter.next()
    img = images[4].permute((1,2,0))
    lbl = imshow(img)
    plt.imshow(img)
    img = torch.from_numpy(img.numpy())
    img = img.reshape(1,128*128*3)
    with torch.no_grad():
        output = linear_regression_model.forward(img.to(device))
    prediction = float(output)
    print(f"神经网络预测为{prediction},实际为{lbl}")

if __name__ == '__main__':
    main()
