##注意力机制
##缩放点积注意力（Scaled Dot-Product Attention,SDPA)
##运用多头注意力机制，将256维的查询、键和值向量分成8个头，每个头有一组32维的Q（查询）、K（键）和值向量（V）,
##每个头关注输入的不同部分，使得模型能够捕捉到更广泛的信息。
##defint The Function of Attention:
import torch
DEVICE = 'cuda'
import numpy as np
from torch import nn
from copy import deepcopy
import math
#query的维度（batch,h,seq_len_q,d_k)
def attention(query,key,value,mask=None,dropout = None):
    d_k = query.size(-1)#query.size(-1)：
    # 取query张量最后一个维度的大小，这个维度代表每个注意力头中向量的维度（记为d_k）。
    #注意：query、key的最后一维必须相同（都是d_k），否则无法计算点积。
    #通过点积计算query和key的相似度，再除以√d_k进行缩放
    #key.transpose(-2, -1)：交换key的最后两个维度（例如：原始key形状为(batch, h,seq_len_k, d_k)
    #转置后为(batch, h,d_k, seq_len_k)），目的是让query和key的维度匹配，满足矩阵乘法要求
    #除以√d_k：当d_k较大时，点积结果的方差会很大，导致 softmax 后权重过于集中（大值更突出），缩放后可让梯度更稳定
    scores = torch.matmul(query,key.transpose(-2,-1))/math.sqrt(d_k)##torch.matmul是点积，这里除以了维度来进行归一化
    if mask is not None:
        scores =scores.masked_fill(mask==0,-1e9)#将mask==0的地方转化为-1e9
    p_attn = nn.functional.softmax(scores,dim=-1)#计算softmax概率（注意力权重）
    #p_attn的形状仍为(batch,h, seq_len_q, seq_len_k)，每个元素代表第i个query对第j个key的关注权重
    if dropout is not None:
        p_attn = dropout(p_attn)
    return torch.matmul(p_attn,value),p_attn
    #torch.matmul(p_attn, value)：用注意力权重p_attn对value进行加权求和，得到注意力输出，形状为(batch, h,seq_len_q, d_k)。
class MultiHeadedAttention(nn.Module):
    def __init__(self,h,d_model,dropout=0.1):
        super().__init__()
        # 1. 检查维度是否可分：d_model必须能被头数h整除
        assert  d_model%h==0;
        # 2. 计算每个头的维度
        self.d_k=d_model/h
        self.h=h
        # 3. 创建4个线性层（用ModuleList管理）
        # 前3个：分别将query、key、value映射到多头空间
        # 第4个：将多头合并后的结果映射回d_model维度
        self.linear = nn.ModuleList([deepcopy(nn.Linear(d_model,d_model))for i in range(4)])
        self.attn =None
        self.dropout = nn.Dropout(p=dropout)

    def forward(self,query,key,value,mask=None):
        if mask is not None:
            mask =mask.unsqueeze(1)
            # 原始mask形状可能为(batch, seq_len_q, seq_len_k)
            # 增加一个维度后为(batch, 1, seq_len_q, seq_len_k)，与多头的h维度对齐（通过广播）
        nbatches = query.size(0)
        #l(x)：通过第 1 个线性层（self.linear[0]）对query进行映射（形状不变，仍为(batch, seq_len_q, d_model)）
        #view(nbatches, -1, self.h, self.d_k)：重塑为(batch, seq_len_q, h, d_k)，将d_model拆分为h个d_k（例如：512→8×64）
        #transpose(1, 2)：交换第 1 和第 2 维，得到(batch, h, seq_len_q, d_k)，让 “头” 作为独立维度，方便并行计算。
        #最终query、key、value的形状都是(batch, h, seq_len, d_k)，准备好进入注意力函数
        query,key,value=[l(x).view(nbatches,-1,self.h,self.d_k).transpose(1,2) for l,x in zip(self.linear,(query,key,value))]
        x,self.attn = attention(query,key,value,mask=mask,dropout=self.dropout)
        #这里的attention函数会同时处理所有头（因为query等已包含h维度），输出x的形状为(batch, h, seq_len_q, d_k)（每个头的注意力结果）。
        #transpose(1, 2)：将形状从(batch, h, seq_len_q, d_k)转置为(batch, seq_len_q, h, d_k)（把头的维度移到中间）。
        #contiguous()：确保张量在内存中连续（避免view操作报错）
        #view(...)：将h和d_k合并为d_model（h*d_k = d_model），最终形状为(batch, seq_len_q, d_model)。
        x=x.transpose(1,2).contiguous().view(nbatches,-1,self.h*self.d_k)
        output = self.linear[-1](x)
        return output

class PositionwiseFeedForward(nn.Module):
    def __init__(self,d_model,d_ff,dropout=0.1):
        super().__init__()
        self.w_1=nn.Linear(d_model,d_ff)##通常这里d_ff是d_model的四倍，将隐藏层扩大为模型数倍来增强模型捕获复杂特征的能力，是transformer架构的标准方法。
        self.w_2=nn.Linear(d_ff,d_model)
        self.dropout=nn.Dropout(dropout)

    def forward(self,x):
        h1=self.w_1(x)
        h2=self.dropout(h1)
        return self.w_2(h2)

class SublayerConnection(nn.Module):
    def __init__(self,size,dropout):
        super().__init__()
        self.norm = nn.LayerNorm(size)# 层归一化（这里直接用PyTorch内置的，和自定义LayerNorm功能一致）
        self.dropout = nn.Dropout(dropout)#drop层，防止过拟合

    def forward(self,x,sublayer):
        # x：子层的输入（形状：(batch_size, seq_len, d_model)）
        # sublayer：子层函数（如自注意力、前馈网络）

        # 1. 先对输入x做层归一化（Pre-LN结构，比先过子层再归一化更稳定）
        # 2. 将归一化结果传入子层（sublayer）计算
        # 3. 对子网输出做dropout
        # 4. 残差连接：输入x + 子层输出（保留原始输入信息，缓解梯度消失）
        output = x + self.dropout(sublayer(self.norm(x)))

    output = x + self.dropout(sublayer(self.norm(x)))
        return output


class EncoderLayer(nn.Module):
    def __init__(self,size,self_attn,feed_forward,dropout):
        super().__init__()
        self.self_attn = self_attn # 多头自注意力组件（q=k=v，输入序列自己和自己做注意力）
        self.feed_forward=feed_forward # 前馈网络组件（对每个位置独立做非线性变换）
        # 创建2个子层连接
        self.sublayer = nn.ModuleList([deepcopy((SublayerConnection(size,dropout))for i in range(2)])
        self.size = size  # 特征维度d_model（确保维度匹配）

    def forward(self,x,mask):
        # x：输入特征（形状：(batch_size, seq_len, d_model)）
        # mask：掩码（用于屏蔽Padding位置，形状：(batch_size, 1, seq_len)）

        # 第一个子层：多头自注意力（用lambda封装，传入自注意力所需参数）
        # self_attn(x, x, x, mask)：q=k=v=x，即自注意力
        x=self.sublayer[0](x,lambda x:self.self_attn(x,x,x,mask))
        # 第二个子层：前馈网络（直接传入前馈网络函数)
        output = self.sublayer[1](x,self.feed_forward)
        return output

#归一化层：对每个样本的每个序列位置，在特征维度上做归一化，让模型在训练时输入分布更稳定，加速收敛。
class LayerNorm(nn.Module):
    def __init__(self,features,eps=1e-6):
        super().__init__()
        #可学习的缩放参数（初始为1）和偏移参数（初始化为0）
        self.a_2 = nn.Parameter(torch.ones(features))#形状：（d_model,)
        self.b_2= nn.Parameter(torch.zeros(features))#形状：（d_model,)
        self.eps=eps#防止分母为0的微小值

    def forward(self,x):
        #x形状：（batch_size，seq_len，d_model)最后一位是特征维度
        #1.计算最后一维的均值和标准差
        mean = x.mean(-1,keepdim=True)
        std = x.std(-1,keepdim=True)
        #2.标准化：得到均值0、方差1的分布
        x_zscore = (x-mean)/torch.sqrt(std**2+self.eps)
        #3.缩放和平移:通过可学习参数恢复特征表达能力（不一定局限于0均值1方差）
        output = self.a_2*x_zscore+self.b_2
        return output

class Encoder(nn.Module):
    def __init__(self,layer,N):
        super().__init__()
        # 堆叠N个编码器层（深拷贝确保每层参数独立）
        self.layers = nn.ModuleList([deepcopy(layer)for i in range(N)])
        self.norm =LayerNorm(layer.size)

    def forward(self,x,mask):
        # x：原始输入特征（如词嵌入+位置编码，形状：(batch_size, seq_len, d_model)）
        # mask：源序列掩码（屏蔽Padding）
        # 依次经过N个编码器层（每层都做上下文建模）
        for layer in self.layers
            x=layer(x,mask) # 每层输出作为下一层输入

        # 最后做一次层归一化，稳定输出分布
        output = self.norm(x)
        return output# 输出：(batch_size, seq_len, d_model)，即memory

class DecoderLayer(nn.Module):
    def __init__(self,size,self_attn,src_attn,feed_forward,dropout):
        super().__init__()
        self.size=size
        self.self_attn=self_attn
        self.src_attn = src_attn
        self.feed_forward = feed_forward
        self.sublayer =nn.ModuleList([deepcopy(SublayerConnection(size,dropout))for i in range(3)])

    def forward(self,x,memory,src_mask,tgt_mask):
        x=self.sublayer[0](x,lambda x:self.self_attn(x,x,x,tgt_mask))
        x=self.sublayer[1](x,lambda x:self.src_attn(x,memory,memory,src_mask))
        output = self.sublayer[2](x,self.feed_forward)
        return output

class Transformer(nn.Module):
    def __init__(self,encoder,decoder,src_embed,tgt_embed,generator):
        super().__init__()
        self.encoder=encoder
        self.decoder= decoder
        self.src_ember=src_embed
        self.tgt_embed=tgt_embed
        self.generator=generator

    def encode(self,src,src_mask):
        return self.encoder(self.src_ember(src),src_mask)
    def decode(self,memory,src_mask,tgt,tgt_mask):
        return self.decoder(self.tgt_embed,memory,src_mask,tgt_mask)

    def forward(self,src,tgt,src_mask,tgt_mask):
        memory = self.encode(src,src_mask)
        output = self.decode(memory,src_mask,tgt,tgt_mask)
        return output