import torch
from torch.utils.data import Dataset
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import numpy as np
import random
import math
import time
import os
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
from torch.utils.data import DataLoader
from torch.nn.utils.rnn import pad_sequence
import numpy as np
import random
import math
import time
from tqdm import tqdm
from dataloader import WikibioDataset
from model import Encoder, Decoder, Attention, Seq2Seq


class E2EDataset(Dataset):
    def __init__(self):
        pass
        '''
            __init__()函数的内容：
            1.读取数据集
            2.对读取的数据表做序列化处理
            3.构建输入(序列化后的数据表)和输出(参考文本)的词典
        '''

    def __len__(self):
        pass

    def __getitem__(self, index):
        pass
        '''
            __getitem__()函数的内容：
                根据index读取输入输出序列，并根据词典转化为对应
            的id序列
        '''


class Encoder(nn.Module):
    def __init__(self):
        super().__init__()
        pass
        '''
            在__init__()函数初始化使用的网络
        '''

    def forward(self):
        pass
        '''
            1.forward()函数接收输入的id序列，送入Encoder，编码返回最后
        一个token对应的隐藏状态用于初始化Decoder的隐藏状态
            2.如果要采用Attention，Encoder需返回所有token对应的隐藏状态
        '''


class Decoder(nn.Module):
    def __init__(self):
        super().__init__()
        pass
        '''
            在__init__()函数初始化使用的网络
        '''

    def forward(self):
        pass
        '''
            1.forward()函数以<sos>对应的id作为第一个输入，t时刻的输入为
        t-1时刻的输出，解码出预测序列第t个token
            2.解码过程迭代至预测出<eos>这一token，或者达到预设最大长度结束
            3.如果采用Attention，需要在Decoder计算Attention Score
        '''


def train(model, iterator):
    model.train()
    for i, batch in enumerate(iterator):
        pass


def evaluate(model, iterator):
    model.eval()
    with torch.no_grad():
        for i, batch in enumerate(iterator):
            pass


if __name__ == '__main__':
    # 读取数据
    train_set = E2EDataset()
    dev_set = E2EDataset()
    train_loader = DataLoader(train_set)
    dev_loader = DataLoader(dev_set)

    # 初始化batch大小，epoch数，损失函数，优化器等
    batch_size = 1
    model = None
    criterion = None
    optimizer = None
    MAX_EPOCH = 1

    # 训练集用于训练，验证集用于评估
    for epoch in range(MAX_EPOCH):
        train()
        evaluate()
        pass

    # 将测试集的预测结果按要求保存在txt文件中
    test_set = E2EDataset()
    test_loader = DataLoader(test_set)
    model.eval()
    with torch.no_grad():
        for i, batch in enumerate(test_loader):
            pass

