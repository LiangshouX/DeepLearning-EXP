import math
from abc import ABC

import numpy as np
import re
from collections import Counter, defaultdict
import matplotlib.pyplot as plt
import nltk
import pandas as pd
import seaborn as sns
from Config import Config

from torch.utils.data import Dataset

from Config import Config
from utils import DataProcess

config = Config()

PAD_ID = 0  # 填充词padding的编码
PAD = '0'
NAME_TOKEN = "[NAME]"
NEAR_TOKEN = "[NEAR]"

class E2EDataset(Dataset):
    """PyTorch数据加载类，其中构造了数据处理类DataProcess的对象
    Args:

    Returns:
        None

    dataProcessor:
        >> raw_data_x(list):
    """

    def __init__(self, file_path, train_mod, attributes_vocab=None, tokenizer=None):
        self.train_mod = train_mod
        self.dataProcessor = DataProcess(file_path=file_path, train_mod=self.train_mod,
                                         attributes_vocab=attributes_vocab, tokenizer=tokenizer)

        self.max_mr_len = config.max_mr_length
        self.max_ref_len = config.max_sentence_length
        self.ref = self.dataProcessor.ref
        self.attributes_vocab = self.dataProcessor.attributes_vocab
        self.tokenizer = self.dataProcessor.tokenizer
        self.raw_data_x = self.dataProcessor.raw_data_x
        self.raw_data_y = self.dataProcessor.raw_data_y
        self.multi_data_y = self.dataProcessor.multi_data_y

    def __len__(self):
        return len(self.ref)

    def __getitem__(self, index):
        x = np.array(self.dataProcessor.sequence_padding(self.dataProcessor.tokenizer.encode(self.raw_data_x[index]),
                                                         self.max_mr_len))
        y = np.array(self.dataProcessor.sequence_padding(self.dataProcessor.tokenizer.encode(self.raw_data_y[index]),
                                                         self.max_ref_len))
        # x, y = torch.tensor(x), torch.tensor(y)
        if self.train_mod == 'train':
            return x, y
        else:
            lex = self.dataProcessor.lexicalizations[index]
            multi_y = self.multi_data_y[' '.join(self.raw_data_x[index])]
            return x, y, lex, multi_y


if __name__ == "__main__":
    train_dataSet = E2EDataset(config.root_path + config.train_data_path, train_mod='train')
    print(train_dataSet[0][0], '\n', train_dataSet[0][1])

    dev_dataSet = E2EDataset(config.root_path + config.dev_data_path,
                             train_mod='valid',
                             attributes_vocab=train_dataSet.attributes_vocab,
                             tokenizer=train_dataSet.tokenizer)
    print('\n{},\n{},\n{},\n{}'.format(dev_dataSet[3300][0],
                                       dev_dataSet[3300][1],
                                       '0',
                                       '0'))
    # print(dev_dataSet.multi_data_y)
    token_indexes = np.array([123, 45, 65, 78])
    print(train_dataSet.tokenizer.decode(token_indexes))
