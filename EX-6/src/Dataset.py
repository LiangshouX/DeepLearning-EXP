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


def str2dict(str_list):
    """将字符串格式的结构化文本(mr) 处理为字典格式
    Args:
        str_list(list): 字符串列表
    Returns:
        dict_list(list): 以字典形式存储的列表

    Examples:
            >> str_list_test = ["name[The Wrestlers], eatType[coffee shop], food[English]"]
            >> str2dict(str_list_test)
            >> [{'name': 'The Wrestlers', 'eatType': 'coffee shop', 'food': 'English'}]
    """
    dict_list = []
    # 分离属性(key)和值(value)
    map_keys = list(map(lambda x: x.split(', '), str_list))
    for map_key in map_keys:
        _dict = {}
        for item in map_key:  # 获取键值对 'A[a]'
            # print(item)
            key = item.split('[')[0]
            # print("In function str2dict:\t", key, end='\t')
            value = item.split('[')[1].replace(']', '')
            # print("In function str2dict:\t", value)
            _dict[key] = value
        dict_list.append(_dict)
    return dict_list


class Tokenizer:
    """分词器，构建单词与索引之间的映射，将文本句子分割成可处理单元
    Args:
        token_dict(dict): 词典
    """

    def __init__(self, token_dict):
        self.word2index = token_dict
        self.index2word = {value: key for key, value in self.word2index.items()}
        self.vocab_size = len(self.word2index)

    def index_to_token(self, token_index):
        """给定编号，查找词汇表中对应的词"""
        return self.index2word[token_index]

    def token_to_index(self, token):
        """给定一个词，查询其在词汇表中的位置，设置默认值为低频词[UNK]的编号"""
        return self.word2index.get(token, self.word2index['[UNK]'])

    def encode(self, tokens):
        """对字符串编码"""
        # 开始标记
        token_ids = [self.token_to_index('[BOS]'), ]
        for token in tokens:
            token_ids.append(self.token_to_index(token))
        # 结束标记
        token_ids.append(self.token_to_index('[EOS]'))
        return token_ids

    def decode(self, token_indexes):
        """给定序列的编号，解析成为字符串"""
        # 起止标记处理
        special_tokens = {'[BOS]', '[EOS]', '[PAD]'}

        # 解析产生的字符列表
        tokens = []
        for token_index in token_indexes:
            token = self.index_to_token(token_index)
            if token in special_tokens:
                continue
            tokens.append(token)
        return " ".join(tokens)


class E2EDataset_(Dataset):
    def __init__(self, file_path, train_mod, attributes_vocab=None, tokenizer=None):
        self.train_mod = train_mod
        self.max_mr_len = config.max_mr_length
        self.max_ref_len = config.max_sentence_length
        # print(self.train_mod)
        df = pd.read_csv(file_path)
        # print(df.info())

        if self.train_mod == 'train' or self.train_mod == 'valid':
            # 训练集与验证集读取
            self.mr = str2dict(df['mr'].values.tolist())  # type: list[dict]
            # print("self.mr finished!")
            self.ref = df['ref'].values.tolist()  # type: list[str]
        elif self.train_mod == 'test':
            self.mr = str2dict(df['MR'].values.tolist())
            self.ref = ['' for _ in range(len(self.mr))]
        else:
            print("Error! mode must be in ['train','valid','test']")
            exit(-2)

        self.raw_data_x = []  # 存储结构化文本数据mr, feature
        self.raw_data_y = []  # 存储ref, target
        self.lexicalizations = []  # 存储去词化原词？
        self.multi_data_y = {}  # 结构化文本对应的多个ref

        self.attributes_vocab = None  # 属性词典
        self.key_num = None
        self.tokenizer = None
        self.padding = None

        if self.train_mod == 'train':
            self.build_attributes_vocab()  # 构建属性词典
            self.preprocess()  # 数据预处理、去词化
            self.build_vocab()  # 构建文本词典
        else:
            if attributes_vocab is None or tokenizer is None:
                raise ValueError("For test set, attributes_vocab and tokenizer are necessary!")
            self.attributes_vocab = attributes_vocab
            self.key_num = len(self.attributes_vocab)
            self.tokenizer = tokenizer
            self.preprocess()

    def preprocess(self):
        """文本预处理函数"""
        for index in range(len(self.mr)):
            # 将结构化文本 mr 转换成长度为 key_num 的属性列表，列表不同位置对应不同的属性
            mr_data = [PAD] * self.key_num
            lex = ['', '']  # 存储当前文本的去词化原词，长度为2，分别对应name和near

            # 遍历 mr 的字典，得属性和值，利用 属性词典 得到属性名的编号，在属性列表对应位置记录 属性值
            # 最终的目的是得到描述 mr 的定长列表
            for item in self.mr[index].items():
                key = item[0]
                value = item[1]
                key_index = self.attributes_vocab[key]

                # 将结构化文本mr转换为属性列表并去词化处理
                if key == 'name':
                    mr_data[key_index] = NAME_TOKEN
                    lex[0] = value
                elif key == 'near':
                    mr_data[key_index] = NEAR_TOKEN
                    lex[1] = value
                else:
                    mr_data[key_index] = value
            # print("mr_data in preprocess:{}".format(mr_data))
            # 将ref也处理成列表
            ref_data = self.ref[index]
            if ref_data == "":
                ref_data = ['']
            else:
                if lex[0]:
                    ref_data = ref_data.replace(lex[0], NAME_TOKEN)
                if lex[1]:
                    ref_data = ref_data.replace(lex[1], NEAR_TOKEN)
                # 正则表达式去除句子中的标点
                ref_data = list(map(lambda x: re.split(r"([.,!?\"':;)(])", x)[0], ref_data.split()))

            # 将处理后的单条结构化文本数据 mr_data、参考文本数据 ref_data 以及去
            # 词化原词追加到相应的列表中，对于多个参考文本，则将其添加到字典中。
            self.raw_data_x.append(mr_data)
            self.raw_data_y.append(ref_data)
            self.lexicalizations.append(lex)
            # print(value)
            # print("mr_data:\t", mr_data)
            # print("ref_data:\t", ref_data)

            mr_data_str = ' '.join(mr_data)
            if mr_data_str in self.multi_data_y.keys():
                self.multi_data_y[mr_data_str].append(self.ref[index])
            else:
                self.multi_data_y[mr_data_str] = [self.ref[index]]

    def build_vocab(self):
        """构建词典"""
        # 统计词频
        counter = Counter()
        for item in self.raw_data_x:
            counter.update(item)
        for item in self.raw_data_y:
            counter.update(item)
        # 按照词频进行排序
        tokens_count_list = [(token, count) for token, count in counter.items()]
        tokens_count_list = sorted(tokens_count_list, key=lambda x: -x[1])

        # 去除词频的word列表
        tokens_list = ['[PAD]', '[BOS]', '[EOS]', '[UNK]'] + [token for token, count in tokens_count_list]

        token_index_dict = dict(zip(tokens_list, range(len(tokens_list))))
        # 建立分词器
        self.tokenizer = Tokenizer(token_index_dict)

    def build_attributes_vocab(self):
        """构建属性词典，对mr字段中的key 统计词频"""
        mr_key = list(map(lambda x: list(x.keys()), self.mr))  # type: list[list[str]]
        # print(mr_key)
        # 词频统计
        counter = Counter()
        for item in mr_key:
            counter.update(item)
        # 按照词频进行排序
        keys_count_list = [(key, count) for key, count in counter.items()]
        keys_count_list = sorted(keys_count_list, key=lambda x: -x[1])

        # 去除词频的key列表
        keys_list = [key for key, count in keys_count_list]
        self.attributes_vocab = dict(zip(keys_list, range(len(keys_list))))
        self.key_num = len(self.attributes_vocab)

    def sequence_padding(self, data_, max_len=config.max_sentence_length, padding=None):
        """数据填充"""
        if padding is None:
            padding = self.tokenizer.token_to_index('[PAD]')
        self.padding = padding
        # 开始填充
        padding_length = max_len - len(data_)

        if padding_length > 0:
            outputs = data_ + [padding] * padding_length
        else:
            outputs = data_[:max_len]
        return outputs

    def __len__(self):
        return len(self.ref)

    def __getitem__(self, index):
        x = np.array(self.sequence_padding(self.tokenizer.encode(self.raw_data_x[index]),
                                           self.max_mr_len))
        y = np.array(self.sequence_padding(self.tokenizer.encode(self.raw_data_y[index]),
                                           self.max_ref_len))
        # x, y = torch.tensor(x), torch.tensor(y)
        if self.train_mod == 'train':
            return x, y
        else:
            lex = self.lexicalizations[index]
            multi_y = self.multi_data_y[' '.join(self.raw_data_x[index])]
            return x, y, lex, multi_y


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
