"""
    这个文件中可以添加数据预处理相关函数和类等
    如词汇表生成，Word转ID（即词转下标）等
    此文件为非必要部分，可以将该文件中的内容拆分进其他部分
"""
import time

import jieba
import re

# 加载停用词
# stopwords = [line.strip() for line in open('./data/baidu_stopwords.txt', 'r', encoding='utf-8').readlines()]
stopwords = ['\t', '\n', '，', ',', '。', ';', '：', '；', '：', '、', ' ', '(', ')', '（', '）', '/', '@', '#', '-']

# 正则表达式匹配英文单词或者短语
pattern_en = re.compile(r'[a-zA-Z]+[-]*[a-zA-Z]*')

def tokenize(text):
    """
    使用jieba分词对中文文本进行分词，并去除停用词
    """
    tokens = jieba.lcut(text)
    tokens = [token for token in tokens if token not in stopwords]
    return tokens

def build_vocabulary():
    """
    构建词汇表
    """
    word2idx = {"<PAD>": 0, "<UNK>": 1}
    idx2word = {0: "<PAD>", 1: "<UNK>"}
    word_freq = {}

    file_lists = ['./data/data_train.txt', './data/data_val.txt', './data/test_exp3.txt']

    for file_path in file_lists:
        with open(file_path, 'r', encoding='utf-8') as f:
            texts = f.readlines()

        for text in texts:
            # 中文分词
            tokens = tokenize(text)

            # 英文部分识别
            for i, token in enumerate(tokens):
                if not pattern_en.fullmatch(token):
                    continue
                # 匹配到英文单词或者短语
                start, end = i, i
                while start > 0 and pattern_en.fullmatch(tokens[start-1]):
                    start -= 1
                while end < len(tokens)-1 and pattern_en.fullmatch(tokens[end+1]):
                    end += 1
                # 将匹配到的英文部分合并为一个词，并加入词频统计
                en_part = ' '.join(tokens[start:end+1])
                if en_part in word_freq:
                    word_freq[en_part] += 1
                else:
                    word_freq[en_part] = 1

            # 中文部分加入词频统计
            for token in tokens:
                if not pattern_en.fullmatch(token):
                    if token in word_freq:
                        word_freq[token] += 1
                    else:
                        word_freq[token] = 1

    # 根据词频排序
    sorted_word_freq = sorted(word_freq.items(), key=lambda x: x[1], reverse=True)

    for word, freq in sorted_word_freq:
        idx = len(word2idx)
        word2idx[word] = idx
        idx2word[idx] = word

    return word2idx, idx2word

def words2index(word2idx, sentence):
    """
        将句子查表转换为索引序列
    """
    inp = []
    for ch in jieba.lcut(sentence):
        try:
            inp.append(word2idx[ch])
        except Exception as e:
            continue
    return inp

if __name__ == '__main__':
    print("数据预处理开始......")
    # with open('./data/data_train.txt', 'r', encoding='utf-8') as f:
    #     text = f.readlines()
    #     print(type(text))
    #     print(len(text))
    #     print(text[:10])
    t1 = time.time()
    w2i, i2w = build_vocabulary()

    tes_txt = '病毒性肺炎	病毒抗原或核酸检测	实验室检查	' \
              '第八章肺部感染性疾病肺炎是儿科常见病、多发病，而且有资料表明，小儿肺炎是目前我国婴幼儿死亡的首位原因，迄今仍严重威胁着小儿的生命和健康。【辅助检查】（一）特异性病原学检查病毒性肺炎早期、尤其是病程在5' \
              '天以内者，可采集鼻咽部吸出物或痰（脱落上皮细胞），进行病毒抗原或核酸检测。 '
    tes_list = jieba.lcut(tes_txt)
    # print(w2i)
    # for tmp in tes_list:
    #     print(w2i[tmp])

    inp_ = words2index(word2idx=w2i, sentence=tes_txt)
    print(inp_)

    t2 = time.time()
    print("数据预处理完毕！\n耗时{:.4f}s".format(t2-t1))
