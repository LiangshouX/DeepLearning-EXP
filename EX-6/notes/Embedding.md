# 词嵌入

## 1. 什么是词嵌入？

​	自然语⾔是⽤来表达⼈脑思维的复杂系统。在这个系统中，词是意义的基本单元。

​	顾名思义，词向量则是⽤于表示单词意义的向量，并且还可以被认为是单词的**特征向量**或**表示**。将单词映射到实向量的技术称为词嵌⼊。



## 2. PyTorch中怎么做词嵌入？

在PyTorch中执行词嵌入操作主要有以下几种；

### 2.1 使用nn.Embedding模块

​	`nn.Embedding`是PyTorch中用于实现词嵌入的模块之一。它将离散的**词语索引**映射到连续的**词嵌入向量**，从而将词语表示为密集的数值向量。这种映射过程使得模型能够更好地理解和处理文本数据。

​	初始化`nn.Embedding`的语法为`embedding = nn.Embedding(vocab_size, embed_dim)`，在初始化`nn.Embedding`时，需要指定两个参数：

* `vocab_size`：词汇表的大小，即不同词语的总数。

- `embed_dim`：词嵌入向量的维度，即每个词语被表示为多少维的向量。



​	`nn.Embedding`的输入是一个整数张量，表示词语的索引。该张量的形状通常为`(batch_size, sequence_length)`，其中`batch_size`为批次大小，`sequence_length`为输入序列的长度。

​	`nn.Embedding`的输出是词嵌入向量的张量。输出张量的形状与输入张量相同，即`(batch_size, sequence_length, embed_dim)`，其中`embed_dim`为词嵌入的维度。

​	例如在下面的代码中，输入一个形状为 [2, 3] 的张量，表示`batch_size`为2，`sequence_length`为3的一个数据，进行词嵌入操作，即可得到一个词嵌入向量，维度为`torch.Size([2, 3, 4])`。

```python
embedding=nn.Embedding(10, 4)
input_data = torch.tensor([[1, 2, 3], [4, 5, 6]])  # 替换为实际的词语索引
embeddings = embedding(input_data)
```



**注意事项：**

- 在使用`nn.Embedding`时，通常需要在模型训练过程中学习到词嵌入的权重。可以通过反向传播来更新这些权重，从而优化模型在特定任务上的性能。
- 在训练过程中，可以使用梯度下降等优化算法来调整词嵌入权重，使得词嵌入能够更好地捕捉词语之间的语义关系。
- 在使用`nn.Embedding`之前，需要根据任务和数据集的特点选择合适的词嵌入维度和词汇表大小。

​	通过使用`nn.Embedding`，可以在PyTorch中轻松地实现词嵌入，并将文本数据转换为连续的词嵌入向量，为模型提供更丰富、更有效的输入表示。



​	以下是一个实际的使用案例：

```python
import torch
import torch.nn as nn

instr = "词嵌入怎么用？词嵌入怎么用？词嵌入怎么用？"
print("原句子为：", instr)

# 该查找表即为极简词汇表(vocabulary)
lookup_table = list("啊窝饿词入嵌怎用么？")
print("词汇表（极简版）为：", lookup_table, len(lookup_table))

inp = []
for ch in instr:
    inp.append(lookup_table.index(ch))
print("经过查找表之后，原句子变为：", inp, len(inp))

inp = torch.tensor(inp)
embedding_dim = 3
emb = nn.Embedding(
    num_embeddings=len(lookup_table),
    embedding_dim=embedding_dim)

print("inp:\t", inp)

print("最终结果：")
print(emb(inp), emb(inp).size())
print("词嵌入就是这样用的！")
```

​	输出如下：

```
原句子为： 词嵌入怎么用？词嵌入怎么用？词嵌入怎么用？
词汇表（极简版）为： ['啊', '窝', '饿', '词', '入', '嵌', '怎', '用', '么', '？'] 10
经过查找表之后，原句子变为： [3, 5, 4, 6, 8, 7, 9, 3, 5, 4, 6, 8, 7, 9, 3, 5, 4, 6, 8, 7, 9] 21
inp:	 tensor([3, 5, 4, 6, 8, 7, 9, 3, 5, 4, 6, 8, 7, 9, 3, 5, 4, 6, 8, 7, 9])
最终结果：
tensor([[ 2.0954,  0.4945, -1.6357],
        [ 0.0480, -0.6967,  0.1153],
        [-0.3985,  1.0209, -2.1598],
        [-0.1714,  0.3929,  0.3688],
        [-0.7504,  0.0922, -0.9730],
        [ 0.6285, -0.4498, -1.4877],
        [ 1.6779,  0.6328, -0.6207],
        [ 2.0954,  0.4945, -1.6357],
        [ 0.0480, -0.6967,  0.1153],
        [-0.3985,  1.0209, -2.1598],
        [-0.1714,  0.3929,  0.3688],
        [-0.7504,  0.0922, -0.9730],
        [ 0.6285, -0.4498, -1.4877],
        [ 1.6779,  0.6328, -0.6207],
        [ 2.0954,  0.4945, -1.6357],
        [ 0.0480, -0.6967,  0.1153],
        [-0.3985,  1.0209, -2.1598],
        [-0.1714,  0.3929,  0.3688],
        [-0.7504,  0.0922, -0.9730],
        [ 0.6285, -0.4498, -1.4877],
        [ 1.6779,  0.6328, -0.6207]], grad_fn=<EmbeddingBackward0>) torch.Size([21, 3])
词嵌入就是这样用的！
```





### 2.2 使用预训练的词向量模型

```python
import torch
from gensim.models import KeyedVectors

# 加载预训练的词向量模型
model = KeyedVectors.load_word2vec_format('path/to/pretrained_model.bin', binary=True)

# 获取词语的词向量
word = '中国'
embedding = torch.tensor(model[word])
```







### 2.3 使用预训练的词向量作为初始权重

```python
import torch
import torch.nn as nn

# 加载预训练的词向量模型
pretrained_embeddings = torch.tensor(load_pretrained_embeddings())

# 定义词嵌入层，并设置权重为预训练的词向量
embedding = nn.Embedding.from_pretrained(pretrained_embeddings)
```

