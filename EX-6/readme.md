# E2E数据集Seq2Seq自然语言生成实验



## 1. E2E数据集介绍

E2E数据集是由$j.novikova$等人与2017年发表的一个用于在餐饮领域训练端到端、数据驱动的自然语言生成系统的数据集。E2E数据集含有50502条数据示例，并按照$ 76.5:8.5:15 $的比例划分训练集、验证集和测试集。数据集中含有两个部分：意义表示（Meaning Representation） $MR$ 和 自然语言参考文本（human reference text） $NL Reference$或$ref$。

一个数据的示例如下：

![image-20230524153413591](https://typora-1308640872.cos-website.ap-beijing.myqcloud.com/img/image-20230524153413591.png)

数据集中的每个MR包括有3至8个属性（也叫槽），如_name,food, area, values_等。属性-值的详细情况如下表所示：



![image-20230524153350124](https://typora-1308640872.cos-website.ap-beijing.myqcloud.com/img/image-20230524153350124.png)



## 2. 数据集处理

### 2.1 分词器Tokenizer

在自然语言处理（NLP）深度学习方法中，"Tokenizer"通常指的是分词器。分词器是一种将文本句子分割成**单个词语**或**子词**的工具。在NLP任务中，如文本分类、机器翻译或命名实体识别等，首先需要将原始文本转换成计算机可以理解和处理的形式。分词器在这个过程中起到重要的作用。通过将句子分割成单个的词语或子词，分词器为文本提供了一个基本的单位，使得后续的处理更加精确和高效。

本次实践构造分词器时需使用一个词表进行初始化,实现的分词器中除了初始化函数外共实现了4个方法：

* index_to_token：给定编号，查找词汇表中对应的词

* token_to_index：给定一个词，查询其在词汇表中的位置，设置默认值为低频词[UNK]的编号

* encode：对字符串进行编码

  encode函数的编码如下，传入参数为

  ```python
  def encode(self, tokens):
          """对字符串编码"""
          # 开始标记
          token_ids = [self.token_to_index('[BOS]'), ]
          for token in tokens:
              token_ids.append(self.token_to_index(token))
          # 结束标记
          token_ids.append(self.token_to_index('[EOS]'))
          return token_ids
  ```

  

* decode：将编码转换成字符串

  decode函数的编码如下：

  ```python
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
  ```



### 2.2 数据预处理类

​    在数据集处理类中，封装了关于数据集预处理的方法，用于将文本形式的E2E数据集处理成为适合于本次任务的数据形式。数据集以.csv形式的文件给出，在数据预处理类中使用pandas科学数据库来读取，同时，由于E2E数据集包括训练集(trainset)、验证集(devset)与测试集(testset)三个部分，而训练集与验证集均包含两列：$mr$与$ref$，测试集则只包含$mr$(原数据集中列名表示为了MR，可适当修改)，从而需要分不同情况进行读入。

#### 2.2.1 初始化函数\_\_init\_\_ 

​    考虑到训练集与验证集均是在模型训练过程中使用，且数据的结构相同，故初始化函数\_\_init\_\_中设置`train_mod`这一参数，用于在不同的场景下进行不同的读取操作。初始化函数的另外两个参数为结构化文本中的属性的词典`attributes_vocab`和分词器`tokenizer`。初始化函数的编写如下，其中使用到的功能函数`str2dict`在后续部分做详细介绍：

```python
    def __init__(self, file_path, train_mod=True, attributes_vocab=None, tokenizer=None):
        self.train_mod = train_mod
        df = pd.read_csv(file_path)
        
        if self.train_mod:
            # 训练集与验证集读取
            self.mr = str2dict(df['mr'].values.tolist())  # type: list[dict]
            self.ref = df['ref'].values.tolist()  # type: list[str]
        else:
            self.mr = str2dict(df['mr'].values.tolist())
            self.ref = ['' for _ in range(len(self.mr))]

        self.raw_data_x = []  # 存储结构化文本数据mr, feature
        self.raw_data_y = []  # 存储ref, target
        self.lexicalizations = []  # 存储去词化原词？
        self.multi_data_y = {}  # 结构化文本对应的多个ref
```

​    对于数据中的$mr$列，如前文对数据集的介绍，其形式为若干"Attribute[Value]"的类似键值对的字符串形式，如"`name[Alimentum], area[city centre], familyFriendly[no]`"等。我们希望将这样形式的数据转换为一个真正的键值对形式的字典形式的数据，于是使用如下的功能函数`str2dict`，其输入是一个存储字符串的列表，列表中具体内容则是结构化文本中$mr$列的数据。代码编写细节以及pydoc说明如下：

```python
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
        for item in map_key:  # 获取键值对 'Attribute[value]'
            key = item.split('[')[0]
            # print("In function str2dict:\t", key, end='\t')
            value = item.split('[')[1].replace(']', '')
            # print("In function str2dict:\t", value)
            _dict[key] = value
        dict_list.append(_dict)
    return dict_list
```



​    此外在数据预处理类中定义了四个功能函数，其函数名以及功能概述如下，具体各函数的实现细节将在后续部分作详细介绍：

* `preprocess`: 对数据进行预处理，执行词汇化、去词化等操作，处理后将结果存入本类的`raw_data_x`等成员变量中
* `build_vocab`:  利用描述文本$ref$构建词汇表，并利用构建的词汇表来构建分词器`tokenizer`
* `build_attribute_vocab`: 利用结构化文本的$mr$列来构建属性的词汇表，主要根据键的词频来构建
* `sequence_padding`: 对文本进行填充、超长截断等操作。



​    最后再于\_\_init\_\_函数中进行最后的处理，即根据不同的数据集调用不同的操作函数完成类的初始化过程。在后续的主函数调用时，首先加载训练集，之后再加载验证集测试集。加载训练集时会构建得到一个词典、分词器，这个词典和分词器用作加载后两个数据集时传入。

```python
        if self.train_mod:
            self.build_attributes_vocab()   # 构建属性词典
            self.preprocess()               # 数据预处理、去词化
            self.build_vocab()              # 构建文本词典
        else:
            if attributes_vocab is None or tokenizer is None:
                raise ValueError("For test set, attributes_vocab and tokenizer are necessary!")
            self.attributes_vocab = attributes_vocab
            self.key_num = len(self.attributes_vocab)
            self.tokenizer = tokenizer
            self.preprocess()
```



#### 2.2.2 数据预处理函数preprocess

```python
    def preprocess(self):
        """文本预处理函数"""
        for index in range(len(self.mr)):
            # 将结构化文本 mr 转换成长度为 key_num 的属性列表，列表不同位置对应不同的属性
            mr_data = [PAD_ID] * self.key_num
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

            # 将ref也处理成列表
            ref_data = self.ref[index]
            if self.train_mod:
                if lex[0]:
                    ref_data = ref_data.replace(lex[0], NAME_TOKEN)
                if lex[1]:
                    ref_data = ref_data.replace(lex[1], NEAR_TOKEN)
                # 正则表达式去除句子中的标点
                ref_data = list(map(lambda x: re.split(r"([.,!?\"':;)(])", x)[0], ref_data.split()))
            else:
                ref_data = ['']

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

```

在自然语言处理（NLP）中，delexicalization（去词汇化）是指将具体的词语或短语替换为抽象的占位符或符号的过程。它是将文本中的词汇信息去除或抽象化的一种操作。

### 2.3 PyTorch数据加载类





## 3. 搭建Seq2Seq模型

此部分详细代码见Model.py文件

Seq2Seq模型是用于可变长度的输入序列到可变长度的输出序列任务的经典模型，常见于机器翻译等典型任务中。Seq2Seq模型是一种经典的Encoder-Decoder结构，结构中包含一个编码器与一个解码器。编码器使⽤⻓度可变的序列作为输⼊，将其转换为固定形状的隐状态。即输⼊序列的信息被编码到循环神经⽹络编码器的隐状态中。为了连续⽣成输出序列的词元，独⽴的解码器是基于输⼊序列的编码信息和输出序列已经看⻅的或者⽣成的词元来预测下⼀个词元。

![image-20230602152421745](https://typora-1308640872.cos-website.ap-beijing.myqcloud.com/img/image-20230602152421745.png)



### 3.1 Encoder

如前所述，编码器的作用是将⻓度可变的输⼊序列编码成⼀个“状态”，以便后续对该状态进⾏解码。从技术上讲，编码器将⻓度可变的输⼊序列转换成形状固定的上下⽂变量c，并且将输⼊序列的信息在该上下⽂变量中进⾏编码。受算力的限制，这里只使用线性层来实现一个简单的Encoder。

Encoder的构造函数接受两个参数：`input_size`和`hidden_size`，分别表示输入特征的维度和编码器隐藏层的维度。

在`__init__`函数中，代码定义了一个线性层`self.W`，它使用`nn.Linear`将输入特征的维度转换为隐藏层的维度。`nn.Linear`接受一个形状为`(batch_size, input_size)`的张量作为输入，并输出一个经过线性变换后形状为`(batch_size, out_features)`的张量。同时定义一个ReLU激活函数`self.relu`。

```python
    def __init__(self, input_size, hidden_size):
        super(Encoder, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.W = nn.Linear(in_features=self.input_size,
                           out_features=self.hidden_size)
        self.relu = nn.ReLU()
```



Encoder的的前向传播函数。它接受的输入是一个已经经过词嵌入处理的张量`input_embedded`，形状为`[seq_len, batch_size, embed_dim]`。在Seq2Seq模型构建时，接受的输入为经过reshape后的原始张量数据，形状为`[seq_len, batch_size]`,对其进行词嵌入后进入Encoder进行编码。

首先，代码通过`input_embedded.size()`获取输入张量的形状信息，并将其分别赋值给`seq_len`、`batch_size`和`embed_dim`。接下来，代码将输入张量进行重塑，将其形状变为`[seq_len*batch_size, embed_dim]`，然后通过线性层`self.W`和ReLU激活函数`self.relu`对输入进行处理，得到输出张量`outputs`。最后，代码将`outputs`重新调整为形状为`[seq_len, batch_size, -1]`的张量，并通过`torch.sum`对其进行求和操作，得到编码器的隐藏状态`decoder_hidden`。最终，函数返回`outputs`和经过`unsqueeze(0)`操作后的`decoder_hidden`。

```python
    def forward(self, input_embedded):
        seq_len, batch_size, embed_dim = input_embedded.size()
        # 将词嵌入的输入 reshape 为 [seq_len*batch_size, embed_dim]
        outputs = self.relu(self.W(input_embedded.view(-1, embed_dim)))
        outputs = outputs.view(seq_len, batch_size, -1)
        decoder_hidden = torch.sum(outputs, 0)
        return outputs, decoder_hidden.unsqueeze(0)
```



需要注意的是，`input_size`需要和`embed_dim`相等。



## 00 评价指标BLEU







