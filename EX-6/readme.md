# E2E数据集Seq2Seq自然语言生成实验



## 1. E2E数据集介绍

E2E数据集是由$j.novikova$等人与2017年发表的一个用于在餐饮领域训练端到端、数据驱动的自然语言生成系统的数据集。E2E数据集含有50502条数据示例，并按照$ 76.5:8.5:15 $的比例划分训练集、验证集和测试集。数据集中含有两个部分：意义表示（Meaning Representation） $MR$ 和 自然语言参考文本（human reference text） $ NL Reference $或$ ref $。

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
        
        if self.train_mod=='train' or self.train_mod=='valid':
            # 训练集与验证集读取
            self.mr = str2dict(df['mr'].values.tolist())  # type: list[dict]
            self.ref = df['ref'].values.tolist()  # type: list[str]
        elif self.train_mod=='test':
            self.mr = str2dict(df['mr'].values.tolist())
            self.ref = ['' for _ in range(len(self.mr))]
        else:
            print("Error! mode must be in ['train','valid','test']")
            exit(-2)

        self.raw_data_x = []  # 存储结构化文本数据mr, feature
        self.raw_data_y = []  # 存储ref, target
        self.lexicalizations = []  # 存储去词化原词
        self.multi_data_y = {}  # 结构化文本对应的多个ref
```

​    对于数据中的$mr$列，如前文对数据集的介绍，其形式为若干`Attribute[Value]`的类似键值对的字符串形式，如"`name[Alimentum], area[city centre], familyFriendly[no]`"等。我们希望将这样形式的数据转换为一个真正的键值对形式的字典形式的数据，于是使用如下的功能函数`str2dict`，其输入是一个存储字符串的列表，列表中具体内容则是结构化文本中$mr$列的数据。代码编写细节以及pydoc说明如下：

```python
def str2dict(str_list):
    """将字符串格式的结构化文本(mr) 处理为字典格式
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
            value = item.split('[')[1].replace(']', '')
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
        if self.train_mod=='train':
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

首先，代码通过遍历mr的字典，获取属性和对应的值。通过属性词典（attributes_vocab）找到属性名的编号（key_index），并在属性列表（mr_data）的对应位置记录属性值。属性列表的长度为key_num，不同位置对应不同的属性。其中，如果属性名是'name'，则将对应位置的属性值设置为特殊的NAME_TOKEN，如果属性名是'near'，则将对应位置的属性值设置为特殊的NEAR_TOKEN。属性值会被保存在lex列表中，分别对应name和near的去词化原词，这里的去词汇化delexicalization是指在自然语言处理（NLP）中，将具体的词语或短语替换为抽象的占位符或符号的过程。它是将文本中的词汇信息去除或抽象化的一种操作。

```python
def preprocess(self):
        for index in range(len(self.mr)):
            mr_data = [PAD_ID] * self.key_num
            lex = ['', '']  
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
```



接下来，代码处理参考文本数据。首先，将ref_data初始化为self.ref[index]的值，如果该值为空字符串，则将其设置为包含一个空字符串的列表。然后，如果lex[0]非空（即name存在），则将ref_data中的lex[0]替换为NAME_TOKEN；如果lex[1]非空（即near存在），则将ref_data中的lex[1]替换为NEAR_TOKEN。最后，代码使用正则表达式去除句子中的标点符号，并将处理后的句子切分成单词列表。

```python
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
```



最后，代码将处理后的结构化文本数据（mr_data）、参考文本数据（ref_data）以及去词化原词（lex）追加到相应的列表中。如果多个结构化文本具有相同的mr_data_str（通过将mr_data转换为字符串），则将相应的参考文本追加到字典的值列表中。如果mr_data_str不存在于字典的键中，则将mr_data_str作为键，将参考文本作为值列表添加到字典中。

```python
            self.raw_data_x.append(mr_data)
            self.raw_data_y.append(ref_data)
            self.lexicalizations.append(lex)
            mr_data_str = ' '.join(mr_data)
            if mr_data_str in self.multi_data_y.keys():
                self.multi_data_y[mr_data_str].append(self.ref[index])
            else:
                self.multi_data_y[mr_data_str] = [self.ref[index]]
```



#### 2.2.2 构建词汇表函数build_vocab

构建词汇表，并利用构建的词汇表来构建分词器`tokenizer`。

首先使用Counter对象统计self.raw_data_x和self.raw_data_y中的词频。self.raw_data_x是一个包含结构化文本数据的列表，self.raw_data_y是一个包含参考文本数据的列表。

接下来，将词频统计结果按照词频进行排序，得到tokens_count_list，其中每个元素是一个包含词和对应词频的元组。

然后，将特殊的标记符号（'[PAD]', '[BOS]', '[EOS]', '[UNK]'）添加到tokens_list中，标记符号依次表示填充token、开始token、结束token、未知或低频token，并将tokens_count_list中的词按照词频排序后依次添加到tokens_list中。

最后创建一个token_index_dict字典，其中键为词（tokens_list中的元素），值为该词在tokens_list中的索引，并使用token_index_dict来构建一个分词器（Tokenizer），该分词器将词转换为对应的索引值。这个分词器可以用于将文本数据转换为模型可以理解的数字表示形式，Tokenizer的实现见前文。

```python
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
```



#### 2.2.3 构建属性词典函数build_attributes_vocab

`build_attributes_vocab`函数用于根据数据中的结构化文本$ mr $中的属性值来构建属性词典（attributes_vocab），用于记录结构化文本中的属性名（key）以及对应的索引。

首先使用map函数和lambda表达式，对self.mr中的每个元素（字典）应用list(x.keys())，将每个字典的键转换为列表。这样得到的mr_key是一个包含多个列表的列表，每个子列表包含一个结构化文本的所有属性名。

接下来，使用Counter对象对mr_key中的属性名进行词频统计，之后按照词频进行排序，得到keys_count_list，其中每个元素是一个包含属性名和对应词频的元组。

然后，将keys_count_list中的属性名按照词频排序后依次添加到keys_list中。

最后，代码创建一个attributes_vocab字典，其中键为属性名（keys_list中的元素），值为该属性名在keys_list中的索引，并代码将attributes_vocab的长度赋值给self.key_num，表示属性词典中不同属性的数量。

```python
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
```



#### 2.2.4 数据填充函数sequence_padding

数据填充函数用于对数据进行填充和截断，确保数据具有相同的长度。函数接受三个参数：

* `data_`：待填充的数据

* `max_len`：最大长度，默认值为config.max_sentence_length

* `padding`：填充的标记，默认为None

首先，判断如果padding为None，则将其设置为特殊标记符号'[PAD]'在分词器（tokenizer）中对应的索引。这样可以保证在填充时使用相同的填充标记。之后将padding赋值给self.padding，以便在其他方法中可以访问到填充标记。

之后，代码计算需要填充的长度（padding_length），即max_len减去数据data_的长度，并根据填充长度的情况进行填充操作。如果padding_length大于0，即数据长度小于max_len，就在数据的末尾添加padding元素（填充标记）若干次，使数据长度达到max_len。如果padding_length小于等于0，即数据长度大于等于max_len，就截取数据的前max_len个元素。最后代码返回填充后的数据（outputs）。

```python
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
```



### 2.3 PyTorch数据加载类

此部分定义了一个自定义的PyTorch数据加载类`E2EDataset`，用于加载数据并准备进行训练或推断。

在类的初始化方法`__init__`中，接收以下参数：file_path（数据文件路径）、train_mod（训练模式）、attributes_vocab（属性词典，默认为None）、tokenizer（分词器，默认为None）。

初始化方法中创建了一个`DataProcess`类的对象（dataProcessor），并将输入的参数传递给该对象进行数据处理。通过dataProcessor可以获得处理后的数据和相关信息，如self.ref（参考文本数据）、self.attributes_vocab（属性词典）、self.tokenizer（分词器）、self.raw_data_x（结构化文本数据）、self.raw_data_y（参考文本数据）和self.multi_data_y（多个参考文本数据的字典）。

类中定义了两个方法：`__len__`和`__getitem__`。`__len__`方法返回数据集的长度，即参考文本数据的数量。`__getitem__`方法用于获取指定索引的数据样本。首先，根据索引获取对应的原始结构化文本数据（raw_data_x）和参考文本数据（raw_data_y）。然后，使用dataProcessor中的分词器（tokenizer）对原始文本进行编码（encode），并使用dataProcessor中的数据填充方法（sequence_padding）对编码后的数据进行填充。填充后的数据被转换为NumPy数组（np.array）。最后，如果训练模式为'train'，则返回结构化文本数据和参考文本数据。否则，返回结构化文本数据、参考文本数据、去词化原词（lex）和多个参考文本数据的字典（multi_y）。

```python
    def __getitem__(self, index):
        x = np.array(self.dataProcessor.sequence_padding(
            self.dataProcessor.tokenizer.encode(self.raw_data_x[index]), 
            self.max_mr_len))
        y = np.array(self.dataProcessor.sequence_padding(
            self.dataProcessor.tokenizer.encode(self.raw_data_y[index]), 
            self.max_ref_len))
        if self.train_mod == 'train':
            return x, y
        else:
            lex = self.dataProcessor.lexicalizations[index]
            multi_y = self.multi_data_y[' '.join(self.raw_data_x[index])]
            return x, y, lex, multi_y
```



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



### 3.2 Decoder

实现解码器的结构。在类的初始化方法`__init__`中，接收以下参数：

* input_size（解码器输入特征的维度，即输入张量的最后一个维度大小）
* hidden_size（解码器隐藏状态的维度，即GRU单元的输出大小）、output_size（解码器输出维度，即目标（ref文本）的词表大小）
* embedding_dim（词嵌入维度）
* encoder_hidden_size（编码器隐藏层输出维度）。

初始化方法中定义了解码器的各个组件：一个GRU层（self.rnn），一个注意力机制（self.attn），一个线性层（self.W_combine），一个线性层（self.W_out），和一个对数softmax函数（self.log_softmax）。

```python
class Decoder(nn.Module):
    def __init__(self, input_size, hidden_size, output_size, embedding_dim, encoder_hidden_size):
        super(Decoder, self).__init__()
        self.rnn = nn.GRU(input_size, hidden_size, bidirectional=False)
        self.attn = Attention(encoder_hidden_size, hidden_size)
        self.W_combine = nn.Linear(embedding_dim + encoder_hidden_size, hidden_size)
        self.W_out = nn.Linear(hidden_size, output_size)
        self.log_softmax = nn.LogSoftmax(dim=1)
```



在前向传播方法（forward）中，接收三个输入参数：prev_y_batch（前一个时间步的输出）、prev_h_batch（前一个时间步的隐藏状态）、encoder_outputs_batch（编码器的输出）。

首先，通过注意力机制（self.attn）计算注意力权重（attn_weights），并利用这些权重对编码器的输出（encoder_outputs_batch）进行加权求和，得到上下文向量（context）。

然后，将前一个时间步的输出（prev_y_batch）和上下文向量（context）进行拼接，形成新的输入向量（y_ctx）。

接下来，将新的输入向量（y_ctx）通过线性层（self.W_combine）进行变换，得到GRU的输入（rnn_input）。

将GRU的输入（rnn_input）和前一个时间步的隐藏状态（prev_h_batch）作为输入传入GRU层（self.rnn）。GRU的输出包括每个时间步的隐藏状态输出（dec_rnn_output）和最后一个时间步的隐藏状态（dec_hidden）。

最后，将GRU的输出（dec_rnn_output）通过线性层（self.W_out）进行变换，并应用对数softmax函数（self.log_softmax）得到解码器的输出（dec_output）。

```python
    def forward(self, prev_y_batch, prev_h_batch, encoder_outputs_batch):
        attn_weights = self.attn(prev_h_batch, encoder_outputs_batch)
        context = torch.bmm(attn_weights.unsqueeze(1), encoder_outputs_batch.transpose(0, 1))
        y_ctx = torch.cat((prev_y_batch, context.squeeze(1)), 1)       
        rnn_input = self.W_combine(y_ctx)
        dec_rnn_output, dec_hidden = self.rnn(rnn_input.unsqueeze(0), prev_h_batch)
        unnormalized_logits = self.W_out(dec_rnn_output[0])
        dec_output = self.log_softmax(unnormalized_logits)
        return dec_output, dec_hidden, attn_weights
```



这个解码器类用于实现解码器的前向传播过程，将输入的前一个时间步的输出和隐藏状态与编码器的输出进行处理，得到解码器的输出和注意力权重。



### 3.3 Attention

Attention模块用于在解码器中计算注意力权重，以便将编码器的不同部分的信息聚焦到解码器的当前步骤上。

在初始化方法中，Attention模块接收编码器隐藏层输出维度（encoder_hidden_dim）和解码器隐藏状态的维度（decoder_hidden_dim）。它还可以选择性地接收注意力权重的维度（attn_dim）。

在前向传播方法中，Attention模块接收解码器的先前隐藏状态（prev_h_batch）和编码器的输出（enc_outputs）。这里的enc_outputs是一个三维张量，形状为[seq_len, batch_size, encoder_hidden_dim]，其中seq_len表示编码器输出的序列长度，batch_size表示批量大小，encoder_hidden_dim表示编码器隐藏层输出的维度。

首先，通过线性变换U，将编码器的输出enc_outputs进行变换，使其维度变为[self.h_dim * self.num_directions, self.a_dim]，其中self.h_dim表示编码器隐藏层输出的维度，self.num_directions为1（因为不考虑双向编码器），self.a_dim表示注意力权重的维度。

然后，通过线性变换W，将解码器的先前隐藏状态prev_h_batch进行变换，使其维度变为[self.s_dim, self.a_dim]，其中self.s_dim表示解码器隐藏状态的维度。

接下来，使用unsqueeze(0)将变换后的解码器隐藏状态变为三维张量，形状为[1, batch_size, self.a_dim]，以便与编码器输出uh进行相加。

将变换后的解码器隐藏状态进行扩展，使其与编码器输出uh的形状相同。

将扩展后的解码器隐藏状态和编码器输出进行元素级相加，并通过tanh函数进行激活，得到wquh。

通过线性变换v，将wquh进行变换，使其维度变为[batch_size, src_seq_len]，其中src_seq_len表示编码器输出的序列长度。

最后，通过softmax函数对得到的注意力得分进行归一化，得到注意力权重attn_weights。attn_weights的形状为[batch_size, src_seq_len]，表示每个样本在编码器输出的每个位置上的注意力权重。

注意力权重attn_weights被返回，以便在解码器的后续步骤中使用。

```python

class Attention(nn.Module):
    def __init__(self, encoder_hidden_dim, decoder_hidden_dim, attn_dim=None):
        super(Attention, self).__init__()
        self.num_directions = 1     # 不考虑双向编码器
        self.h_dim = encoder_hidden_dim
        self.s_dim = decoder_hidden_dim
        self.a_dim = self.s_dim if attn_dim is None else attn_dim
        # 构建注意力
        self.U = nn.Linear(self.h_dim * self.num_directions, self.a_dim)
        self.W = nn.Linear(self.s_dim, self.a_dim)
        self.v = nn.Linear(self.a_dim, 1)
        self.tanh = nn.Tanh()
        self.softmax = nn.Softmax()

    def forward(self, prev_h_batch, enc_outputs):
        src_seq_len,batch_size,enc_dim = enc_outputs.size()
        uh = self.U(enc_outputs.view(-1, self.h_dim)).
        		view(src_seq_len,batch_size, self.a_dim)  
        wq = self.W(prev_h_batch.view(-1, self.s_dim)).unsqueeze(0)  
        wq3d = wq.expand_as(uh)
        wquh = self.tanh(wq3d + uh)
        attn_unnorm_scores = self.v(wquh.view(-1, self.a_dim)).
        		view(batch_size, src_seq_len)
        attn_weights = self.softmax(attn_unnorm_scores)  
        return attn_weights
```





### 3.4 Seq2Seq

Seq2Seq模型的定义，它包含了编码器（Encoder）和解码器（Decoder）。

在初始化方法中，Seq2Seq模型接收了一些配置信息（config）、设备信息（device）、源语言词表大小（src_vocab_size）和目标语言词表大小（tgt_vocab_size）。

```python
class Seq2Seq(nn.Module):
    
    def __init__(self, config, device, src_vocab_size, tgt_vocab_size):
        super(Seq2Seq, self).__init__()
        self.device = device  # 设备
        self.config = config
        self.src_vocab_size = src_vocab_size
        self.tgt_vocab_size = tgt_vocab_size
        # 构建词嵌入层
        self.embedding_mat = nn.Embedding(src_vocab_size, config.embedding_dimension, padding_idx=PAD_ID)
        self.embedding_dropout_layer = nn.Dropout(config.dropout)
        # 构建编码器和解码器
        self.encoder = Encoder(input_size=config.encoder_input_size,
                               hidden_size=config.encoder_hidden_size)

        self.decoder = Decoder(input_size=config.decoder_input_size,
                               hidden_size=config.decoder_hidden_size,
                               output_size=tgt_vocab_size,
                               embedding_dim=config.embedding_dimension,
                               encoder_hidden_size=config.encoder_hidden_size)
```

在前向传播方法中，Seq2Seq模型接收输入数据（data），其中包含源语言数据和目标语言数据（batch_x_var和batch_y_var）。首先，将源语言数据通过词嵌入层（embedding_mat）进行词嵌入得到编码器的输入（encoder_input_embedded）。然后，将编码器的输入传递给编码器（encoder）进行编码，得到编码器的输出（encoder_outputs）和最后一个时间步的隐藏状态（encoder_hidden）。

接下来，根据目标语言数据的长度和批量大小，初始化解码器的隐藏状态（dec_hidden）和解码器的输入（dec_input）。然后，通过一个循环，依次解码目标语言的每个词。在每个解码步骤中，将上一个输出的词嵌入（prev_y）传递给解码器（decoder）进行解码，得到解码器的输出（dec_output）、更新后的解码器隐藏状态（dec_hidden）和注意力权重（attn_weights）。将解码器的输出记录到logits中，用于计算损失函数。同时，将解码器的输出作为下一个解码步骤的输入（dec_input），继续进行解码。

具体来说，在前向传播方法中：

1. 首先，输入数据被拆分为源语言数据（batch_x_var）和目标语言数据（batch_y_var）。
2. 源语言数据通过词嵌入层（embedding_mat）进行词嵌入操作，得到编码器的输入（encoder_input_embedded）。词嵌入操作将每个词的索引转换为一个词向量表示。
3. 词嵌入后，编码器（encoder）接收编码器的输入，进行编码操作。编码器使用GRU单元，将输入序列逐步编码为一系列隐藏状态。编码器输出包括编码器的输出（encoder_outputs）和最后一个时间步的隐藏状态（encoder_hidden）。
4. 解码器的初始化：
   - 解码器的隐藏状态（dec_hidden）使用编码器最后一个时间步的隐藏状态（encoder_hidden）来初始化。
   - 解码器的输入（dec_input）初始化为起始符号的索引（BOS_ID）。
5. 循环解码：
   - 在每个解码步骤中，首先将解码器的输入词嵌入（prev_y）传递给解码器（decoder）进行解码操作。解码器使用GRU单元，结合上一个输出的词嵌入和解码器的隐藏状态，生成当前时间步的输出。
   - 解码器的输出经过线性变换（self.W_out）和对数softmax函数（self.log_softmax），得到当前时间步的输出概率分布（dec_output）。
   - 解码器的输出概率分布记录在logits中，用于计算损失函数。
   - 当前时间步的目标词（batch_y_var[di]）作为下一个解码步骤的输入（dec_input），继续解码。
6. 返回logits作为模型的输出。



```python
def forward(self, data):
        """
        Args:
            data(tuple): (source, target)
        Returns:
            [seq_len, batch_size, vocab_size]
        """
        batch_x_var, batch_y_var = data     # [seq_len, batch_size] * 2
        # 词嵌入
        # [seq_len, batch_size, embed_dim]
        encoder_input_embedded = self.embedding_mat(batch_x_var)
        encoder_input_embedded = self.embedding_dropout_layer(encoder_input_embedded)
        # Encode
        # [batch_size, seq_len, embed_size], [1,batch_size,embed_size]
        encoder_outputs, encoder_hidden = self.encoder(encoder_input_embedded)
        # Decode
        dec_len, batch_size = batch_y_var.size()[0], batch_y_var.size()[1]
        # 当实现解码器时，直接使用编码器最后⼀个时间步的隐状态来初始化解码器的隐状态。
        dec_hidden = encoder_hidden
       
        dec_input = Variable(torch.LongTensor([BOS_ID] * batch_size)).to(self.device)

        logits = Variable(torch.zeros(dec_len, batch_size, self.tgt_vocab_size)).to(self.device)

        for di in range(dec_len):
            # 上一个输出的词嵌入
            prev_y = self.embedding_mat(dec_input)      # [seq_len?batch_size,embed_dim]
            dec_output, dec_hidden, attn_weights = self.decoder(prev_y, dec_hidden, encoder_outputs)
            logits[di] = dec_output  # 记录输出词的概率
            dec_input = batch_y_var[di]

        return logits
```

在预测方法中，根据输入矩阵（source_tensor）进行预测输出。首先，将输入矩阵进行词嵌入得到编码器的输入。然后，将编码器的输入传递给编码器进行编码，得到编码器的输出和最后一个时间步的隐藏状态。接下来，初始化解码器的输入为起始符号，并将起始符号的词嵌入（prev_y）传递给解码器进行解码，得到解码器的输出、更新后的解码器隐藏状态和注意力权重。重复这个过程，直到遇到终止符号或达到最大长度。在每个解码步骤中，记录解码结果和注意力权重，并将解码器的输出作为下一个解码步骤的输入。

更具体来说：

1. 首先，将输入矩阵（source_tensor）进行词嵌入操作，得到编码器的输入。
2. 将编码器的输入传递给编码器进行编码操作，得到编码器的输出和最后一个时间步的隐藏状态。
3. 初始化解码器的输入为起始符号的索引（BOS_ID），并将起始符号的词嵌入传递给解码器进行解码操作。
4. 循环解码：
   - 在每个解码步骤中，解码器接收上一个输出的词嵌入（prev_y）和解码器的隐藏状态（dec_hidden），生成当前时间步的输出。
   - 记录当前时间步的解码结果（decoded_ids）和注意力权重（attn_w）。
   - 根据当前时间步的输出概率分布，选择具有最高概率的词作为当前时间步的预测结果。
   - 将当前预测结果作为下一个解码步骤的输入，并更新解码步骤的索引（curr_dec_idx）。
     1. 继续循环解码，直到遇到终止符号（EOS_ID）或达到最大句子长度（config.max_sentence_length）为止。
     2. 返回解码结果（decoded_ids）和注意力权重（attn_w）作为预测方法的输出。

这样，Seq2Seq模型的前向传播方法可以将源语言数据作为输入，生成目标语言数据的概率分布。预测方法可以根据输入矩阵预测输出序列。



```python

    def predict(self, source_tensor):
        encoder_input_embedded = self.embedding_mat(source_tensor)
        encoder_outputs, encoder_hidden = self.encoder(encoder_input_embedded)

        decoded_ids, attn_w = [], []
        curr_token_id = BOS_ID
        curr_dec_idx = 0
        dec_input_var = Variable(torch.LongTensor([curr_token_id]))

        dec_input_var = dec_input_var.to(self.device)
        dec_hidden = encoder_hidden[:1] 
        # 直到 EOS 或达到最大长度
        while curr_token_id != EOS_ID and curr_dec_idx <= self.config.max_sentence_length:
            prev_y = self.embedding_mat(dec_input_var)  
            decoder_output, dec_hidden, decoder_attention = self.decoder(prev_y, dec_hidden, encoder_outputs)
            attn_w.append(decoder_attention.data.cpu().numpy().tolist()[0])
            topval, topidx = decoder_output.data.topk(1)  
            curr_token_id = topidx[0][0]
            decoded_ids.append(int(curr_token_id.cpu().numpy()))
            dec_input_var = (Variable(torch.LongTensor([curr_token_id]))).to(self.device)
            curr_dec_idx += 1
        return decoded_ids, attn_w
```





## 4. 评价指标BLEU

Papineni K, Roukos S, Ward T, et al. Bleu: a method for automatic evaluation of machine translation[C]//Proceedings of the 40th annual meeting of the Association for Computational Linguistics. 2002: 311-318.



BLEU（Bilingual Evaluation Understudy）是一种用于评估机器翻译或文本生成质量的自动化评估指标，评估生成文本与参考文本之间的相似性。BLEU-4是BLEU指标的一种变体，特别关注4-gram精确度。

BLEU-4中有如下的几个概念：

* **N-gram匹配**：机器生成的输出和参考翻译被分割成n-gram（连续的n个词）片段。例如，1-gram由单个单词组成，2-gram由相邻的词对组成，3-gram由三个词组成，依此类推。
* **N-gram计数**：计算机器生成的输出中每个n-gram的出现次数。

* **N-gram精确度计算**：BLEU-4计算机器生成的输出与参考（人工生成的）翻译之间1-gram、2-gram、3-gram和4-gram的匹配精确度。精确度衡量机器生成的输出中有多少个n-gram与参考翻译完全匹配。

* **简洁度惩罚**：BLEU-4引入了简洁度惩罚，以应对机器生成的输出明显短于参考翻译的情况。简洁度惩罚是一个因子，如果机器生成的输出较短，则会降低BLEU分数。它鼓励生成与参考翻译长度更接近的翻译结果。简洁惩罚度由BP表示，BP的定义为：

$$
BP=\begin{cases}1\quad if \quad c>r\\e^{1-r/c}\quad if\quad c\le r\end{cases}
$$

​	其中𝑐为候选文本的长度，𝑟 为与候选文本长度最近接的参考文本的长

* **综合精确度**：将各个n-gram精确度通过加权几何平均进行合并。通常为每个n-gram精确度设置权重为1/4。

* **最终BLEU分数**：将综合精确度乘以简洁度惩罚，得到最终的BLEU-4分数。简洁度惩罚有助于惩罚过短的翻译结果。

BLEU-4分数的范围在0到1之间，1表示机器生成的输出与参考翻译在4-gram精确度方面完全匹配。BLEU的计算公式如下：
$$
BLEU=BP\cdot exp(\sum_{n=1}^{N}w_nlogp_n)
$$
​	其中$w_n$为n-gram的权重，原文中描述的$w_n=1/N$，$p_n $为候选文本n-gram的得分。



此部分实现计算BLEU分数的类`BLEUScore`。

首先设置初始化函数`__init__`。初始化BLEUScore对象时，可以指定最大的n-gram大小（默认为4）和是否区分大小写（默认为False）。

```python
def __init__(self, max_ngram=4, case_sensitive=False):
        self.max_ngram = max_ngram
        self.case_sensitive = case_sensitive
        self.hits = [0] * self.max_ngram
        self.cand_lens = [0] * self.max_ngram
        self.ref_len = 0
        self.reset()
```



定义`reset`方法用于重置计数器，将命中次数、预测长度和参考长度都重置为0。

```python
def reset(self):
        self.hits = [0] * self.max_ngram
        self.cand_lens = [0] * self.max_ngram
        self.ref_len = 0
```



定义分词函数`tokenize`，该方法用于对输入的句子进行分词处理，主要是去除一些标点符号并将词按空格分开。

```python
def tokenize(self, sentence):
        """对输入的句子进行分词，主要是去除一些标点符号并将词按空格分开"""
        sentence = re.sub(r'[^\w\s]', '', sentence)
        return sentence.split()
```



定义`append`方法，用于将预测句子和参考句子添加到计算中。首先对预测句子和参考句子进行分词处理，然后根据n-gram的大小计算命中次数和预测长度。选择与预测句子长度最接近的参考句子，并记录参考句子的长度。

```python
def append(self, predicted_sentence, ref_sentences):
        predicted_sentence = predicted_sentence if 
        				isinstance(predicted_sentence, list) else 
                        self.tokenize(predicted_sentence)
        ref_sentences = [ref_sent if isinstance(ref_sent, list) else
                         self.tokenize(ref_sent) for ref_sent in ref_sentences]
        for i in range(self.max_ngram):
            # 计算每个 gram 的命中次数
            self.hits[i] += self.compute_hits(i + 1, 
                                              predicted_sentence, ref_sentences)
            # 计算每个 gram 的预测长度
            self.cand_lens[i] += len(predicted_sentence) - i
        # 选择长度最相近的参考文本
        closest_ref = min(ref_sentences, 
                          key=lambda ref_sent: 
                          (abs(len(ref_sent) - len(predicted_sentence)),
                           len(ref_sent)))
        # 记录参考文本长度
        self.ref_len += len(closest_ref)
```



定义`compute_hits`方法，用于计算给定n-gram大小的命中次数。首先将参考句子进行n-gram的统计，然后对预测句子进行n-gram的统计，计算预测句子中命中的n-gram个数。

```python
 def compute_hits(self, n, predicted_sentence, ref_sentences):
        merged_ref_ngrams = self.get_ngram_counts(n, ref_sentences)
        pred_ngrams = self.get_ngram_counts(n, [predicted_sentence])
        hits = 0
        for ngram, cnt in pred_ngrams.items():
            hits += min(merged_ref_ngrams.get(ngram, 0), cnt)
        return hits
```



定义`get_ngram_counts`方法，用于获取给定n-gram大小的统计信息。首先将句子按照n-gram大小聚合，然后统计每个n-gram的出现次数，并取最大值。

```python
def get_ngram_counts(self, n, sentences):
        merged_ngrams = {}
        # 按 gram 数聚合句子
        for sent in sentences:
            ngrams = defaultdict(int)
            if not self.case_sensitive:
                ngrams_list = list(zip(*[[tok.lower() for tok in sent[i:]] for i in range(n)]))
            else:
                ngrams_list = list(zip(*[sent[i:] for i in range(n)]))
            for ngram in ngrams_list:
                ngrams[ngram] += 1
            for ngram, cnt in ngrams.items():
                merged_ngrams[ngram] = max((merged_ngrams.get(ngram, 0), cnt))
        return merged_ngrams
```



最后定义`score`方法，该方法用于计算最终的BLEU分数。首先计算短句惩罚因子（bp），根据预测长度和参考长度的比例来确定惩罚因子的大小。然后计算每个n-gram的精确度，并累加其对数值。最终得到BLEU分数。

```python 
    def score(self):
        bp = 1.0
        # c <= r : BP=e^(1-r/c)
        # c > r : BP=1.0
        if self.cand_lens[0] <= self.ref_len:
            bp = math.exp(1.0 - self.ref_len / (float(self.cand_lens[0])
                                                if self.cand_lens[0] else 1e-5))
        prec_log_sum = 0.0
        for n_hits, n_len in zip(self.hits, self.cand_lens):
            n_hits = max(float(n_hits), self.TINY)

            n_len = max(float(n_len), self.SMALL)
            # 计算∑logPn=∑log(n_hits/n_len)
            prec_log_sum += math.log(n_hits / n_len)
        return bp * math.exp((1.0 / self.max_ngram) * prec_log_sum)

```



## 5. 训练与验证函数

### 5.1 训练函数

最后定义训练函数与测试函数。训练函数的定义如下，入参为Dataloader加载的对象、当前的epoch数和总共的epoch数。将数据加载到进度条库tqdm方法上。此处需要注意的是，加载的batch_data中的source和target是`[batch_size,seq_len]`形状的，需对其进行转置后传入模型进行计算，之后进行梯度下降等训练过程。

```python
def train(data_loader, epoch_current, epoch_total):
    """模型训练函数"""
    model.train()
    total_loss = 0.0  # 打印输出的loss
    t1 = time.time()
    with tqdm(total=len(data_loader),
              desc='Training epoch[{}/{}]'.format(epoch_current, epoch_total),
              file=sys.stdout) as t:
        for index, batch_data in enumerate(data_loader):
            source, target = batch_data
            source = source.to(device).transpose(0, 1)
			target = target.to(device).transpose(0, 1)
            optimizer.zero_grad()  # 梯度值初始化
            
            model_outputs = model((source, target))
            model_outputs = model_outputs.contiguous().view(-1, vocab_size)
            targets = target.contiguous().reshape(-1, 1).squeeze(1)
            
            loss = loss_function(model_outputs, targets.long())
            total_loss += loss.data.item()

            # 梯度下降
            loss.backward()
            optimizer.step()
            t.set_postfix(loss=total_loss / (index + 1), lr=scheduler.get_last_lr()[0], timecost=time.time()-t1)
            t.update(1)
        loss_list.append(total_loss / len(data_loader))
        lr_list.append(scheduler.get_last_lr()[0])
        scheduler.step()
```



### 5.2 验证函数

验证函数需要借助`Seq2Seq`模型中实现的predict方法，逐个对待验证的数据集中的结构化文本进行生成。之后计算在验证集上的BLEU-4值，若验证效果有所提升，就将当前模型保存至本地。

```python
def validation(data_iterator, epoch_now):
    global best_bleu
    model.eval()
    sentences = []
    with torch.no_grad():
        for data in tqdm(data_iterator, desc="[Validation]{}".format(" "*(5+len(str(epoch_now)))), file=sys.stdout):
            src, tgt, lex, multi_target = data
            src = torch.as_tensor(src[:, np.newaxis]).to(device)
            sentence, attention = model.predict(src)
            # 解码句子
            sentence = train_dataset.tokenizer.decode(sentence).replace('[NAME]', lex[0]).replace('[NEAR]', lex[1])
            sentences.append(sentence)
            scorer.append(sentence, multi_target)
        bleu = scorer.score()
        bleu_list.append(bleu)
        print("BLEU SCORE: {:.4f}".format(bleu))
        if bleu > best_bleu:
            state = {
                'model': model.state_dict(),
                'optimizer': optimizer.state_dict(),
                'bleu': bleu,
                'epoch': epoch_now,
            }
            if not os.path.exists(config.checkpoint_path):
                os.mkdir(config.checkpoint_path)
            torch.save(state, config.checkpoint_path + 'checkpoint.pth')
            print("模型保存成功！！")
            best_bleu = bleu
```





## 6. 主程序与结果

### 6.1 主程序

主程序的定义如下，其中完成了以下工作：

1. 创建`BLEUScore`实例作为评估指标。
2. 定义训练集、验证集和测试集的路径，并创建相应的`E2EDataset`实例。
3. 创建`DataLoader`实例，用于按批次加载数据。
4. 初始化模型，并根据需要加载预训练的模型。
5. 设置损失函数和优化器。这里使用了交叉熵损失函数和随机梯度下降（SGD）优化器。
6. 打印模型设置和代码运行环境的相关信息。
7. 进行训练和验证循环。首先进行训练，然后根据设定的验证频率进行验证。训练和验证的具体实现可能在后续的代码中定义。
8. 在训练过程中，记录每个epoch的损失值、BLEU分数和学习率等信息。
9. 在训练完成后，绘制验证BLEU分数、训练损失和学习率的曲线图。
10. 使用训练好的模型进行测试集上的预测。

```python
if __name__ == "__main__":
    scorer = BLEUScore(max_ngram=4)
    trainSet_path = config.root_path + config.train_data_path
    devSet_path = config.root_path + config.dev_data_path
    testSet_path = config.root_path + config.test_data_path

    train_dataset = E2EDataset(trainSet_path, train_mod='train')

    dev_dataset = E2EDataset(devSet_path, train_mod='valid',
                             attributes_vocab=train_dataset.attributes_vocab,
                             tokenizer=train_dataset.tokenizer)

    test_dataset = E2EDataset(testSet_path, train_mod='test',
                              attributes_vocab=train_dataset.attributes_vocab,
                              tokenizer=train_dataset.tokenizer)

    train_loader = DataLoader(dataset=train_dataset, batch_size=config.batch_size)
    
    # 初始化模型
    vocab_size = train_dataset.tokenizer.vocab_size
   
    model = Seq2Seq(config=config,
                    device=device,
                    src_vocab_size=vocab_size,
                    tgt_vocab_size=vocab_size).to(device)
    best_bleu = 0.0
    loss_list = []
    bleu_list = []
    lr_list = []

    # 加载ckpt
    if not os.path.exists(config.checkpoint_path):
        print("Warning: checkpoint directory not found!")
        start_epoch = 0
        best_bleu = 0.0
    else:
        # 加载模型
        print("===> Resume from checkpoint...")
        checkpoint = torch.load(config.checkpoint_path + 'checkpoint.pth')
        model.load_state_dict(checkpoint['model'])
        best_bleu = checkpoint['bleu']
        start_epoch = checkpoint['epoch']

    # 设置损失函数和优化器
    weight = torch.ones(train_dataset.tokenizer.vocab_size)
    weight[PAD_ID] = 0
    loss_function = nn.NLLLoss(weight, reduction='mean').to(device)
    optimizer = optim.SGD(model.parameters(), lr=config.lr)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=200)

    msg = """
                +-------------------------------------------------------+
                |{:^47}|
                |=======================================================|
                |Optimizer: {:<10}  lr: {:<10}  Device: {}    |
                |-------------------------------------------------------|
                |Loss Function:{:<20}  Max_Epoch:{:<3}      |
                |=======================================================|
                |vocab size: {:<15}  batch size: {:<14}|
                |=======================================================|
                |训练集长度: {:<17}  测试集长度: {:<16}|
                |=======================================================|
                |{:<55}|
                +-------------------------------------------------------+
                """.format("模型设置以及代码运行环境", 'SGD', config.lr,
                           device,
                           str(loss_function), 
                           config.epoch, vocab_size, 
                           config.batch_size, len(train_dataset),
                           len(test_dataset), str(datetime.now()))
    print(msg)

    # 训练和验证
    print("Start Epoch ====>\t", start_epoch)
    for i in range(start_epoch, config.epoch):
        train(model, train_loader, i + 1, config.epoch)
        if (i + 1) % config.num_val == 0:
            validation(model, dev_dataset, i)
    plot_carve(title="valid_bleu", save_path="../res_img/valid_bleu.png",
               x=len(bleu_list), y=bleu_list)
    plot_carve(title="train_loss", save_path="../res_img/train_loss.png", x=len(loss_list), y=loss_list)
    plot_carve(title="train_lr", save_path="../res_img/train_lr.png", x=len(lr_list), y=lr_list)

    predict(model, test_dataset)
```



### 6.2 部分结果

经过50次训练，训练过程中的部分打印输出如图，在第4轮迭代时验证集上的BLEU-4值达到了0.6072，第21轮迭代时达到了0.7099



![image-20230605143922789](https://typora-1308640872.cos-website.ap-beijing.myqcloud.com/img/image-20230605143922789.png)

![train_loss](https://typora-1308640872.cos-website.ap-beijing.myqcloud.com/img/train_loss.png)

![train_lr](https://typora-1308640872.cos-website.ap-beijing.myqcloud.com/img/train_lr.png)

![valid_bleu](https://typora-1308640872.cos-website.ap-beijing.myqcloud.com/img/valid_bleu.png)





## 7. Attention可视化

Attention可视化的理论基础是为了解释和理解深度学习模型在处理任务时的注意力分配机制。深度学习模型通常在处理自然语言处理和计算机视觉等任务时具有很高的性能，但其内部工作机制往往是黑盒子，难以解释。通过可视化Attention，我们可以更好地理解模型如何在输入中选择和聚焦于相关的部分，以便生成或预测输出。

Attention可视化通过将模型的注意力权重与输入对齐，以图形化方式显示模型对输入的关注程度。这样一来，我们可以直观地看到模型在输入中着重关注的区域和特征。通过可视化Attention，我们可以识别出模型在处理不同任务时的注意力分布模式，进而推断模型学习到的特征和决策依据。



```python
def visualize_attention(dataset, data_index=0):
    """Attention可视化"""
    src, tgt, lex, _ = dataset[data_index]
    src = torch.as_tensor(src[np.newaxis, :]).to(device).transpose(0, 1)
    sentence, attention = model.predict(src)
    src_txt = list(map(lambda x: dataset.tokenizer.index_to_token(x),
                       src.flatten().cpu().numpy().tolist()[:10]))
    for i in range(len(src_txt)):
        if src_txt[i] == '[NAME]':
            src_txt[i] = lex[0]
        elif src_txt[i] == '[NEAR]':
            src_txt[i] = lex[1]
    sentence_txt = list(map(lambda x: dataset.tokenizer.index_to_token(x),
                            sentence))
    for i in range(len(src_txt)):
        if sentence_txt[i] == '[NAME]':
            sentence_txt[i] = lex[0]
        elif sentence_txt[i] == '[NEAR]':
            sentence_txt[i] = lex[1]

    # 绘制热力图
    ax = sns.heatmap(np.array(attention)[:, :10] * 100, cmap='YlGnBu')
    # 设置坐标轴
    plt.yticks([i + 0.5 for i in range(len(sentence_txt))], labels=sentence_txt, rotation=360, fontsize=12)
    plt.xticks([i + 0.5 for i in range(len(src_txt))], labels=src_txt, fontsize=12)
    plt.show()
```



如图是根据Attention权重绘制的热力图，其中横坐标为结构化文本中的属性值，分别对应name、餐food、priceRange 、customer rating、 familyFriendly、 area、near、 eatType，0 则是未给定属性值；纵坐标表示编码的结果句子结果。颜色越深的块表示对应的横坐标对于推理出纵坐标的值提供的帮助越大，例如图中 no 对于推理出located、near两个单词的帮助很大。

![image-20230605161008503](https://typora-1308640872.cos-website.ap-beijing.myqcloud.com/img/image-20230605161008503.png)
