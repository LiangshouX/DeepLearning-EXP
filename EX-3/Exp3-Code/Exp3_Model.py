import torch
import torch.nn as nn
import torch.nn.functional as functions


class TextCNN_Model(nn.Module):

    def __init__(self, configs):
        super(TextCNN_Model, self).__init__()

        vocab_size = configs.vocab_size
        embedding_dimension = configs.embedding_dimension
        label_num = configs.label_num

        num_channels = [100, 100, 100]
        kernel_sizes = [3, 4, 5]

        # 词嵌入和dropout
        self.const_embed = nn.Embedding(vocab_size, embedding_dimension)
        self.embed = nn.Embedding(vocab_size, embedding_dimension)
        self.dropout = nn.Dropout(configs.dropout)
        self.decoder = nn.Linear(sum(num_channels), configs.label_num)

        # 最大时间汇聚层没有参数，因此可以共享此实例
        self.pool = nn.AdaptiveAvgPool1d(1)
        self.relu = nn.ReLU()

        # 创建多个一维卷积
        self.convs = nn.ModuleList()
        for c, k in zip(num_channels, kernel_sizes):
            self.convs.append(nn.Conv1d(2 * embedding_dimension, c, k))

    def forward(self, sentences):
        print(sentences)
        x = torch.cat(
            (self.embed(sentences), self.const_embed(sentences)),
            dim=2
        )
        # 根据一维卷积的输入格式，重新排列张量，以便通道作为第二维
        x = x.permute(0, 2, 1)

        encoding = torch.cat(
            [
                torch.squeeze(self.relu(self.pool(conv(x))), dim=-1)
                for conv in self.convs
            ],
            dim=1
        )

        outputs = self.decoder(self.dropout(encoding))
        return outputs
