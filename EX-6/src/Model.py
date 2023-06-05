"""
    构建Encoder-Decoder模型
"""
import torch
import torch.nn as nn
import numpy as np
import torch.optim as optim
from torch.nn.utils import clip_grad_norm_
from torch.autograd import Variable
from Config import Config

PAD_ID = 0
BOS_ID = 1
EOS_ID = 2

class Encoder(nn.Module):
    """一个简单地编码器 将⻓度可变的输⼊序列编码成⼀个“状态”，以便后续对该状态进⾏解码。
        从技术上讲，编码器将⻓度可变的输⼊序列转换成形状固定的上下⽂变量c，并且将输⼊序列的信息在该上下⽂变量中进⾏编码
        Args:
            input_size(int): 输入特征的维度
            hidden_size(int): 编码器隐藏层的维度，输入特征维度对应的输出
        Note:
            input_size 需要和 embed_dim 相等
    """
    def __init__(self, input_size, hidden_size):
        super(Encoder, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        # nn.Linear 的输入是一个张量，形状通常为 (batch_size, input_size), input_size 表示输入特征的维度
        # nn.Linear 的输出是经过线性变换后的张量，形状为 (batch_size, out_features)，其中 out_features 表示输出特征的维度。
        self.W = nn.Linear(in_features=self.input_size, out_features=self.hidden_size)
        # self.gru = nn.GRU(input_size=self.input_size,
        #                   hidden_size=self.hidden_size,
        #                   batch_first=False)
        self.relu = nn.ReLU()

    def forward(self, input_embedded):
        """前向传播函数，数据进入模型经过 嵌入后 首先调用encoder
        Args:
            input_embedded(Tensor): [seq_len, batch_size, embed_dim]
        Returns:
            outputs(Tensor): [seq_len, batch_size, embed_dim]
            decoder_hidden(Tensor): [1,batch_size,embed_size](经过unsqueeze(0)操作之后)
        """
        seq_len, batch_size, embed_dim = input_embedded.size()
        # 将词嵌入的输入 reshape 为 [seq_len*batch_size, embed_dim]
        outputs = self.relu(self.W(input_embedded.view(-1, embed_dim)))
        outputs = outputs.view(seq_len, batch_size, -1)
        # print(outputs.shape)
        decoder_hidden = torch.sum(outputs, 0)
        # outputs, decoder_hidden = self.gru(input_embedded)
        return outputs, decoder_hidden.unsqueeze(0)
        # return outputs, decoder_hidden


class Attention(nn.Module):
    """Attention模块
    Args:
        encoder_hidden_dim(int): 编码器隐藏层输出维度
        decoder_hidden_dim(int): 对应解码器隐藏状态的维度，即 GRU 单元的输出大小。
        attn_dim(optional): Attention权重的维度
    """
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
        self.softmax = nn.Softmax(dim=1)

    def forward(self, prev_h_batch, enc_outputs):
        """Attention前向传播
        Args:
            prev_h_batch(Tensor):
            enc_outputs(Tensor):    [seq_len, batch_size, encoder_hidden_dim]
        """
        src_seq_len, batch_size, enc_dim = enc_outputs.size()
        uh = self.U(enc_outputs.view(-1, self.h_dim)).view(src_seq_len,
                                                           batch_size, self.a_dim)  # SL x B x self.attn_dim
        wq = self.W(prev_h_batch.view(-1, self.s_dim)).unsqueeze(0)  #
        wq3d = wq.expand_as(uh)
        wquh = self.tanh(wq3d + uh)
        attn_unnorm_scores = self.v(wquh.view(-1, self.a_dim)).view(batch_size, src_seq_len)
        attn_weights = self.softmax(attn_unnorm_scores)  # B x SL
        return attn_weights


class Decoder(nn.Module):
    """解码器结构
    Args:
        input_size(int): 解码器输入输入特征的维度，即输入张量的最后一个维度大小。
        hidden_size(int): 解码器隐藏状态的维度，即 GRU 单元的输出大小。
        output_size(int): 解码器输出维度，为 target(即ref文本) 的词表大小
        embedding_dim(int): 词嵌入维度
        encoder_hidden_size(int): 编码器隐藏层输出维度
    """
    def __init__(self, input_size, hidden_size, output_size, embedding_dim, encoder_hidden_size):
        super(Decoder, self).__init__()
        self.rnn = nn.GRU(input_size, hidden_size, bidirectional=False)
        self.attn = Attention(encoder_hidden_size, hidden_size)
        self.W_combine = nn.Linear(embedding_dim + encoder_hidden_size, hidden_size)
        self.W_out = nn.Linear(hidden_size, output_size)
        self.log_softmax = nn.LogSoftmax(dim=1)

    def forward(self, prev_y_batch, prev_h_batch, encoder_outputs_batch):
        """前向传播
        Args:
            prev_y_batch
            prev_h_batch
            encoder_outputs_batch
        Returns:
            pass
        """
        attn_weights = self.attn(prev_h_batch, encoder_outputs_batch)
        context = torch.bmm(attn_weights.unsqueeze(1), encoder_outputs_batch.transpose(0, 1))
        # print("prev_y_batch.shape:\t{}".format(prev_y_batch.shape))  # [batch_size, 100]
        # print("attn_weights.shape:\t{}".format(attn_weights.shape))  # [batch_size, seq_len]
        # print("context.shape:\t{}".format(context.shape))       # [batch_size,1,100]
        # print("context.squeeze(1).shape:\t{}".format(context.squeeze(1).shape))  # [batch_size,100]
        y_ctx = torch.cat((prev_y_batch, context.squeeze(1)), 1)        # [batch_size, 100+100]
        # print("y_ctx.shape:\t{}".format(y_ctx.shape))
        rnn_input = self.W_combine(y_ctx)
        # GRU 输入的形状应为 `(sequence_length, batch_size, input_size)`
        # 第一个输出表示每个时间步的隐藏状态输出，形状为 `(sequence_length, batch_size, hidden_size * num_directions)`
        # 第二个输出表示最后一个时间步的隐藏状态，形状为 `(num_layers * num_directions, batch_size, hidden_size)`
        dec_rnn_output, dec_hidden = self.rnn(rnn_input.unsqueeze(0), prev_h_batch)
        unnormalized_logits = self.W_out(dec_rnn_output[0])
        dec_output = self.log_softmax(unnormalized_logits)
        # print("decoder output in Decoder:{}".format(dec_output))
        return dec_output, dec_hidden, attn_weights


class Seq2Seq(nn.Module):
    """端到端模型
    Args:
        config(Config):
        device(torch.device):
        src_vocab_size(int):
        tgt_vocab_size(int):
    """

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
        # print("encoder_input_embedded.shape:\t{}".format(encoder_input_embedded.shape))

        # Encode
        # [batch_size, seq_len, embed_size], [1,batch_size,embed_size]
        encoder_outputs, encoder_hidden = self.encoder(encoder_input_embedded)
        # print("encoder_outputs.shape:\t{}\n"
        #       "encoder_hidden.shape:\t{}".format(encoder_outputs.shape,
        #                                          encoder_hidden.shape))

        # Decode
        dec_len, batch_size = batch_y_var.size()[0], batch_y_var.size()[1]
        # 当实现解码器时，直接使用编码器最后⼀个时间步的隐状态来初始化解码器的隐状态。
        dec_hidden = encoder_hidden
        #   [batch_size]    ?
        dec_input = Variable(torch.LongTensor([BOS_ID] * batch_size)).to(self.device)

        # [seq_len, batch_size, vocab_size]
        logits = Variable(torch.zeros(dec_len, batch_size, self.tgt_vocab_size)).to(self.device)

        for di in range(dec_len):
            # 上一个输出的词嵌入
            prev_y = self.embedding_mat(dec_input)      # [seq_len?batch_size,embed_dim]
            # print("prev_y.shape:\t{}".format(prev_y.shape))
            # print("dec_input_later.shape:\t{}".format(batch_y_var[di].shape))
            dec_output, dec_hidden, attn_weights = self.decoder(prev_y, dec_hidden, encoder_outputs)
            logits[di] = dec_output  # 记录输出词的概率
            dec_input = batch_y_var[di]

        return logits

    def predict(self, source_tensor):
        """根据输入矩阵预测输出
        Args:
            source_tensor(Tensor): [seq_len, batch_size]
        Returns:
            decoded_ids(list):
            attn_w(list):
        """
        # print(source_tensor.size())     # [seq_len, 1]
        encoder_input_embedded = self.embedding_mat(source_tensor)
        encoder_outputs, encoder_hidden = self.encoder(encoder_input_embedded)

        # 解码
        decoded_ids, attn_w = [], []
        curr_token_id = BOS_ID
        curr_dec_idx = 0
        dec_input_var = Variable(torch.LongTensor([curr_token_id]))

        dec_input_var = dec_input_var.to(self.device)
        dec_hidden = encoder_hidden[:1]  # 1 x B x enc_dim
        # 直到 EOS 或达到最大长度
        while curr_token_id != EOS_ID and curr_dec_idx <= self.config.max_sentence_length:
            prev_y = self.embedding_mat(dec_input_var)  # 上一输出的词嵌入，B x E
            # 解码
            decoder_output, dec_hidden, decoder_attention = self.decoder(prev_y, dec_hidden, encoder_outputs)
            # 记录注意力
            attn_w.append(decoder_attention.data.cpu().numpy().tolist()[0])
            # print("decoder_output:{}".format(decoder_output))   # 为什么全是0？
            topval, topidx = decoder_output.data.topk(1)  # 选择最大概率
            curr_token_id = topidx[0][0]
            # 记录解码结果
            # print("topidx:{}".format(topidx))
            # print("curr_token_id:{}".format(curr_token_id))
            decoded_ids.append(int(curr_token_id.cpu().numpy()))
            # 下一输入
            dec_input_var = (Variable(torch.LongTensor([curr_token_id]))).to(self.device)
            curr_dec_idx += 1
        # print("decoded_ids: {}".format(decoded_ids))
        return decoded_ids, attn_w


if __name__ == "__main__":
    cfg = Config()

    # 注：input_size 与 embed_size 需相等
    test_input_size = 8
    test_hidden_size = 8
    test_output_size = 20

    test_seq_len = 5
    test_batch_size = 4
    test_vocab_size = 10
    test_embed_size = 8

    test_embedding = nn.Embedding(num_embeddings=test_vocab_size,
                                  embedding_dim=test_embed_size)

    test_tensor = torch.randn(test_batch_size, test_seq_len, test_embed_size)
    print("test_tensor shape:\t{}".format(test_tensor.shape))

    # test_tensor_embed = test_embedding(test_tensor)
    # print(test_tensor_embed, test_tensor_embed.shape)

    # 实例化Encoder  查看相关信息
    test_encoder = Encoder(input_size=test_input_size, hidden_size=test_hidden_size)
    test_encoder_outputs, test_encoder_dec_hidden = test_encoder(test_tensor)
    print("encoder_outputs.size:{}\t"
          "encoder_dec_hidden.size():{}".format(test_encoder_outputs.size(),
                                                test_encoder_dec_hidden.size()))
    # [seq_len,batch_size,embed_size], [1,batch_size,embed_size]
