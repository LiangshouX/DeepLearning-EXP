import torch
import torch.nn as nn
import torch.optim as optim
from torch.nn.utils import clip_grad_norm_
from Config import Config

config = Config()


class Attention(nn.Module):
    def __init__(self, encoder_hidden_size, decoder_hidden_size):
        super(Attention, self).__init__()

        self.encoder_hidden_size = encoder_hidden_size
        self.decoder_hidden_size = decoder_hidden_size

        self.attn = nn.Linear(encoder_hidden_size + decoder_hidden_size, 1)

    def forward(self, encoder_hidden_states, decoder_hidden_state):
        # 计算注意力分数
        seq_len = encoder_hidden_states.size(1)
        decoder_hidden_state = decoder_hidden_state.unsqueeze(1).repeat(1, seq_len, 1)
        energy = torch.tanh(self.attn(torch.cat((encoder_hidden_states, decoder_hidden_state), dim=2)))

        # 使用softmax计算注意力权重
        attention_weights = torch.softmax(energy, dim=1)

        # 计算上下文向量
        context_vector = torch.bmm(attention_weights.transpose(1, 2), encoder_hidden_states)

        return context_vector, attention_weights


from nltk.translate.bleu_score import sentence_bleu

reference = [['this', 'is', 'a', 'test'], ['this', 'is' 'test']]
candidate = ['this', 'is', 'a']
score = sentence_bleu(reference, candidate, weights=(1, 0, 0, 0))
print(score)
