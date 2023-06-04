"""
    构建Encoder-Decoder模型
"""
import torch
import torch.nn as nn
import torch.optim as optim
from torch.nn.utils import clip_grad_norm_
from Config import Config

config = Config()


class Encoder(nn.Module):
    def __init__(self, input_size, hidden_size):
        super(Encoder, self).__init__()

        self.hidden_size = hidden_size

        self.embedding = nn.Embedding(input_size, hidden_size)
        self.rnn = nn.GRU(hidden_size, hidden_size, batch_first=True)

    def forward(self, input_seq):
        # print(type(input_seq))
        embedded = self.embedding(input_seq)
        output, hidden = self.rnn(embedded)
        return output, hidden


class Attention(nn.Module):
    def __init__(self, hidden_size):
        super(Attention, self).__init__()

        self.hidden_size = hidden_size

        self.attn = nn.Linear(hidden_size * 2, hidden_size)
        self.v = nn.Linear(hidden_size, 1, bias=False)

    def forward(self, decoder_hidden, encoder_output):
        seq_len = encoder_output.size(1)

        # Repeat decoder hidden state for each time step
        repeated_decoder_hidden = decoder_hidden.unsqueeze(1).repeat(1, seq_len, 1)

        # Concatenate encoder output and repeated decoder hidden state
        concatenated = torch.cat((encoder_output, repeated_decoder_hidden), dim=2)

        # Calculate attention energies
        energies = torch.tanh(self.attn(concatenated))

        # Calculate attention weights
        attention_weights = torch.softmax(self.v(energies).squeeze(2), dim=1)

        # Calculate the context vector
        context_vector = torch.bmm(attention_weights.unsqueeze(1), encoder_output).squeeze(1)

        return context_vector, attention_weights


class Decoder(nn.Module):
    def __init__(self, output_size, hidden_size):
        super(Decoder, self).__init__()

        self.hidden_size = hidden_size

        self.embedding = nn.Embedding(output_size, hidden_size)
        self.rnn = nn.GRU(hidden_size + hidden_size, hidden_size, batch_first=True)
        self.attention = Attention(hidden_size)
        self.out = nn.Linear(hidden_size, output_size)

    def forward(self, input_seq, hidden, encoder_output):
        embedded = self.embedding(input_seq)

        # Calculate attention context vector
        context_vector, attention_weights = self.attention(hidden.squeeze(0), encoder_output)

        # Concatenate embedded input and context vector
        # print("Decoder embedded:\t{}\n"
        #       "Decoder context_vector:\t{}\n"
        #       "Decoder context_vector.unsqueeze(1):\t{}".format(embedded.shape,
        #                                                         context_vector.shape,
        #                                                         context_vector.unsqueeze(1).shape))
        # [3,256] [3,256] [3,1,256]

        rnn_input = torch.cat((embedded.unsqueeze(1), context_vector.unsqueeze(1)), dim=2)

        # Pass through the GRU
        output, hidden = self.rnn(rnn_input, hidden)

        # Predict the output
        output = self.out(output.squeeze(1))

        return output, hidden, attention_weights


class Seq2Seq(nn.Module):
    def __init__(self, input_size, output_size, hidden_size):
        super(Seq2Seq, self).__init__()

        self.encoder = Encoder(input_size, hidden_size)
        self.decoder = Decoder(output_size, hidden_size)

    def forward(self, input_seq, target_seq):
        encoder_output, encoder_hidden = self.encoder(input_seq)
        decoder_input = target_seq  # Remove the last token from the target sequence
        decoder_hidden = encoder_hidden

        output_seq = []
        attention_weights_seq = []

        for i in range(decoder_input.size(1)):
            decoder_output, decoder_hidden, attention_weights = self.decoder(decoder_input[:, i], decoder_hidden,
                                                                             encoder_output)
            output_seq.append(decoder_output.unsqueeze(1))
            attention_weights_seq.append(attention_weights.unsqueeze(1))

        # Concatenate the predicted outputs
        output_seq = torch.cat(output_seq, dim=1)
        attention_weights_seq = torch.cat(attention_weights_seq, dim=1)
        return output_seq, attention_weights_seq


if __name__ == "__main__":
    in_size = 100  # Input vocabulary size
    out_size = 120  # Output vocabulary size
    hi_size = 256  # Hidden state size
    device_test = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    encoder_test = Encoder(in_size, hi_size).to(device_test)
    decoder_test = Decoder(hi_size, out_size).to(device_test)
    model = Seq2Seq(in_size, out_size, hi_size).to(device_test)

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    # 相当于 batch_size=3，seq_len=5 的数据
    src = torch.tensor([[1, 2, 3, 4, 5],
                        [1, 2, 3, 4, 5],
                        [1, 2, 3, 4, 5]]).to(device_test)
    src_ = torch.tensor([[1, 2, 3, 4]]).to(device_test)

    tgt = torch.tensor([[1, 1, 1, 1, 1],
                        [1, 1, 1, 1, 1],
                        [1, 1, 1, 1, 1]]).to(device_test)
    tgt_ = torch.tensor([[1, 1, 1, 1]]).to(device_test)

    print("src shape:\t{}\ntgt shape:\t{}\n".format(src.shape, tgt.shape))

    model_outputs, att_w = model(src, tgt)
    print(model_outputs.size())         # [batch_size, seq_len - 1, out_size]
    print(att_w.size())
