# -*- coding: utf-8 -*-

import math
import torch
import torch.nn as nn
import torch.nn.functional as F


class Encoder(nn.Module):

    def __init__(self, enc_method, input_size, hidden_size, out_size):
        '''
        input_size
        hidden_size: the output size of CNN/RNN/TR
        outpu_size: the final size of the encoder (after pooling)
        w
        CNN:
        - filters_num: feature_dim
        - filter_size: 3
        - pooling: max_pooling
        RNN:
        - hidden_size: feature_dim // 2
        - pooling: last hidden status
        Transformer
        - nhead: 2
        - nlayer: 1
        - pooling: average
        -------
        '''
        super(Encoder, self).__init__()
        self.enc_method = enc_method.lower()
        if self.enc_method == 'cnn':
            self.conv = nn.Conv2d(1, hidden_size, (3, input_size))
            nn.init.xavier_uniform_(self.conv.weight)
            nn.init.constant_(self.conv.bias, 0.0)
            f_dim = hidden_size
        elif self.enc_method == 'rnn':
            self.rnn = nn.GRU(input_size, hidden_size//2, batch_first=True, bidirectional=True)
            f_dim = hidden_size
        elif self.enc_method == 'transformer':
            self.pe = PositionEmbedding(input_size, 512)
            self.layer = nn.TransformerEncoderLayer(d_model=input_size, nhead=1)
            self.tr = nn.TransformerEncoder(self.layer, num_layers=1)
            f_dim = input_size
        else:
            f_dim = input_size

        self.fc = nn.Linear(f_dim, out_size)
        nn.init.uniform_(self.fc.weight, -0.5, 0.5)
        nn.init.uniform_(self.fc.bias, -0.1, 0.1)

    def forward(self, inputs):
        if self.enc_method == 'cnn':
            x = inputs.unsqueeze(1)
            x = F.relu(self.conv(x).squeeze(3))
            out = x.permute(0, 2, 1)
        elif self.enc_method == 'rnn':
            out, _ = self.rnn(inputs)
        elif self.enc_method == 'transformer':
            inputs = self.pe(inputs)
            out = self.tr(inputs.permute(1, 0, 2)).permute(1, 0, 2)
        else:
            out = inputs
        return self.fc(out.mean(1))


class PositionEmbedding(nn.Module):
    def __init__(self, d_model, max_len):
        super(PositionEmbedding, self).__init__()
        self.pe = nn.Embedding(max_len, d_model)
        nn.init.uniform_(self.pe.weight, -0.1, 0.1)

    def forward(self, x):
        b, l, d = x.size()
        seq_len = torch.arange(l).to(x.device)
        return x + self.pe(seq_len).unsqueeze(0)


# performance poor
class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len):
        super(PositionalEncoding, self).__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)
        self.register_buffer('pe', pe)

    def forward(self, x):
        x = x + self.pe[:, :x.size(1)]
        return x
