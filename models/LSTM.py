# -*- coding: utf-8 -*-

"""
@Author  : captain
@time    : 18-7-10 下午8:44
@ide     : PyCharm  
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from .BasicModule import BasicModule


class LSTM(BasicModule):
    def __init__(self, args, vectors):
        self.args = args
        super(LSTM, self).__init__()
        self.hidden_dim = args.hidden_dim
        self.batch_size = args.batch_size
        self.use_gpu = args.cuda
        self.lstm_layers = args.lstm_layers

        self.embedding = nn.Embedding(args.vocab_size, args.embedding_dim)
        self.embedding.weight.data.copy_(vectors)

        self.bidirectional = True
        if self.lstm_layers > 1:
            self.dropout = args.lstm_dropout
            self.bilstm = nn.LSTM(args.embedding_dim, self.hidden_dim // 2, num_layers=self.lstm_layers,
                                  dropout=self.dropout, bidirectional=True)
        else:
            self.bilstm = nn.LSTM(args.embedding_dim, self.hidden_dim // 2, num_layers=self.lstm_layers,
                                  bidirectional=True)
        self.hidden2label = nn.Linear(self.hidden_dim, args.label_size)
        # self.hidden = self.init_hidden(args.batch_size)
        # self.mean = args.__dict__.get("lstm_mean", True)

    def init_hidden(self, batch_size=None):
        if batch_size is None:
            batch_size = self.batch_size

        '''
        if self.use_gpu:
            h0 = nn.Parameter(nn.init.xavier_uniform_(torch.Tensor(2 * self.lstm_layers, batch_size, self.hidden_dim // 2).type(torch.FloatTensor).cuda()), requires_grad=True)
            c0 = nn.Parameter(nn.init.xavier_uniform_(torch.Tensor(2 * self.lstm_layers, batch_size, self.hidden_dim // 2).type(torch.FloatTensor).cuda()), requires_grad=True)
        else:
        '''
        h0 = nn.Parameter(nn.init.xavier_uniform_(
            torch.Tensor(2 * self.lstm_layers, batch_size, self.hidden_dim // 2).type(torch.FloatTensor)),
            requires_grad=True)
        c0 = nn.Parameter(nn.init.xavier_uniform_(
            torch.Tensor(2 * self.lstm_layers, batch_size, self.hidden_dim // 2).type(torch.FloatTensor)),
            requires_grad=True)
        return (h0, c0)

    def forward(self, sentence):
        embed = self.embedding(sentence)

        x = embed
        # x = embeds.permute(1, 0, 2)  # we do this because the default parameter of lstm is False
        lstm_out, _ = self.bilstm(x)  # lstm_out: 1000x128x100
        '''
        if self.mean == "mean":
            out = lstm_out.permute(1, 0, 2)
            final = torch.mean(out, 1)
        else:
        '''
        final = lstm_out[-1]
        y = self.hidden2label(final)  # 64x3  #lstm_out[-1]
        return y
