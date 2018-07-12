# -*- coding: utf-8 -*-

"""
@Author  : captain
@time    : 18-7-10 下午8:44
@ide     : PyCharm  
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
from .BasicModule import BasicModule


class LSTM(BasicModule):
    def __init__(self, opt, vectors):
        super(LSTM, self).__init__()
        self.opt = opt
        emb_dim = opt.embedding_dim
        self.hidden_dim = opt.hidden_dim
        self.batch_size = opt.batch_size

        self.embed = nn.Embedding(opt.vocab_size, emb_dim)
        self.embed.weight.data.copy_(vectors)

        # 双向LSTM
        self.bilstm = nn.LSTM(emb_dim, self.hidden_dim // 2, num_layers=1, dropout=opt.lstm_dropout,
                              bidirectional=True)
        self.hidden2label = nn.Linear(self.hidden_dim, self.label_size)
        self.hidden = self.init_hidden()
        self.mean = opt.__dict__.get("lstm_mean", True)

    def init_hidden(self, batch_size=None):
        if batch_size is None:
            batch_size = self.batch_size

        if self.use_gpu:
            h0 = Variable(torch.zeros(2 * self.lstm_layers, batch_size, self.hidden_dim // 2).cuda())
            c0 = Variable(torch.zeros(2 * self.lstm_layers, batch_size, self.hidden_dim // 2).cuda())
        else:
            h0 = Variable(torch.zeros(2 * self.lstm_layers, batch_size, self.hidden_dim // 2))
            c0 = Variable(torch.zeros(2 * self.lstm_layers, batch_size, self.hidden_dim // 2))
        return (h0, c0)


    def forward(self, inputs):
        embeds = self.word_embeddings(inputs)

        #        x = embeds.view(sentence.size()[1], self.batch_size, -1)
        x = embeds.permute(1, 0, 2)  # we do this because the default parameter of lstm is False
        self.hidden = self.init_hidden(inputs.size()[0])  # 2x64x64
        lstm_out, self.hidden = self.bilstm(x, self.hidden)  # lstm_out:200x64x128
        if self.mean == "mean":
            out = lstm_out.permute(1, 0, 2)
            final = torch.mean(out, 1)
        else:
            final = lstm_out[-1]
        y = self.hidden2label(final)  # 64x3  #lstm_out[-1]
        return y
