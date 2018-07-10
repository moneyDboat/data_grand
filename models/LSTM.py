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


class LSTM(nn.Module):
    def __init__(self, args):
        self.args = args
        super(LSTM, self).__init__()
        self.hidden_dim = args.hidden_dim
        self.batch_size = args.batch_size
        self.use_gpu = torch.cuda.is_available()

        self.word_embeddings = nn.Embedding(args.vocab_size, args.embedding_dim)
        self.word_embeddings.weight = nn.Parameter(args.embeddings, requires_grad=False)
        #        self.word_embeddings.weight.data.copy_(torch.from_numpy(opt.embeddings))

        # self.bidirectional = True
        self.dropout = self.keep_dropout
        self.bilstm = nn.LSTM(self.embedding_dim, self.hidden_dim // 2, num_layers=1, dropout=self.dropout,
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

        #    @profile

    def forward(self, sentence):
        embeds = self.word_embeddings(sentence)

        #        x = embeds.view(sentence.size()[1], self.batch_size, -1)
        x = embeds.permute(1, 0, 2)  # we do this because the default parameter of lstm is False
        self.hidden = self.init_hidden(sentence.size()[0])  # 2x64x64
        lstm_out, self.hidden = self.bilstm(x, self.hidden)  # lstm_out:200x64x128
        if self.mean == "mean":
            out = lstm_out.permute(1, 0, 2)
            final = torch.mean(out, 1)
        else:
            final = lstm_out[-1]
        y = self.hidden2label(final)  # 64x3  #lstm_out[-1]
        return y


