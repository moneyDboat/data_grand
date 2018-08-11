# -*- coding: utf-8 -*-

"""
@Author  : captain
@time    : 18-8-10 下午2:58
@ide     : PyCharm  
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from .BasicModule import BasicModule
from config import DefaultConfig


def kmax_pooling(x, dim, k):
    index = x.topk(k, dim=dim)[1].sort(dim=dim)[0]
    return x.gather(dim, index)


class RCNN(BasicModule):
    def __init__(self, args, vectors):
        self.kmax_k = args.kmax_pooling
        super(RCNN, self).__init__()

        #
        self.embedding = nn.Embedding(args.vocab_size, args.embedding_dim)
        self.embedding.weight.data.copy_(vectors)
        self.lstm = nn.LSTM(
            input_size=args.embedding_dim,
            hidden_size=args.hidden_dim,
            num_layers=args.lstm_layers,
            batch_first=False,
            dropout=args.lstm_dropout,
            bidirectional=True
        )
        self.conv = nn.Sequential(
            nn.Conv1d(in_channels=args.hidden_dim * 2 + args.embedding_dim, out_channels=200, kernel_size=3),
            nn.BatchNorm1d(200),
            nn.ReLU(inplace=True),
            nn.Conv1d(in_channels=200, out_channels=200, kernel_size=3),
            nn.BatchNorm1d(200),
            nn.ReLU(inplace=True)
        )

        # classifer
        # self.fc = nn.Linear(2 * (100 + 100), args.label_size)
        self.fc = nn.Sequential(
            nn.Linear(2 * 200, args.linear_hidden_size),
            nn.BatchNorm1d(args.linear_hidden_size),
            nn.ReLU(inplace=True),
            nn.Linear(args.linear_hidden_size, args.label_size)
        )

    def forward(self, text):
        embed = self.embedding(text)
        out = self.lstm(embed)[0].permute(1, 2, 0)
        out = torch.cat((out, embed.permute(1, 2, 0)), dim=1)
        conv_out = kmax_pooling(self.conv(out), 2, self.kmax_k)

        flatten = conv_out.view(conv_out.size(0), -1)
        logits = self.fc(flatten)
        return logits





