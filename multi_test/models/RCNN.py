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
    def __init__(self, args, word_vectors, art_vectors):
        self.kmax_k = args.kmax_pooling
        super(RCNN, self).__init__()

        # word
        self.word_embedding = nn.Embedding(args.word_vocab_size, args.embedding_dim)
        self.word_embedding.weight.data.copy_(word_vectors)
        self.word_lstm = nn.LSTM(
            input_size=args.embedding_dim,
            hidden_size=args.hidden_dim,
            num_layers=args.lstm_layers,
            batch_first=False,
            dropout=args.lstm_dropout,
            bidirectional=True
        )
        self.word_conv = nn.Sequential(
            nn.Conv1d(in_channels=args.hidden_dim * 2 + args.embedding_dim, out_channels=100, kernel_size=3),
            nn.BatchNorm1d(100),
            nn.ReLU(inplace=True),
            nn.Conv1d(in_channels=100, out_channels=100, kernel_size=3),
            nn.BatchNorm1d(100),
            nn.ReLU(inplace=True)
        )

        # article
        self.art_embedding = nn.Embedding(args.art_vocab_size, args.embedding_dim)
        self.art_embedding.weight.data.copy_(art_vectors)
        self.art_lstm = nn.LSTM(
            input_size=args.embedding_dim,
            hidden_size=args.hidden_dim,
            num_layers=args.lstm_layers,
            batch_first=False,
            dropout=args.lstm_dropout,
            bidirectional=True
        )
        self.art_conv = nn.Sequential(
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
            nn.Linear(2 * (100 + 200), args.linear_hidden_size),
            nn.BatchNorm1d(args.linear_hidden_size),
            nn.ReLU(inplace=True),
            nn.Linear(args.linear_hidden_size, args.label_size)
        )

    def forward(self, text, article):
        word_embed = self.word_embedding(text)
        word_out = self.word_lstm(word_embed)[0].permute(1, 2, 0)
        word_out = torch.cat((word_out, word_embed.permute(1, 2, 0)), dim=1)
        word_conv_out = kmax_pooling(self.word_conv(word_out), 2, self.kmax_k)

        art_embed = self.art_embedding(article)
        art_out = self.art_lstm(art_embed)[0].permute(1, 2, 0)
        art_out = torch.cat((art_out, art_embed.permute(1, 2, 0)), dim=1)
        art_conv_out = kmax_pooling(self.art_conv(art_out), 2, self.kmax_k)

        out_cat = torch.cat((word_conv_out, art_conv_out), dim=1)
        flatten = out_cat.view(out_cat.size(0), -1)
        logits = self.fc(flatten)
        return logits





