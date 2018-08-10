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


class RCNN(BasicModule):
    def __init__(self, args, word_vectors, art_vectors):
        super(RCNN, self).__init__()

        # word
        self.word_embedding = nn.Embedding(args.word_vocab_size, args.embedding_dim)
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
        self.art_lstm = nn.LSTM(
            input_size=args.embedding_dim,
            hidden_size=args.hidden_dim,
            num_layers=args.lstm_layers,
            batch_first=False,
            dropout=args.lstm_dropout,
            bidirectional=True
        )
        self.art_conv = nn.Sequential(
            nn.Conv1d(in_channels=args.hidden_dim * 2 + args.embedding_dim, out_channels=100, kernel_size=3),
            nn.BatchNorm1d(100),
            nn.ReLU(inplace=True),
            nn.Conv1d(in_channels=100, out_channels=100, kernel_size=3),
            nn.BatchNorm1d(100),
            nn.ReLU(inplace=True)
        )

        # classifer
        self.fc = nn.Linear(2 * (100 + 100), args.label_size)
        # self.fc = nn.Sequential(
        #     nn.Linear(2 * (100 + 100), 2000),
        #     nn.BatchNorm1d(2000),
        #     nn.ReLU(inplace=True),
        #     nn.Linear(2000, args.label_size)
        # )

    def forward(self, text, article):
        word_embed = self.word_embedding(text)
        word_out, _ = self.word_lstm(word_embed)


