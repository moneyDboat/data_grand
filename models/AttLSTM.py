# -*- coding: utf-8 -*-

"""
# @Author  : captain
# @Time    : 2018/8/12 15:35
# @Ide     : PyCharm
"""

import torch
import numpy as np
import torch.nn as nn
from .BasicModule import BasicModule
from config import DefaultConfig


class AttLSTM(BasicModule):
    def __init__(self, args, vectors):
        super(AttLSTM, self).__init__()
        self.args = args

        # LSTM
        self.embedding = nn.Embedding(args.vocab_size, args.embedding_dim)
        self.embedding.weight.data.copy_(vectors)
        self.bilstm = nn.LSTM(
            input_size=args.embedding_dim,
            hidden_size=args.hidden_dim,
            num_layers=args.lstm_layers,
            batch_first=False,
            dropout=args.lstm_dropout,
            bidirectional=True
        )

        self.fc = nn.Sequential(
            nn.Linear(args.hidden_dim * 2, args.linear_hidden_size),
            nn.BatchNorm1d(args.linear_hidden_size),
            nn.ReLU(inplace=True),
            nn.Linear(args.linear_hidden_size, args.label_size)
        )

    def attention(self, rnn_out, state):
        merged_state = torch.cat([s for s in state], 1)
        merged_state = merged_state.unsqueeze(2)
        # (batch, seq, hidden) * (batch, hidden, 1) = (batch, seq, 1)
        weights = torch.bmm(rnn_out.permute(0, 2, 1), merged_state)
        weights = torch.nn.functional.softmax(weights.squeeze(2), dim=1).unsqueeze(2)
        # (batch, hidden, seq) * (batch, seq, 1) = (batch, hidden, 1)
        return torch.bmm(rnn_out, weights).squeeze(2)

    def forward(self, text):
        embed = self.embedding(text)  # seq * batch * emb
        out, hidden = self.bilstm(embed)
        out = out.permute(1, 2, 0)  # batch * hidden * seq
        h_n, c_n = hidden
        att_out = self.attention(out, h_n)

        logits = self.fc(att_out)
        return logits
