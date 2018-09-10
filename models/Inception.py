# -*- coding: utf-8 -*-

"""
# @Author  : captain
# @Time    : 2018/9/5 22:43
# @Ide     : PyCharm
"""

from .BasicModule import BasicModule
import torch as t
import torch
import numpy as np
from torch import nn
from collections import OrderedDict


class Ince(nn.Module):
    def __init__(self, cin, co, relu=True, norm=True):
        super(Ince, self).__init__()
        assert (co % 4 == 0)
        cos = [co // 4] * 4
        self.activa = nn.Sequential()
        if norm:
            self.activa.add_module('norm', nn.BatchNorm1d(co))
        if relu:
            self.activa.add_module('relu', nn.ReLU(True))
        self.branch1 = nn.Sequential(OrderedDict([
            ('conv1', nn.Conv1d(cin, cos[0], 1, stride=1)),
        ]))
        self.branch2 = nn.Sequential(OrderedDict([
            ('conv1', nn.Conv1d(cin, cos[1], 1)),
            ('norm1', nn.BatchNorm1d(cos[1])),
            ('relu1', nn.ReLU(inplace=True)),
            ('conv3', nn.Conv1d(cos[1], cos[1], 3, stride=1, padding=1)),
        ]))
        self.branch3 = nn.Sequential(OrderedDict([
            ('conv1', nn.Conv1d(cin, cos[2], 3, padding=1)),
            ('norm1', nn.BatchNorm1d(cos[2])),
            ('relu1', nn.ReLU(inplace=True)),
            ('conv3', nn.Conv1d(cos[2], cos[2], 5, stride=1, padding=2)),
        ]))
        self.branch4 = nn.Sequential(OrderedDict([
            # ('pool',nn.MaxPool1d(2)),
            ('conv3', nn.Conv1d(cin, cos[3], 3, stride=1, padding=1)),
        ]))

    def forward(self, x):
        branch1 = self.branch1(x)
        branch2 = self.branch2(x)
        branch3 = self.branch3(x)
        branch4 = self.branch4(x)
        result = self.activa(torch.cat((branch1, branch2, branch3, branch4), 1))
        return result


class InCNN(BasicModule):
    def __init__(self, config, vectors=None):
        super(InCNN, self).__init__()
        self.opt = config
        self.embedding = nn.Embedding(config.vocab_size, config.embedding_dim)
        if vectors is not None:
            self.embedding.weight.data.copy_(vectors)

        self.conv = nn.Sequential(
            Ince(config.embedding_dim, 200),
            Ince(200, 200),
            nn.MaxPool1d(config.max_text_len)
        )

        self.fc = nn.Sequential(
            nn.Linear(200, config.linear_hidden_size),
            nn.BatchNorm1d(config.linear_hidden_size),
            nn.ReLU(inplace=True),
            nn.Linear(100, config.label_size)
        )

    def forward(self, text):
        embed = self.embedding(text)  # seq*batch*emb
        out = self.conv(embed.permute(1, 2, 0))  # batch*emb*seq

        flatten = out.view(out.size(0), -1)
        logits = self.fc(flatten)

        return logits
