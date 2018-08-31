# -*- coding: utf-8 -*-

# @Time    : 18/7/16 下午9:17
# @Author  : Captain
# @Site    : 
# @File    : FastText.py
# @Software: PyCharm Community Edition

from .BasicModule import BasicModule
from config import DefaultConfig
import torch
from torch import nn
import numpy as np


class FastText(BasicModule):
    def __init__(self, config, vectors=None):
        super(FastText, self).__init__()
        self.config = config

        self.embedding = nn.Embedding(config.vocab_size, config.embedding_dim)
        if vectors is not None:
            self.embedding.weight.data.copy_(vectors)

        self.fc = nn.Sequential(
            nn.Linear(config.embedding_dim, config.linear_hidden_size),
            nn.BatchNorm1d(config.linear_hidden_size),
            nn.ReLU(inplace=True),
            nn.Linear(config.linear_hidden_size, config.label_size)
        )

    def forward(self, sentence):
        embed = self.embedding(sentence)  # seq * batch * emb
        mean_out = torch.mean(embed, dim=0).squeeze()  # batch * 2emb

        logit = self.fc(mean_out)
        return logit
