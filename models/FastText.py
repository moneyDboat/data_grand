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
    def __init__(self, opt, vectors):
        super(FastText, self).__init__()
        self.model_name = 'fastText'
        self.opt = opt
        self.embedding = nn.Embedding(opt.vocab_size, opt.embedding_dim)
        self.embedding.weight.data.copy_(vectors)

        self.pre1 = nn.Sequential(
            nn.Linear(opt.embedding_dim, opt.embedding_dim * 2),
            nn.BatchNorm1d(opt.embedding_dim * 2),
            nn.ReLU(True)
        )
        self.fc = nn.Sequential(
            nn.Linear(opt.embedding_dim * 4, opt.linear_hidden_size),
            nn.BatchNorm1d(opt.linear_hidden_size),
            nn.ReLU(inplace=True),
            nn.Linear(opt.linear_hidden_size, opt.label_size)
        )

    def forward(self, sentence):
        embed = self.embedding(sentence)


