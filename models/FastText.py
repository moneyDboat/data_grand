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
    def __init__(self, opt):
        opt = DefaultConfig
        super(FastText, self).__init__()
        self.model_name = 'fastText'
        self.opt = opt
        self.embedding = nn.Embedding(opt.vocab_size, opt.embedding_dim)
        self.pre1 = nn.Sequential(
            nn.Linear(opt.embedding_dim, opt.embedding_dim * 2),
            nn.BatchNorm1d(opt.embedding_dim * 2),
            nn.ReLU(True)
        )
        self.fc = nn.Sequential