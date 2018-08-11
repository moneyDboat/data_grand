# -*- coding: utf-8 -*-

"""
@Author  : captain
@time    : 18-7-10 下午8:55
@ide     : PyCharm  
"""

import torch
import torch.nn as nn
import numpy as np
from .BasicModule import BasicModule
from config import DefaultConfig

kernal_sizes = [1, 2, 3, 4, 5]


class TextCNN(BasicModule):
    def __init__(self, opt, vectors):
        super(TextCNN, self).__init__()

        '''Embedding Layer'''
        # 使用预训练的词向量
        self.embedding = nn.Embedding(opt.vocab_size, opt.embedding_dim)
        self.embedding.weight.data.copy_(vectors)

        convs = [
            nn.Sequential(
                nn.Conv1d(in_channels=opt.embedding_dim,
                          out_channels=opt.kernel_num,
                          kernel_size=kernel_size),
                nn.BatchNorm1d(opt.kernel_num),
                nn.ReLU(inplace=True),

                nn.Conv1d(in_channels=opt.kernel_num,
                          out_channels=opt.kernel_num,
                          kernel_size=kernel_size),
                nn.BatchNorm1d(opt.kernel_num),
                nn.ReLU(inplace=True),
                nn.MaxPool1d(kernel_size=(opt.max_text_len - kernel_size*2 + 2))
            )
            for kernel_size in kernal_sizes
        ]

        self.convs = nn.ModuleList(convs)

        self.fc = nn.Sequential(
            nn.Linear(5 * opt.kernel_num, opt.linear_hidden_size),
            nn.BatchNorm1d(opt.linear_hidden_size),
            nn.ReLU(inplace=True),
            nn.Linear(opt.linear_hidden_size, opt.label_size)
        )

    def forward(self, inputs):
        embeds = self.embedding(inputs)  # seq * batch * embed
        # 进入卷积层前需要将Tensor第二个维度变成emb_dim，作为卷积的通道数
        conv_out = [conv(embeds.permute(1, 2, 0)) for conv in self.convs]
        conv_out = torch.cat(conv_out, dim=1)

        flatten = conv_out.view(conv_out.size(0), -1)
        logits = self.fc(flatten)
        return logits
