# -*- coding: utf-8 -*-

"""
@Author  : captain
@time    : 18-7-10 下午8:55
@ide     : PyCharm  
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from .BasicModule import BasicModule


class TextCNN(BasicModule):
    def __init__(self, opt, vectors):
        super(TextCNN, self).__init__()
        self.opt = opt
        emb_dim = opt.embedding_dim
        kernel_num = opt.kernel_num
        max_text_len = opt.max_text_len

        '''Embedding Layer'''
        # 最后一个索引为填充的标记文本
        # 使用预训练的词向量
        self.embedding = nn.Embedding(opt.vocab_size + 1, emb_dim)
        self.embedding.weight.data.copy_(vectors)

        self.conv1 = nn.Sequential(
            # 卷积层的激活函数
            # 将embedding dimension看做channels数
            nn.Conv1d(emb_dim, kernel_num, kernel_size=3),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=max_text_len - 3 + 1)
        )
        self.conv2 = nn.Sequential(
            nn.Conv1d(emb_dim, kernel_num, kernel_size=4),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=max_text_len - 4 + 1)
        )
        self.conv3 = nn.Sequential(
            nn.Conv1d(emb_dim, kernel_num, kernel_size=5),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=max_text_len - 5 + 1)
        )

        '''Dropout Layer'''
        self.layer = nn.Linear(opt.embedding_dim * 3, 100)
        self.dropout = nn.Dropout(opt.dropout_rate)

        '''Projection Layer & Output Layer'''
        self.output_layer = nn.Linear(100, opt.label_size)

    def forward(self, inputs):
        input_n = inputs.shape[1]
        emb_dim = self.opt.embedding_dim

        embeds = self.embedding(inputs)
        # 进入卷积层前需要将Tensor第二个维度变成emb_dim，作为卷积的通道数
        embeds = embeds.view([input_n, emb_dim, -1])

        # concatenate the tensors
        x = self.conv1(embeds)
        y = self.conv2(embeds)
        z = self.conv3(embeds)
        flatten = torch.cat((x.view(input_n, -1), y.view(input_n, -1), z.view(input_n, -1)), 1)

        out = F.relu(self.layer(flatten))
        out = self.dropout(out)
        out = self.output_layer(out)

        return out
