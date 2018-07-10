# -*- coding: utf-8 -*-

"""
@Author  : captain
@time    : 18-7-10 下午8:55
@ide     : PyCharm  
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable


class TextCNN(nn.Module):
    def __init__(self, args):
        super(TextCNN, self).__init__()
        emb_dim = args.embed_dim
        vocab_size = args.embed_num
        class_num = args.class_num
        n_filters = args.kernel_num
        max_len = args.max_len
        dropout = args.dropout

        self.emb_dim = emb_dim

        '''Embedding Layer'''
        # if init_W is None:
        # 最后一个索引为填充的标记文本
        self.embedding = nn.Embedding(vocab_size + 1, emb_dim)

        self.conv1 = nn.Sequential(
            # 卷积层的激活函数
            # 将embedding dimension看做channels数
            nn.Conv1d(emb_dim, n_filters, kernel_size=3),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=max_len - 3 + 1)
        )
        self.conv2 = nn.Sequential(
            nn.Conv1d(emb_dim, n_filters, kernel_size=4),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=max_len - 4 + 1)
        )
        self.conv3 = nn.Sequential(
            nn.Conv1d(emb_dim, n_filters, kernel_size=5),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=max_len - 5 + 1)
        )

        '''Dropout Layer'''
        self.layer = nn.Linear(n_filters * 3, 100)
        self.dropout = nn.Dropout(dropout)

        '''Projection Layer & Output Layer'''
        self.output_layer = nn.Linear(100, class_num)

    def forward(self, inputs):
        size = len(inputs)
        embeds = self.embedding(inputs)

        # 进入卷积层前需要将Tensor第二个维度变成emb_dim，作为卷积的通道数
        embeds = embeds.view([len(embeds), self.emb_dim, -1])
        # concatenate the tensors
        x = self.conv1(embeds)
        y = self.conv2(embeds)
        z = self.conv3(embeds)
        flatten = torch.cat((x.view(size, -1), y.view(size, -1), z.view(size, -1)), 1)

        out = F.tanh(self.layer(flatten))
        out = self.dropout(out)
        out = F.tanh(self.output_layer(out))

        return out
