# -*- coding: utf-8 -*-

"""
@Author  : captain
@time    : 18-7-10 下午8:44
@ide     : PyCharm  
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from .BasicModule import BasicModule
from config import DefaultConfig


# 有两个做法有待实验验证
# 1、kmax_pooling的使用，对所有RNN的输出做最大池化
# 2、分类器选用两层全连接层+BN层，还是直接使用一层全连接层
# 3、是否需要init_hidden

def kmax_pooling(x, dim, k):
    index = x.topk(k, dim=dim)[1].sort(dim=dim)[0]  # torch.Tensor.topk()的输出有两项，后一项为索引
    return x.gather(dim, index)


class LSTM(BasicModule):
    def __init__(self, config, vectors):
        super(LSTM, self).__init__()
        self.config = config
        self.kmax_pooling = config.kmax_pooling

        # LSTM
        self.embedding = nn.Embedding(config.vocab_size, config.embedding_dim)
        self.embedding.weight.data.copy_(vectors)
        self.bilstm = nn.LSTM(
            input_size=config.embedding_dim,
            hidden_size=config.hidden_dim,
            num_layers=config.lstm_layers,
            batch_first=False,
            dropout=config.lstm_dropout,
            bidirectional=True)

        # self.fc = nn.Linear(args.hidden_dim * 2 * 2, args.label_size)
        # 两层全连接层，中间添加批标准化层
        # 全连接层隐藏元个数需要再做修改
        self.fc = nn.Sequential(
            nn.Linear(self.kmax_pooling * (config.hidden_dim * 2), config.linear_hidden_size),
            nn.BatchNorm1d(config.linear_hidden_size),
            nn.ReLU(inplace=True),
            nn.Linear(config.linear_hidden_size, config.label_size)
        )

    # 对LSTM所有隐含层的输出做kmax pooling
    def forward(self, text):
        embed = self.embedding(text)  # seq*batch*emb
        out = self.bilstm(embed)[0].permute(1, 2, 0)  # batch * hidden * seq
        pooling = kmax_pooling(out, 2, self.kmax_pooling)  # batch * hidden * kmax

        # word+article
        flatten = pooling.view(pooling.size(0), -1)
        logits = self.fc(flatten)

        return logits
