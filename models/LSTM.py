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

def kmax_pooling(x, dim, k):
    index = x.topk(k, dim=dim)[1].sort(dim=dim)[0] # torch.Tensor.topk()的输出有两项，后一项为索引
    return x.gather(dim, index)


class LSTM(BasicModule):
    def __init__(self, args, word_vectors, art_vectors):
        self.kmax_pooling = args.kmax_pooling
        super(LSTM, self).__init__()

        # word lstm
        self.word_embedding = nn.Embedding(args.word_vocab_size, args.embedding_dim)
        self.word_embedding.weight.data.copy_(word_vectors)
        self.word_bilstm = nn.LSTM(
            input_size=args.embedding_dim,
            hidden_size=args.hidden_dim,
            num_layers=args.lstm_layers,
            batch_first=False,
            dropout=args.lstm_dropout,
            bidirectional=True)

        # article lstm
        self.art_embedding = nn.Embedding(args.art_vocab_size, args.embedding_dim)
        self.art_embedding.weight.data.copy_(art_vectors)
        self.art_bilstm = nn.LSTM(
            input_size=args.embedding_dim,
            hidden_size=args.hidden_dim,
            num_layers=args.lstm_layers,
            batch_first=False,
            dropout=args.lstm_dropout,
            bidirectional=True)

        # self.fc = nn.Linear(args.hidden_dim * 2 * 2, args.label_size)
        # 两层全连接层，中间添加批标准化层
        # 全连接层隐藏元个数需要再做修改
        self.fc = nn.Sequential(
            nn.Linear(self.kmax_pooling * (args.hidden_dim * 2 * 2), args.linear_hidden_size),
            nn.BatchNorm1d(args.linear_hidden_size),
            nn.ReLU(inplace=True),
            nn.Linear(100, args.label_size)
        )

        # self.hidden = self.init_hidden(args.batch_size)
        # self.mean = args.__dict__.get("lstm_mean", True)

    # 初始化参数，暂时用不上
    # def init_hidden(self, batch_size=None):
    #     if batch_size is None:
    #         batch_size = self.batch_size
    #
    #     '''
    #     if self.use_gpu:
    #         h0 = nn.Parameter(nn.init.xavier_uniform_(torch.Tensor(2 * self.lstm_layers, batch_size, self.hidden_dim // 2).type(torch.FloatTensor).cuda()), requires_grad=True)
    #         c0 = nn.Parameter(nn.init.xavier_uniform_(torch.Tensor(2 * self.lstm_layers, batch_size, self.hidden_dim // 2).type(torch.FloatTensor).cuda()), requires_grad=True)
    #     else:
    #     '''
    #     h0 = nn.Parameter(nn.init.xavier_uniform_(
    #         torch.Tensor(2 * self.lstm_layers, batch_size, self.hidden_dim // 2).type(torch.FloatTensor)),
    #         requires_grad=True)
    #     c0 = nn.Parameter(nn.init.xavier_uniform_(
    #         torch.Tensor(2 * self.lstm_layers, batch_size, self.hidden_dim // 2).type(torch.FloatTensor)),
    #         requires_grad=True)
    #     return (h0, c0)

    # 对LSTM所有隐含层的输出做kmax pooling
    def forward(self, text, article):
        word_embed = self.word_embedding(text)  # seq*batch*emb
        word_out = self.word_bilstm(word_embed)[0].permute(1, 2, 0)  # batch * hidden * seq
        word_pooling = kmax_pooling(word_out, 2, self.kmax_pooling)  # batch * hidden * kmax

        art_embed = self.art_embedding(article)
        art_out = self.art_bilstm(art_embed)[0].permute(1, 2, 0)
        art_pooling = kmax_pooling(art_out, 2, self.kmax_pooling)

        # word+article
        cat_out = torch.cat((word_pooling, art_pooling), dim=1)  # batch * (2*2*hidden) * kmax
        flatten = cat_out.view(cat_out.size(0), -1)
        logits = self.fc(flatten)

        return logits

    # def forward(self, text, article):
    #     word_embed = self.word_embedding(text)  # seq*batch*emb
    #     word_out, _ = self.word_bilstm(word_embed)  # seq*batch*hidden
    #     final_word = word_out[-1]  # batch*hidden
    #
    #     art_embed = self.art_embedding(article)
    #     art_out, _ = self.art_bilstm(art_embed)
    #     final_article = art_out[-1]
    #
    #     # word+article
    #     final = torch.cat((final_word, final_article), 1)  # batch*(2*hidden)
    #     y = self.fc(final)
    #
    #     return y
