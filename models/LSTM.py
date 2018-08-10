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


class LSTM(BasicModule):
    def __init__(self, args, word_vectors, art_vectors):
        self.args = args
        super(LSTM, self).__init__()
        self.hidden_dim = args.hidden_dim
        self.batch_size = args.batch_size
        self.lstm_layers = args.lstm_layers

        # word lstm
        self.word_embedding = nn.Embedding(args.word_vocab_size, args.embedding_dim)
        self.word_embedding.weight.data.copy_(word_vectors)
        self.word_bilstm = nn.LSTM(args.embedding_dim, self.hidden_dim // 2, num_layers=self.lstm_layers,
                                   batch_first=False, dropout=args.lstm_dropout, bidirectional=True)

        # article lstm
        self.art_embedding = nn.Embedding(args.art_vocab_size, args.embedding_dim)
        self.art_embedding.weight.data.copy_(art_vectors)
        self.art_bilstm = nn.LSTM(args.embedding_dim, self.hidden_dim // 2, num_layers=self.lstm_layers,
                                  batch_first=False, dropout=args.lstm_dropout, bidirectional=True)

        self.hidden2label = nn.Linear(self.hidden_dim * 2, args.label_size)
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

    def forward(self, text, article):
        word_embed = self.word_embedding(text)  # seq*batch*emb
        word_out, _ = self.word_bilstm(word_embed)  # seq*batch*hidden
        final_word = word_out[-1]  # batch*hidden

        art_embed = self.art_embedding(article)
        art_out, _ = self.art_bilstm(art_embed)
        final_article = art_out[-1]

        # word+article
        final = torch.cat((final_word, final_article), 1)  # batch*(2*hidden)
        y = self.hidden2label(final)

        return y
