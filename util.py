# -*- coding: utf-8 -*-

"""
# @Author  : captain
# @Time    : 2018/8/28 16:15
# @Ide     : PyCharm
"""

from torchtext import data
import pandas as pd
from torchtext.vocab import Vectors
from tqdm import tqdm
from torch.nn import init
import random
import os
import numpy as np


# 定义Dataset
class GrandDataset(data.Dataset):
    name = 'Grand Dataset'

    @staticmethod
    def sort_key(ex):
        return len(ex.text)

    def __init__(self, path, text_field, label_field, text_type='word', test=False, aug=False, **kwargs):
        fields = [('text', text_field), ('label', label_field)]
        examples = []
        csv_data = pd.read_csv(path)
        print('read data from {}'.format(path))

        if text_type == 'word':
            text_type = 'word_seg'

        if test:
            # 如果为测试集，则不加载label
            for text in tqdm(csv_data[text_type]):
                examples.append(data.Example.fromlist([text, None], fields))
        else:
            for text, label in tqdm(zip(csv_data[text_type], csv_data['class'])):
                if aug:
                    # do augmentation
                    rate = random.random()
                    if rate > 0.5:
                        text = self.dropout(text)
                    else:
                        text = self.shuffle(text)
                examples.append(data.Example.fromlist([text, label - 1], fields))
        super(GrandDataset, self).__init__(examples, fields, **kwargs)

    def shuffle(self, text):
        text = np.random.permutation(text.strip().split())
        return ' '.join(text)

    def dropout(self, text, p=0.5):
        # random delete some text
        text = text.strip().split()
        len_ = len(text)
        indexs = np.random.choice(len_, int(len_ * p))
        for i in indexs:
            text[i] = ''
        return ' '.join(text)


def load_data(opt):
    # 不设置fix_length
    TEXT = data.Field(sequential=True, fix_length=opt.max_text_len)  # 词或者字符
    LABEL = data.Field(sequential=False, use_vocab=False)

    # load
    # word/ or article/
    train_path = opt.data_path + opt.text_type + '/train_set.csv'
    val_path = opt.data_path + opt.text_type + '/val_set.csv'
    test_path = opt.data_path + opt.text_type + '/test_set.csv'

    # aug for data augmentation
    if opt.aug:
        print('make augmentation datasets!')
    train = GrandDataset(train_path, text_field=TEXT, label_field=LABEL, text_type=opt.text_type, test=False,
                         aug=opt.aug)
    val = GrandDataset(val_path, text_field=TEXT, label_field=LABEL, text_type=opt.text_type, test=False)
    test = GrandDataset(test_path, text_field=TEXT, label_field=None, text_type=opt.text_type, test=True)

    cache = '.vector_cache'
    if not os.path.exists(cache):
        os.mkdir(cache)
    embedding_path = '{}/{}_{}.txt'.format(opt.embedding_path, opt.text_type, opt.embedding_dim)
    vectors = Vectors(name=embedding_path, cache=cache)
    print('load word2vec vectors from {}'.format(embedding_path))
    vectors.unk_init = init.xavier_uniform_  # 没有命中的token的初始化方式

    # 构建Vocab
    print('building {} vocabulary......'.format(opt.text_type))
    TEXT.build_vocab(train, val, test, min_freq=5, vectors=vectors)
    # LABEL.build_vocab(train)

    # 构建Iterator
    # 在 test_iter, shuffle, sort, repeat一定要设置成 False, 要不然会被 torchtext 搞乱样本顺序
    # 如果输入变长序列，sort_within_batch需要设置成true，使每个batch内数据按照sort_key降序进行排序
    train_iter = data.BucketIterator(dataset=train, batch_size=opt.batch_size, shuffle=True, sort_within_batch=False,
                                     repeat=False, device=opt.device)
    # val_iter = data.BucketIterator(dataset=val, batch_size=opt.batch_size, sort_within_batch=False, repeat=False,
    #                                device=opt.device)
    # train_iter = data.Iterator(dataset=train, batch_size=opt.batch_size, train=True, repeat=False, device=opt.device)
    val_iter = data.Iterator(dataset=val, batch_size=opt.batch_size, shuffle=False, sort=False, repeat=False,
                             device=opt.device)
    test_iter = data.Iterator(dataset=test, batch_size=opt.batch_size, shuffle=False, sort=False, repeat=False,
                              device=opt.device)

    return train_iter, val_iter, test_iter, len(TEXT.vocab), TEXT.vocab.vectors
