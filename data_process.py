# -*- coding: utf-8 -*-

"""
# @Author  : captain
# @Time    : 2018/8/28 15:26
# @Ide     : PyCharm
"""

from torchtext import data
import pandas as pd
from torchtext.vocab import Vectors
from tqdm import tqdm
from torch.nn import init
import os
import fire
import pickle


# 定义Dataset
class GrandDataset(data.Dataset):

    @staticmethod
    def sort_key(ex):
        return len(ex.text)

    def __init__(self, path, text_field, label_field, text_type='word_seg', test=False, **kwargs):
        fields = [('text', text_field), ('label', label_field)]
        examples = []
        csv_data = pd.read_csv(path)
        print('read data from {}'.format(path))

        if test:
            # 如果为测试集，则不加载label
            for text in tqdm(csv_data[text_type]):
                examples.append(data.Example.fromlist([text, None], fields))
        else:
            for text, label in tqdm(zip(csv_data[text_type], csv_data['class'])):
                examples.append(data.Example.fromlist([text, label - 1], fields))
        super(GrandDataset, self).__init__(examples, fields, **kwargs)


def preprocess(data_path='/data/yujun/captain/datasets', text_type='word', max_text_len=2000, embedding_path='emb',
               emb_dim=300):
    # fix_length
    TEXT = data.Field(sequential=True, fix_length=max_text_len)  # 词或者字符
    LABEL = data.Field(sequential=False, use_vocab=False)

    # data path, word/ or article/
    train_path = data_path + '/train_set.csv'
    val_path = data_path + '/val_set.csv'
    test_path = data_path + '/test_set.csv'

    if text_type is 'word':
        text_type = 'word_seg'
    train = GrandDataset(train_path, text_field=TEXT, label_field=LABEL, text_type=text_type, test=False)
    val = GrandDataset(val_path, text_field=TEXT, label_field=LABEL, text_type=text_type, test=True)
    test = GrandDataset(test_path, text_field=TEXT, label_field=None, text_type=text_type, test=True)

    cache = '.vector_cache'
    if not os.path.exists(cache):
        os.mkdir(cache)
    embedding_path = '{}{}_{}.txt'.format(embedding_path, text_type, emb_dim)
    vectors = Vectors(name=embedding_path, cache=cache)
    print('load word2vec vectors from {}'.format(embedding_path))
    # 没有命中的token的初始化方式
    vectors.unk_init = init.xavier_uniform_

    # 构建Vocab
    print('building {} vocabulary......'.format(text_type))
    TEXT.build_vocab(train, val, test, min_freq=5, vectors=vectors)

    # pickle保存
    saved_path = 'data/{}'.format()
    if not os.path.exists('data'):
        os.mkdir('data')
    if not os.path.exists(saved_path):
        os.mkdir(saved_path)
    pickle.dump(train, open('data/{}/train.pkl'.format(text_type), 'wb'))
    pickle.dump(val, open('data/{}/val.pkl'.format(text_type), 'wb'))
    pickle.dump(test, open('data/{}/test.pkl'.format(text_type), 'wb'))
    print('preprocessed datasets saved in data/{}'.format(text_type))


if __name__ == '__main__':
    fire.Fire()
