# -*- coding: utf-8 -*-

"""
@Author  : captain
@time    : 18-7-10 下午3:46
@ide     : PyCharm  
"""
from torchtext import data, datasets
from tqdm import tqdm
import pandas as pd
from config import DefaultConfig
from torchtext.vocab import Vectors
import word2vec


# 定义Dataset
class GrandDataset(data.Dataset):
    name = 'Grand Dataset'

    @staticmethod
    def sort_key(ex):
        return len(ex.text)

    def __init__(self, path, text_field, label_field, test=False, **kwargs):
        fields = [('text', text_field), ('label', label_field)]
        examples = []
        csv_data = pd.read_csv(path)
        print('preparing examples...')
        # 可改成data.Example.fromCsv()
        if test:
            for text in tqdm(csv_data['word_seg']):
                examples.append(data.Example.fromlist([text, None], fields))
        else:
            for text, label in tqdm(zip(csv_data['word_seg'], csv_data['class'])):
                examples.append(data.Example.fromlist([text, label], fields))
        super(GrandDataset, self).__init__(examples, fields, **kwargs)


def load_data(opt):
    TEXT = data.Field(sequential=True, fix_length=1000)
    LABEL = data.Field(sequential=False, use_vocab=False)

    # load data
    train_path = opt.train_data_path
    val_path = opt.val_data_path
    test_path = opt.test_data_path

    train = GrandDataset(train_path, text_field=TEXT, label_field=LABEL)
    val = GrandDataset(val_path, text_field=TEXT, label_field=LABEL)
    test = GrandDataset(test_path, text_field=TEXT, label_field=None, test=True)

    # w2v_path = 'emb-100.bin'
    # print('loading word2vec {}'.format(w2v_path))
    # w2v = word2vec.load(w2v_path)
    # build the vocabulary

    # vectors = Vectors(name='word2vec', cache='.vector_cache/', unk_init=w2v.vectors)
    # 构建Vocub
    TEXT.build_vocab(val)
    LABEL.build_vocab(val)

    # 构建Iterator
    # 在 test_iter , sort一定要设置成 False, 要不然会被 torchtext 搞乱样本顺序
    train_iter = data.Iterator(dataset=train, batch_size=opt.batch_size, train=True, repeat=False, device=0)
    val_iter = data.Iterator(dataset=val, batch_size=opt.batch_size, train=False, repeat=False, sort=False, device=0)
    test_iter = data.Iterator(dataset=test, batch_size=opt.batch_size, train=False, sort=False, device=0)

    return train_iter, val_iter, test_iter, len(TEXT.vocab), len(LABEL.vocab)
