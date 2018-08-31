# -*- coding: utf-8 -*-

"""
# @Author  : captain
# @Time    : 2018/8/31 14:10
# @Ide     : PyCharm
"""

import numpy as np
import pandas as pd

files = ['result/GRU_art_0.736854236207.npy', 'result/GRU_word_0.770.npy']
ensmble = np.zeros([102277, 19])

for file in files:
    prob = np.load(file)
    ensmble += prob

result = np.argmax(ensmble, axis=1)

test = pd.read_csv('/data/yujun/datasets/daguanbei_data/test_set.csv')
test_id = test['id'].copy()

test_pred = pd.DataFrame({'id': test_id, 'class': result})
test_pred['class'] = (test_pred['class'] + 1).astype(int)
test_pred[['id', 'class']].to_csv('ensmble.csv', index=None)
