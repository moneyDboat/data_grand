import pandas as pd
import numpy as np
import random

train_data = pd.read_csv('/data/yujun/captain/datasets1/train_set.csv')
val_data = pd.read_csv('/data/yujun/captain/datasets1/val_set.csv')
test_data = pd.read_csv('/data/yujun/datasets/daguanbei_data/test_set.csv')
print('raw data loaded!')

train_data[['word_seg', 'class']].to_csv('word/train_set.csv')
val_data[['word_seg', 'class']].to_csv('word/val_set.csv')
test_data[['id', 'word_seg']].to_csv('word/test_set.csv')
print('word data made!')

train_data[['article', 'class']].to_csv('article/train_set.csv')
val_data[['article', 'class']].to_csv('article/val_set.csv')
test_data[['id', 'article']].to_csv('article/test_set.csv')
print('article data made!')
