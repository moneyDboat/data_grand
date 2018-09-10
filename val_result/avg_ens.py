import numpy as np
import pandas as pd
from sklearn import metrics

val = pd.read_csv('/data/yujun/captain/datasets/val_set.csv')
val_target = (val['class']-1).astype(int)

# model_list = ['GRU_word_rev_0.721350215541.pth.npy', 'GRU_word_0.771128871335.pth.npy', 'GRU_word_0.770047017921.pth.npy', 'GRU_word_0.769904744261.pth.npy', 'RCNN1_word_0.769018920031.pth.npy', 'LSTM_word_0.768979776301.pth.npy', 'RCNN1_word_0.767783682451.pth.npy', 'TextCNN_word_0.760456622816.pth.npy', 'FastText_word_0.75425891649.pth.npy', 'GRU_article_0.747660923499.pth.npy', 'GRU_article_0.736854236207.pth.npy', 'TextCNN_article_0.735192435177.pth.npy', 'GRU_word_e4_0.77133756.pth.npy']

model_list = ['GRU_word_0.771128871335.pth.npy', 'bigru_attention_word_0.775937678128.pth.npy']

# model_list = model_list[::-1]

prob_list = [np.load(model) for model in model_list]
weight = [1.0] * len(model_list)

avg_res = np.zeros(prob_list[0].shape)
for prob in prob_list:
    avg_res += prob
avg_res = np.argmax(avg_res, axis=1)
avg_f1 = np.mean(metrics.f1_score(val_target, avg_res, average=None))
print('Avg F1score: {}'.format(avg_f1))

