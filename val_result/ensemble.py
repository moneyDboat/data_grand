import numpy as np
import pandas as pd
from sklearn import metrics

val = pd.read_csv('/data/yujun/captain/datasets/val_set.csv')
val_target = (val['class']-1).astype(int)

# model_list = ['bigru_attention_word_0.775937678128.pth.npy', 'GRU_word_0.771128871335.pth.npy', 'GRU_word_e4_0.77133756.pth.npy', 'GRU_word_0.770047017921.pth.npy', 'RCNN1_word_0.769018920031.pth.npy', 'LSTM_word_0.768979776301.pth.npy', 'RCNN1_word_0.767783682451.pth.npy', 'TextCNN_word_0.760456622816.pth.npy', 'FastText_word_0.75425891649.pth.npy', 'GRU_article_0.747660923499.pth.npy', 'GRU_article_0.736854236207.pth.npy', 'TextCNN_article_0.735192435177.pth.npy',  'GRU_word_rev_0.721350215541.pth.npy']

# model_list = ['GRU_article_0.740587991389.pth.npy', 'GRU_word_aug_0.7531487.pth.npy', 'RCNN_article_0.738300624533.pth.npy', 'GRU_article_0.742706978708.pth.npy', 'GRU_word_aug_0.75959.pth.npy', 'RCNN_word_0.75951625072.pth.npy', 'GRU_article_0.746959892485.pth.npy',  'GRU_word_e2_0.766984819683.pth.npy', 'TextCNN_article_0.732215582073.pth.npy', 'GRU_article_0.747660923499.pth.npy', 'GRU_word_e3_0.770530528299.pth.npy', 'TextCNN_article_0.735192435177.pth.npy', 'GRU_article_e4_0.731274858243.pth.npy', 'GRU_word_e4_0.77133756.pth.npy', 'TextCNN_word_0.759799295233.pth.npy', 'GRU_word_0.766268382394.pth.npy', 'GRU_word_rev_0.721350215541.pth.npy', 'TextCNN_word_0.760456622816.pth.npy', 'FastText_article_0.700147562111.pth.npy', 'GRU_word_0.76633007379.pth.npy', 'TextCNN_word_0.760762381324.pth.npy', 'FastText_article_0.716388158416.pth.npy', 'GRU_word_0.769904744261.pth.npy', 'LSTM_word_0.768979776301.pth.npy', 'TextCNN_word_0.761957574957.pth.npy', 'FastText_word_0.739320289459.pth.npy', 'GRU_word_0.771128871335.pth.npy', 'RCNN1_word_0.768352769502.pth.npy',  'GRU_word_100_0.765372.pth.npy', 'RCNN1_word_0.769018920031.pth.npy', 'GRU_article_0.740203015091.pth.npy', 'GRU_word_200_0.76332.pth.npy', 'RCNN1_word_e4_0.773231775419.pth.npy']

model_list = ['bigru_attention_word_0.775937678128.pth.npy', 'GRU_word_rev_0.721350215541.pth.npy', 'GRU_word_0.771128871335.pth.npy', 'GRU_word_0.770047017921.pth.npy', 'GRU_word_0.769904744261.pth.npy', 'RCNN1_word_0.769018920031.pth.npy', 'LSTM_word_0.768979776301.pth.npy', 'RCNN1_word_0.767783682451.pth.npy', 'TextCNN_word_0.760456622816.pth.npy', 'FastText_word_0.75425891649.pth.npy', 'GRU_article_0.747660923499.pth.npy', 'GRU_article_0.736854236207.pth.npy', 'TextCNN_article_0.735192435177.pth.npy', 'GRU_word_e4_0.77133756.pth.npy']

# model_list = model_list[::-1]

prob_list = [np.load(model) for model in model_list]
weight = [1.0] * len(model_list)

avg_res = np.zeros(prob_list[0].shape)
for prob in prob_list:
    avg_res += prob
avg_res = np.argmax(avg_res, axis=1)
avg_f1 = np.mean(metrics.f1_score(val_target, avg_res, average=None))
print('Avg F1score: {}'.format(avg_f1))


score = [0.0] * len(model_list)
score[0] = float(model_list[0].split('_')[-1][:-8])
print('-' * 60)
print('first model {}...'.format(model_list[0]))

for idx, model in enumerate(model_list):
    if idx == 0:
        continue
    print('-' * 60)
    print('ensemble model {}...'.format(model_list[idx]))
    pre_res = np.zeros(prob_list[0].shape)
    sum_weight = 0.0
    for i in range(idx):
        pre_res += prob_list[i] * weight[i]
        sum_weight += weight[i]

    m_prob = np.load(model)
    best_score = score[idx-1]
    best_weight = 0.0
    for w in np.linspace(0, 3, 300):
        val_res = (pre_res + m_prob * w) / (sum_weight + w)
        val_res = np.argmax(val_res, axis=1)
        f1score = np.mean(metrics.f1_score(val_target, val_res, average=None))
        # print('for weight {:.2f}, f1score is {}'.format(w, f1score))
        if f1score > best_score:
            best_score = f1score
            best_weight = w
    score[idx] = best_score
    weight[idx] = best_weight
    print('best weight is {:.2f}, best f1score is {}\n'.format(best_weight, best_score))

print(weight)
print(score)
