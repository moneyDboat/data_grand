import numpy as np
import pandas as pd

mlist = ['GRU_word_rev_0.721350215541.npy', 'GRU_k3_0.771128871335.npy', 'GRU_lay2_0.770047017921.npy', 'GRU_k4_0.769904744261.npy', 'RCNN1_512_0.769018920031.npy', 'LSTM_word_0.768979776301.npy', 'RCNN1_200_0.767783682451.npy', 'TextCNN_base_0.760456622816.npy', 'FastText_word_0.75425891649.npy', 'GRU_art2_0.747660923499.npy', 'GRU_art_0.736854236207.npy', 'TextCNN_art_0.735192435177.npy', 'GRU_e4_1_0.771337563812.npy']

prob_list = [np.load(model) for model in mlist]
pre_res = np.zeros(prob_list[0].shape)
for prob in prob_list:
    pre_res += prob
dl_res = pre_res / len(mlist)

n_dl = np.load('n_dl3.npy')

# svm_res = np.load('ensemble_lgb_svc_preds2.npy')
svm_res = np.load('ensemble_lgb_svc_preds10.npy')
all_res = 0.9*dl_res + svm_res + 0.9*n_dl

# new_res = np.load('n_RCNN1_e4_0.767059778648.npy')
# all_res += new_res

res = np.argmax(all_res, axis=1)

test = pd.read_csv('/data/yujun/datasets/daguanbei_data/test_set.csv')
test_id = test['id'].copy()

test_pred = pd.DataFrame({'id': test_id, 'class': res})
test_pred['class'] = (test_pred['class']+1).astype(int)
test_pred[['id', 'class']].to_csv('esb_3.csv', index=None)
print('ensemble csv saved in {}'.format('esb_3.csv'))
