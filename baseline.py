import pandas as pd, numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import LinearSVC
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
import time
import pickle
import os

t1 = time.time()

train = pd.read_csv('/data/yujun/datasets/daguanbei_data/new_train_set.csv')
val = pd.read_csv('/data/yujun/datasets/daguanbei_data/val_set.csv')
test = pd.read_csv('/data/yujun/datasets/daguanbei_data/test_set.csv')
print('load data completed')
# test_id = pd.read_csv('test_set.csv')[['id']].copy()

# load tfidf features in files if exists
tfidf_path = 'tfidf_feature/'
if os.path.exists(tfidf_path):
    train_feature = pickle.load(open('tfidf_feature/train_fea.pkl'))
    val_feature = pickle.load(open('tfidf_feature/val_fea.pkl'))
    test_feature = pickle.load(open('tfidf_feature/test_fea.pkl'))
    print('load tfidf features from {}'.format(tfidf_path))
else:
    os.mkdit(tfidf_path)
    column = 'word_seg'
    n = train.shape[0]
    vec = TfidfVectorizer(ngram_range=(1, 2), min_df=3, max_df=0.9, use_idf=1, smooth_idf=1, sublinear_tf=True)
    train_feature = vec.fit_transform(train[column])
    print('train tfidf feature completed!')
    val_feature = vec.transform(val[column])
    print('val tfidf feature completed!')
    test_feature = vec.transform(test[column])
    print('test tfidf feature completed!')

    # save tf-idf feature
    pickle.dump(val_feature, open(tfidf_path + 'val_fea.pkl', 'w'))
    pickle.dump(test_feature, open(tfidf_path + 'test_fea.pkl', 'w'))
    pickle.dump(train_feature, open(tfidf_path + 'train_fea.pkl', 'w'))
    print('tfidf features save in {}'.format(tfidf_path))

train_label = (train['class'] - 1).astype(int)
val_label = (val['class'] - 1).astype(int)

# svm classifier
print('training svm')
svm = LinearSVC()
svm.fit(train_feature, train_label)
svm_acc = svm.score(test_feature, val_label)
print('acc of SVM: {}'.format(svm_acc))

# logistic classifier
print('training logistic')
clf = LogisticRegression(C=4, dual=True)
clf.fit(train_feature, label)
log_acc = clf.score(test_feature, val_label)
print('acc of log: {}'.format(log_acc))

# preds = clf.predict_proba(test_feature)

# #保存概率文件
# test_prob=pd.DataFrame(preds)
# test_prob.columns=["class_prob_%s"%i for i in range(1,preds.shape[1]+1)]
# test_prob["id"]=list(test_id["id"])
# test_prob.to_csv('../sub_prob/prob_lr_baseline.csv',index=None)

# #生成提交结果
# preds=np.argmax(preds,axis=1)
# test_pred=pd.DataFrame(preds)
# test_pred.columns=["class"]
# test_pred["class"]=(test_pred["class"]+1).astype(int)
# print(test_pred.shape)
# print(test_id.shape)
# test_pred["id"]=list(test_id["id"])
# test_pred[["id","class"]].to_csv('baseline.csv',index=None)
# t2=time.time()
# print("time use:",t2-t1)
