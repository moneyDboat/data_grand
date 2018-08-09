# -*- coding: utf-8 -*-

"""
@Author  : captain
@time    : 18-7-28 上午11:11
@ide     : PyCharm  
"""

import pandas as pd
import numpy as np
import pickle
from torch.utils.data import Dataset, DataLoader
from MLP import MLP
import torch
from torch import nn
import torch.nn.functional as F
from torch.nn.utils import clip_grad_norm
from sklearn import metrics


class FeaDatasets(Dataset):
    def __int__(self, fea_vectors, labels):
        self.fea_datas = fea_vectors
        self.labels = labels

    def __len__(self):
        return self.fea_datas.shape[0]

    def __getitem__(self, idx):
        return self.fea_datas[idx], self.labels[idx]


print('load data......')
new_train = pd.read_csv('/data/yujun/datasets/daguanbei_data/new_train_set.csv')
val = pd.read_csv('/data/yujun/datasets/daguanbei_data/val_set.csv')
print('load data completed')

print('load feature......')
new_train_feature = pickle.load(open('fea/new_train_fea.pkl'))
val_feature = pickle.load(open('fea/val_fea.pkl'))
print('load complete!')

print('Shape of full_train_feature : {}'.format(new_train_feature.shape))
new_train_label = (new_train['class'] - 1).astype(int)
val_label = (val['class'] - 1).astype(int)

train_data = FeaDatasets(new_train_feature, new_train_label)
data_loader = DataLoader(train_data, batch_size=128, shuffle=True)
val_data = FeaDatasets(val_feature, val_label)
val_loader = DataLoader(val_data, batch_size=128, shuffle=False)

model = MLP()
model.cuda()
criterion = F.cross_entropy()
optimizer = torch.optim.Adam()

for i in range(10):
    total_loss = 0.0
    correct = 0
    total = 0

    model.train()
    for idx, (text, label) in enumerate(data_loader):
        text, label = text.cuda(), label.cuda()
        optimizer.zero_grad()
        pred = model(text)
        loss = criterion(pred, label)
        loss.backward()
        clip_grad_norm(model.parameters(), 10)
        optimizer.step()

        # 更新统计指标
        total_loss += loss.item()
        predicted = pred.max(1)[1]
        total += label.size(0)
        correct += predicted.eq(label).sum().item()

        if idx % 20 == 19:
            print('[{}, {}] loss: {:.3f} | Acc: {:.3f}%({}/{})'.format(i + 1, idx + 1, total_loss / 20,
                                                                       100. * correct / total, correct, total))
            total_loss = 0.0

    f1score = val(model, val_data)
    print('f1score on val set: {}'.format(f1score))


def val(model, dataset):
    # 计算模型在验证集上的分数

    # 将模型设为验证模式
    model.eval()

    acc_n = 0
    val_n = 0
    predict = np.zeros((0,), dtype=np.int32)
    gt = np.zeros((0,), dtype=np.int32)
    with torch.no_grad():
        for batch in dataset:
            text, label = batch.text, batch.label
            if opt.cuda:
                text, label = text.cuda(), label.cuda()
            outputs = model(text)
            pred = outputs.max(1)[1]
            acc_n += (pred == label).sum().item()
            val_n += label.size(0)
            predict = np.hstack((predict, pred.cpu().numpy()))
            gt = np.hstack((gt, label.cpu().numpy()))

    acc = 100. * acc_n / val_n
    f1score = np.mean(metrics.f1_score(predict, gt, average=None))
    print('* Test Acc: {:.3f}%({}/{}), F1 Score: {}'.format(acc, acc_n, val_n, f1score))
    return f1score
