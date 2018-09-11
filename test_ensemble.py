# -*- coding: utf-8 -*-

"""
# @Author  : captain
# @Time    : 2018/8/31 20:21
# @Ide     : PyCharm
"""

import torch
import models
import data
import numpy as np
import torch.nn.functional as F
import pandas as pd
from sklearn import metrics
import fire
from config import DefaultConfig



def val_ensemble():
    files = []
    ensmble = np.zeros([10000, 19])

    for file in files:
        prob = np.load(file)
        ensmble += prob

    result = np.argmax(ensmble, axis=1)
    cal_f1(result)


def cal_f1(result):
    val = pd.read_csv('/data/yujun/captain/datasets/word/val_set.csv')
    val_label = val['class'].as_matrix()
    val_label -= 1
    f1score = np.mean(metrics.f1_score(result, val_label, average=None))
    print('F1 Score: {}'.format(f1score))


def gen_test(model_path=None, device=0):
    saved_model = torch.load(model_path)
    config = saved_model['config']
    config.device = int(device)
    print('Load model from {}'.format(model_path))

    _, val_iter, _, _, _ = util.load_data(config)

    model = getattr(models, config.model)(config)
    model.load_state_dict(saved_model['state_dict'])
    torch.cuda.set_device(config.device)
    model = model.cuda()
    print(model)

    model.eval()
    probs_list = []
    with torch.no_grad():
        for batch in val_iter:
            text, label = batch.text, batch.label
            text, label = text.cuda(), label.cuda()
            outputs = model(text)
            probs = F.softmax(outputs, dim=1)
            probs_list.append(probs.cpu().numpy())
    prob_cat = np.concatenate(probs_list, axis=0)
    np.save('val_{}.npy'.format(model_path), prob_cat)
    print('Prob result val_{}.npy saved!'.format(model_path))


if __name__ == '__main__':
    fire.Fire()
