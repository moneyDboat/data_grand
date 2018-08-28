# -*- coding: utf-8 -*-

"""
# @Author  : captain
# @Time    : 2018/8/27 22:31
# @Ide     : PyCharm
"""

import torch
import time
import torch.nn.functional as F
import models
from config import DefaultConfig
import util
import os
import datetime
from models.TextCNN import TextCNN
from sklearn import metrics
from torch.nn.utils import clip_grad_norm_
import numpy as np
import sys
import pandas as pd

torch.backends.cudnn.benchmark = True


def main(**kwargs):
    args = DefaultConfig()
    if not torch.cuda.is_available():
        args.cuda = False
        args.device = None

    train_iter, val_iter, test_iter, args.vocab_size, vectors = util.load_data(args, args.text_type)

    args.print_config()

    # model
    model = getattr(models, args.model)(args, vectors)
    if args.model_path:
        model.load_state_dict(torch.load(args.model_path)['state_dict'])

    if args.cuda:
        torch.cuda.set_device(args.device)
        model.cuda()

    result = infer(model, test_iter, args)
    # np.save('{}.npy'.format(args.model), result)

    test = pd.read_csv('/data/yujun/datasets/daguanbei_data/test_set.csv')
    test_id = test['id'].copy()

    test_pred = pd.DataFrame({'id': test_id, 'class': result})
    test_pred['class'] = (test_pred['class'] + 1).astype(int)
    test_pred[['id', 'class']].to_csv('{}.csv'.format(args.model), index=None)


def infer(model, dataset, opt):
    # 将模型设为验证模式
    model.eval()

    result = np.zeros((0,))
    with torch.no_grad():
        for batch in dataset:
            text = batch.text
            if opt.cuda:
                text = text.cuda()
            outputs = model(text)
            pred = outputs.max(1)[1]
            # probs = F.softmax(outputs, dim=1)
            result = np.hstack((result, pred.cpu().numpy()))
    return result


if __name__ == '__main__':
    main()
