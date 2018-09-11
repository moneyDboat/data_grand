# -*- coding: utf-8 -*-

"""
# @Author  : captain
# @Time    : 2018/8/27 22:31
# @Ide     : PyCharm
"""

import torch
import models
from config import DefaultConfig
import data
import fire
import numpy as np
import torch.nn.functional as F
import pandas as pd


def main(**kwargs):
    args = DefaultConfig()
    args.parse(kwargs)

    train_iter, val_iter, test_iter, args.vocab_size, vectors = util.load_data(args, args.text_type)

    args.print_config()

    # model
    if args.model_path:
        # 加载模型
        saved_model = torch.load(args.model_path)
        config = saved_model['config']
        config.device = args.device
        model = getattr(models, args.model)(args, vectors)
        model.load_state_dict(saved_model['state_dict'])
        best_score = saved_model['best_score']
        print('Load model from {}!'.format(args.model_path))
    else:
        print("No trained model!")

    if not torch.cuda.is_available():
        config.cuda = False
        config.device = None

    if args.cuda:
        torch.cuda.set_device(args.device)
        model.cuda()

    probs = infer(model, test_iter, config)
    result_path = 'result/' + '{}_{}_{}'.format(args.model, args.id, args.best_score)
    np.save('{}.npy'.format(result_path), probs)
    print('Prob result {}.npy saved!'.format(result_path))
    # np.save('{}.npy'.format(args.model), result)

    # test = pd.read_csv('/data/yujun/datasets/daguanbei_data/test_set.csv')
    # test_id = test['id'].copy()
    #
    # test_pred = pd.DataFrame({'id': test_id, 'class': result})
    # test_pred['class'] = (test_pred['class'] + 1).astype(int)
    # test_pred[['id', 'class']].to_csv('{}.csv'.format(args.model), index=None)


def infer(model, test_iter, opt):
    # 将模型设为验证模式
    model.eval()

    result = np.zeros((0,))
    probs_list = []
    with torch.no_grad():
        for batch in test_iter:
            text = batch.text
            if opt.cuda:
                text = text.cuda()
            outputs = model(text)
            probs = F.softmax(outputs, dim=1)
            probs_list.append(probs.cpu().numpy())

    prob_cat = np.concatenate(probs_list, axis=0)
    return prob_cat


if __name__ == '__main__':
    fire.Fire()
