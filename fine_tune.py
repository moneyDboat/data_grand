# -*- coding: utf-8 -*-

"""
# @Author  : captain
# @Time    : 2018/8/29 23:06
# @Ide     : PyCharm
"""

import torch
import time
import torch.nn.functional as F
import models
import util
import pandas as pd
import os
import fire
from sklearn import metrics
import numpy as np

t1 = time.time()


def tune(model_path=None, device=0):
    # 仅需设置model_path和device
    if model_path is None:
        print('model_path is needed!')
        return

    # load model and set small learning rate
    saved_model = torch.load(model_path)
    config = saved_model['config']
    config.model_path = model_path
    config.device = device
    config.lr1 = 5e-5
    config.lr2 = 5e-5
    config.max_epochs = 10

    config.print_config()

    train_iter, val_iter, test_iter, config.vocab_size, vectors = util.load_data(config)
    config.print_config()

    model = getattr(models, config.model)(config, vectors)
    model.load_state_dict(saved_model['state_dict'])
    print(model)
    best_score = config.best_score
    print('Load model from {}!'.format(config.model_path))
    print('Tmp best f1 score: {}'.format(best_score))

    # 模型保存位置
    save_path = os.path.join(config.save_dir, 'tune_{}_{}.pth'.format(config.model, config.id))

    if config.cuda:
        torch.cuda.set_device(config.device)
        model.cuda()

    # 目标函数和优化器
    criterion = F.cross_entropy
    lr1, lr2 = config.lr1, config.lr2
    optimizer = model.get_optimizer(lr1, lr2, config.weight_decay)

    for i in range(config.max_epochs):
        total_loss = 0.0
        correct = 0
        total = 0

        model.train()

        for idx, batch in enumerate(train_iter):
            # 训练模型参数
            # 使用BatchNorm层时，batch size不能为1
            if len(batch) == 1:
                continue
            text, label = batch.text, batch.label
            if config.cuda:
                text, label = text.cuda(), label.cuda()

            optimizer.zero_grad()
            pred = model(text)
            loss = criterion(pred, label)
            loss.backward()
            optimizer.step()

            # 更新统计指标
            total_loss += loss.item()
            predicted = pred.max(1)[1]
            total += label.size(0)
            correct += predicted.eq(label).sum().item()

            if idx % 80 == 79:
                print('[{}, {}] loss: {:.3f} | Acc: {:.3f}%({}/{})'.format(i + 1, idx + 1, total_loss / 20,
                                                                           100. * correct / total, correct, total))
                total_loss = 0.0

        # 计算再验证集上的分数，并相应调整学习率
        f1score = val(model, val_iter, config)
        if f1score > best_score:
            best_score = f1score
            checkpoint = {
                'state_dict': model.state_dict(),
                'config': config
            }
            torch.save(checkpoint, save_path)
            print('Best tmp model f1score: {}'.format(best_score))
        if f1score < best_score:
            model.load_state_dict(torch.load(save_path)['state_dict'])
            lr1 *= config.lr_decay
            lr2 *= 0.8
            optimizer = model.get_optimizer(lr1, lr2, 0)
            print('* load previous best model: {}'.format(best_score))
            print('* model lr:{}  emb lr:{}'.format(lr1, lr2))
            if lr1 < config.min_lr:
                print('* training over, best f1 score: {}'.format(best_score))
                break

    # 保存训练最终的模型
    config.best_score = best_score
    final_model = {
        'state_dict': model.state_dict(),
        'config': config
    }
    best_model_path = os.path.join(config.save_dir,
                                   'tune_{}_{}_{}.pth'.format(config.model, config.text_type, best_score))
    torch.save(final_model, best_model_path)
    print('Best Final Model saved in {}'.format(best_model_path))

    # 在测试集上运行模型并生成概率结果和提交结果
    if not os.path.exists('result/'):
        os.mkdir('result/')
    probs, test_pred = test(model, test_iter, config)
    result_path = 'result/' + 'tune_{}_{}_{}'.format(config.model, config.id, config.best_score)
    np.save('{}.npy'.format(result_path), probs)
    print('Prob result {}.npy saved!'.format(result_path))

    test_pred[['id', 'class']].to_csv('{}.csv'.format(result_path), index=None)
    print('Result {}.csv saved!'.format(result_path))

    t2 = time.time()
    print('time use: {}'.format(t2 - t1))


def test(model, test_data, args):
    # 生成测试提交数据csv
    # 将模型设为验证模式
    model.eval()

    result = np.zeros((0,))
    probs_list = []
    with torch.no_grad():
        for batch in test_data:
            text = batch.text
            if args.cuda:
                text = text.cuda()
            outputs = model(text)
            probs = F.softmax(outputs, dim=1)
            probs_list.append(probs.cpu().numpy())
            pred = outputs.max(1)[1]
            result = np.hstack((result, pred.cpu().numpy()))

    # 生成概率文件npy
    prob_cat = np.concatenate(probs_list, axis=0)

    test = pd.read_csv('/data/yujun/datasets/daguanbei_data/test_set.csv')
    test_id = test['id'].copy()
    test_pred = pd.DataFrame({'id': test_id, 'class': result})
    test_pred['class'] = (test_pred['class'] + 1).astype(int)

    return prob_cat, test_pred


def val(model, dataset, args):
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
            if args.cuda:
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


if __name__ == '__main__':
    fire.Fire()
