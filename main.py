# -*- coding: utf-8 -*-

"""
@Author  : captain
@time    : 18-7-10 下午8:42
@ide     : PyCharm
"""

import torch
import time
import torch.nn.functional as F
import models
import util
from config import DefaultConfig
import pandas as pd
import os
import fire
from sklearn import metrics
import numpy as np

best_score = 0.0
t1 = time.time()


def main(**kwargs):
    args = DefaultConfig()
    args.parse(kwargs)
    if not torch.cuda.is_available():
        args.cuda = False
        args.device = None

    train_iter, val_iter, test_iter, args.vocab_size, vectors = util.load_data(args)

    args.print_config()

    global best_score

    # init model
    model = getattr(models, args.model)(args, vectors)
    print(model)

    # 模型保存位置
    if not os.path.exists(args.save_dir):
        os.mkdir(args.save_dir)
    save_path = os.path.join(args.save_dir, '{}_{}.pth'.format(args.model, args.id))

    if args.cuda:
        torch.cuda.set_device(args.device)
        model.cuda()

    # 目标函数和优化器
    criterion = F.cross_entropy
    lr1, lr2 = args.lr1, args.lr2
    optimizer = model.get_optimizer(lr1, lr2, args.weight_decay)

    for i in range(args.max_epochs):
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
            if args.cuda:
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
        f1score = val(model, val_iter, args)
        if f1score > best_score:
            best_score = f1score
            checkpoint = {
                'state_dict': model.state_dict(),
                'config': args
            }
            torch.save(checkpoint, save_path)
            print('Best tmp model f1score: {}'.format(best_score))
        if f1score < best_score:
            model.load_state_dict(torch.load(save_path)['state_dict'])
            lr1 *= args.lr_decay
            lr2 = 2e-4 if lr2 == 0 else lr2 * args.lr_decay
            optimizer = model.get_optimizer(lr1, lr2, 0)
            print('* load previous best model: {}'.format(best_score))
            print('* model lr:{}  emb lr:{}'.format(lr1, lr2))
            if lr1 < args.min_lr:
                print('* training over, best f1 score: {}'.format(best_score))
                break

    # 保存训练最终的模型
    args.best_score = best_score
    final_model = {
        'state_dict': model.state_dict(),
        'config': args
    }
    best_model_path = os.path.join(args.save_dir, '{}_{}_{}.pth'.format(args.model, args.text_type, best_score))
    torch.save(final_model, best_model_path)
    print('Best Final Model saved in {}'.format(best_model_path))

    # 在测试集上运行模型并生成概率结果和提交结果
    if not os.path.exists('result/'):
        os.mkdir('result/')
    probs, test_pred = test(model, test_iter, args)
    result_path = 'result/' + '{}_{}_{}'.format(args.model, args.id, args.best_score)
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
