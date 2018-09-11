# -*- coding: utf-8 -*-

"""
# @Author  : captain
# @Time    : 2018/8/27 19:08
# @Ide     : PyCharm
"""

import torch
import time
import torch.nn.functional as F
import models
import data
from config import DefaultConfig
import pandas as pd
import os
import fire
from sklearn import metrics
import numpy as np

t1 = time.time()


def main(**kwargs):
    args = DefaultConfig()
    args.parse(kwargs)
    # boost模型
    args.max_epochs = 5

    if not torch.cuda.is_available():
        args.cuda = False
        args.device = None
        torch.manual_seed(args.seed)  # set random seed for cpu

    train_iter, val_iter, test_iter, args.vocab_size, vectors = util.load_data(args)

    args.print_config()

    # 模型保存位置
    if not os.path.exists(args.save_dir):
        os.mkdir(args.save_dir)
    save_path = os.path.join(args.save_dir, '{}_{}_{}.pth'.format(args.model, args.text_type, args.id))

    if args.cuda:
        torch.cuda.set_device(args.device)
        torch.cuda.manual_seed(args.seed)  # set random seed for gpu

    for lay_i in range(args.bo_layers):
        print('-------------- lay {} ---------------'.format(lay_i))
        model = getattr(models, args.model)(args, vectors)
        model = model.cuda()
        print(model)
        best_score = 0.0

        # 目标函数和优化器
        criterion = F.cross_entropy
        lr1 = args.lr1
        lr2 = args.lr2
        optimizer = model.get_optimizer(lr1, lr2, args.weight_decay)

        if lay_i != 0:
            # 加载上一层模型的loss weight
            saved_model = torch.load(args.model_path)
            loss_weight = saved_model['loss_weight']
            print(list(enumerate(loss_weight)))
            loss_weight = loss_weight.cuda()

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
                if lay_i != 0:
                    loss = criterion(pred, label, weight=loss_weight + 1 - loss_weight.mean())
                else:
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
            f1score, tmp_loss_weight = val(model, val_iter, args)
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
                lr2 = 2e-4 if lr2 == 0 else lr2 * 0.8
                optimizer = model.get_optimizer(lr1, lr2, 0)
                print('* load previous best model: {}'.format(best_score))
                print('* model lr:{}  emb lr:{}'.format(lr1, lr2))
                if lr1 < args.min_lr:
                    print('* training over, best f1 score: {}'.format(best_score))
                    break

        # 保存训练最终的模型
        # 保存当前层的loss weight
        loss_weight = tmp_loss_weight
        args.best_score = best_score
        final_model = {
            'state_dict': model.state_dict(),
            'config': args,
            'loss_weight': loss_weight
        }
        args.model_path = os.path.join(args.save_dir,
                                       '{}_{}_lay{}_{}.pth'.format(args.model, args.text_type, lay_i, best_score))
        torch.save(final_model, args.model_path)
        print('Best Final Model saved in {}'.format(args.model_path))

    t2 = time.time()
    print('time use: {}'.format(t2 - t1))


def val(model, dataset, args):
    # 计算模型在验证集上的分数

    # 将模型设为验证模式
    model.eval()
    sample_per_class = torch.zeros(args.label_size)
    error_per_class = torch.zeros(args.label_size)

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

            for pre, la in zip(pred, label):
                sample_per_class[la] += 1
                if pre != la:
                    error_per_class[la] += 1

    acc = 100. * acc_n / val_n
    f1score = np.mean(metrics.f1_score(predict, gt, average=None))
    print('* Test Acc: {:.3f}%({}/{}), F1 Score: {}'.format(acc, acc_n, val_n, f1score))
    return f1score, error_per_class / sample_per_class


if __name__ == '__main__':
    fire.Fire()
