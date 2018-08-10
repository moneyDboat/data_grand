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
import os
import datetime
from models.TextCNN import TextCNN
from sklearn import metrics
from torch.nn.utils import clip_grad_norm_
import numpy as np

best_score = 0.0


def main(**kwargs):
    args = DefaultConfig()
    train_iter, val_iter, args.word_vocab_size, args.art_vocab_size, word_vectors, art_vectors = util.load_data(args)

    args.cuda = torch.cuda.is_available()
    args.print_config()

    global best_score

    args.kernel_sizes = [int(k) for k in args.kernel_sizes.split(',')]
    # 模型保存位置
    if not os.path.exists(args.save_dir):
        os.mkdir(args.save_dir)
    save_path = os.path.join(args.save_dir, '{}_{}.pth.tar'.format(args.model, args.id))

    # model
    model = getattr(models, args.model)(args, word_vectors, art_vectors)

    # fix the parameters of embedding layers
    # for layer, param in enumerate(model.parameters()):
    #    if layer == 0:
    #        param.requires_grad = False

    if args.cuda:
        torch.cuda.set_device(args.device)
        model.cuda()

    # 目标函数和优化器
    criterion = F.cross_entropy
    lr1, lr2 = args.lr1, args.lr2
    optimizer = model.get_optimizer(lr1, lr2, args.weight_decay)

    for i in range(args.epochs):
        total_loss = 0.0
        correct = 0
        total = 0

        model.train()

        for idx, batch in enumerate(train_iter):
            # 训练模型参数
            text, article, label = batch.text, batch.article, batch.label
            if args.cuda:
                text, article, label = text.cuda(), article.cuda(), label.cuda()

            optimizer.zero_grad()
            pred = model(text, article)
            loss = criterion(pred, label)
            loss.backward()
            # gradient clipping
            total_norm = clip_grad_norm_(model.parameters(), 10)
            # if total_norm > 10:
            #     print("clipping gradient: {} with coef {}".format(total_norm, 10 / total_norm))
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

        f1score = val(model, val_iter, args)
        if f1score > best_score:
            best_score = f1score
            checkpoint = {
                'state_dict': model.state_dict(),
                'f1score': f1score,
                'epoch': i + 1,
                'config': args
            }
            torch.save(checkpoint, save_path)
            print('Best tmp model f1score: {}'.format(best_score))
        if f1score < best_score:
            model.load_state_dict(torch.load(save_path)['state_dict'])
            lr1 *= 0.8
            lr2 = 2e-4 if lr2 == 0 else lr2 * 0.8
            optimizer = model.get_optimizer(lr1, lr2, args.weight_decay)


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
            text, article, label = batch.text, batch.article, batch.label
            if args.cuda:
                text, article, label = text.cuda(), article, label.cuda()
            outputs = model(text, article)
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
    main()
