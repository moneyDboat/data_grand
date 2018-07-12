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

best_acc = 0.0

def main(**kwargs):
    opt = DefaultConfig()
    train_iter, val_iter, test_iter, opt.vocab_size = util.load_data(opt)
    opt.cuda = torch.cuda.is_available()
    opt.print_config()

    global best_acc

    opt.kernel_sizes = [int(k) for k in opt.kernel_sizes.split(',')]
    opt.save_dir = os.path.join(opt.save_dir, 'model_best.pth.tar')

    # model
    model = getattr(models, opt.model)(opt)

    if opt.cuda:
        torch.cuda.set_device(opt.device)
        model.cuda()

    # 目标函数和优化器
    criterion = F.cross_entropy
    optimizer = torch.optim.Adam(model.parameters(), lr=opt.lr)
    model.train()

    for i in range(opt.epochs):
        total_loss = 0.0
        correct = 0
        total = 0
        
        model.train()

        for idx, batch in enumerate(train_iter):
            # 训练模型参数
            text, label = batch.text, batch.label
            if opt.cuda:
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

            if idx % 20 == 19:
                print('[{}, {}] loss: {:.3f} | Acc: {:.3f}%({}/{})'.format(i+1, idx+1, total_loss/20, 100.*correct/total, correct, total))
                total_loss = 0.0

        accuracy = val(model, train_iter, opt)
        if accuracy > best_acc:
            best_acc = accuracy
            checkpoint = {
                'state_dict': model.state_dict(),
                'acc': accuracy,
                'epoch': i + 1
            }
            torch.save(checkpoint, opt.save_dir)


def val(model, dataset, opt):
    # 计算模型在验证集上的分数

    # 将模型设为验证模式
    model.eval()

    acc_n = 0
    val_n = 0
    with torch.no_grad():
        for batch in dataset:
            text, label = batch.text, batch.label
            if opt.cuda:
                text, label = text.cuda(), label.cuda()
            outputs = model(text)
            pred = outputs.max(1)[1]
            acc_n += (pred == label).sum().item()
            val_n += label.size(0)
    
    acc = 100. * acc_n / val_n
    print('* Test Acc: {:.3f}%({}/{})'.format(acc, acc_n, val_n))
    return acc


if __name__ == '__main__':
    main()
