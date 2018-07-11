# -*- coding: utf-8 -*-

"""
@Author  : captain
@time    : 18-7-10 下午8:42
@ide     : PyCharm  
"""

import torch
import time
import torch
import torch.autograd as autograd
import torch.nn.functional as F
import models
import util
from config import DefaultConfig
import os
import datetime
from models.TextCNN import TextCNN


def main(**kwargs):
    opt = DefaultConfig()
    train_iter, val_iter, test_iter, opt.vocab_size, opt.label_size = util.load_data(opt)
    opt.cuda = torch.cuda.is_available()
    opt.print_config()

    opt.kernel_sizes = [int(k) for k in opt.kernel_sizes.split(',')]
    opt.save_dir = os.path.join(opt.save_dir, datetime.datetime.now().strftime('%Y-%m-%d_%H-%M-%S'))

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
        for batch in train_iter:
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
            total_loss += loss

        print('epoch {} loss : {}'.format(i, total_loss))
        val(model, train_iter, opt)


def val(model, dataset, opt):
    # 计算模型在验证集上的分数

    # 将模型设为验证模式
    model.eval()

    acc_n = 0
    val_n = 0
    for batch in dataset:
        text, label = batch.text, batch.label
        if opt.cuda:
            text, label = text.cuda(), label.cuda()
            pred = model(text)
            pred = torch.max(F.softmax(pred), 1)[1]
            pred_label = pred.cpu().data.numpy().squeeze()
            target_y = label.cpu().data.numpy()
            acc_n += sum(pred_label == target_y)
            val_n += len(dataset)

    print('acc : %.2f' % (acc_n * 100 / val_n))

    # 将模型恢复为训练模式
    model.train()


if __name__ == '__main__':
    import fire

    fire.Fire()
