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
import argparse
import os
import datetime
from models.TextCNN import TextCNN

parser = argparse.ArgumentParser(description='CNN text classificer')
# learning
parser.add_argument('-lr', type=float, default=0.001, help='initial learning rate [default: 0.001]')
parser.add_argument('-epochs', type=int, default=400, help='number of epochs for train [default: 256]')
parser.add_argument('-batch-size', type=int, default=64, help='batch size for training [default: 64]')
parser.add_argument('-save-dir', type=str, default='snapshot', help='where to save the snapshot')
# model
parser.add_argument('-dropout', type=float, default=0.5, help='the probability for dropout [default: 0.5]')
parser.add_argument('-max-norm', type=float, default=3.0, help='l2 constraint of parameters [default: 3.0]')
parser.add_argument('-embed-dim', type=int, default=100, help='number of embedding dimension [default: 128]')
parser.add_argument('-kernel-num', type=int, default=100, help='number of each kind of kernel')
parser.add_argument('-kernel-sizes', type=str, default='3,4,5',
                    help='comma-separated kernel size to use for convolution')
parser.add_argument('-max-len', type=int, default=1000)
# device
parser.add_argument('-device', type=int, default=-1, help='device to use for iterate data, -1 mean cpu [default: -1]')
parser.add_argument('-snapshot', type=str, default=None, help='filename of model snapshot [default: None]')

args = parser.parse_args()

val_iter, vocab_size, class_num = util.load_data()

# update args and print
args.vocab_size = vocab_size
args.class_num = class_num
args.cuda = torch.cuda.is_available()
args.kernel_sizes = [int(k) for k in args.kernel_sizes.split(',')]
args.save_dir = os.path.join(args.save_dir, datetime.datetime.now().strftime('%Y-%m-%d_%H-%M-%S'))

print("\nParameters:")
for attr, value in sorted(args.__dict__.items()):
    print("\t{}={}".format(attr.upper(), value))

# model
cnn = TextCNN(args)
if args.snapshot is not None:
    print('\nLoading model from {}...'.format(args.snapshot))
    cnn.load_state_dict(torch.load(args.snapshot))

if args.cuda:
    torch.cuda.set_device(args.device)
    cnn = cnn.cuda()

optimizer = torch.optim.Adam(cnn.parameters(), lr=args.lr)
cnn.train()

for i in range(args.epochs):
    total_loss = 0.0
    for batch in val_iter:
        text, label = batch.text, batch.label
        if args.cuda:
            text, label = text.cuda(), label.cuda()

        optimizer.zero_grad()
        pred = cnn(text)
        loss = F.cross_entropy(pred, label)
        total_loss += loss
        loss.backward()
        optimizer.step()

    print('epoch {} loss : {}'.format(i, total_loss))
