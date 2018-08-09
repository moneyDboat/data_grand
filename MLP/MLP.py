# -*- coding: utf-8 -*-

"""
@Author  : captain
@time    : 18-7-28 上午10:57
@ide     : PyCharm  
"""
import torch
import torch.nn as nn
import torch.nn.functional as F


class MLP(nn.Module):
    def __init__(self):
        super(MLP, self).__init__()
        self.fc1 = nn.Linear(4027880, 10240)
        self.fc2 = nn.Linear(10240, 512)
        self.fc3 = nn.Linear(512, 19)
        self.dropout = nn.Dropout(0.2)

    def forward(self, input):
        out = F.relu(self.fc1(input))
        out = F.relu(self.fc2(self.dropout(out)))
        out = F.relu(self.fc3(self.dropout(out)))

        return F.softmax(out)
