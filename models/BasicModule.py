# -*- coding: utf-8 -*-

"""
@Author  : captain
@time    : 18-7-11 上午10:09
@ide     : PyCharm  
"""
import torch
import time


class BasicModule(torch.nn.Module):
    # 封装了nn.Module， 主要是提供了save和load两个方法

    def __init__(self):
        super(BasicModule, self).__init__()
        self.model_name = str(type(self))  # 模型的默认名字

    def load(self, path):
        # 可加载指定路径的模型
        self.load_state_dict(torch.load(path))

    def save(self, name=None):
        if name is None:
            prefix = 'checkpoints/' + self.model_name + '_'
            name = time.strftime(prefix + '%m%d_%H:%M:%S.pth')
        torch.save(self.state_dict(), name)
        return name
