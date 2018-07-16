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

    def load(self, path, change_opt=True):
        # 可加载指定路径的模型
        data = torch.load(path)
        if 'opt' in data:
            if change_opt:
                self.opt.parse(data['opt'], print_=False)
                self.opt.embedding_path = None
                self.__init__(self.opt)
            self.load_state_dict(data['d'])
        else:
            self.load_state_dict(data)
        return self.cuda()

    def save(self, name=None, new=False):
        prefix = 'checkpoints/' + self.model_name + '_'
        if name is None:
            name = time.strftime(prefix + '%m%d_%H:%M:%S.pth')
        path = prefix + name

        if new:
            data = {'opt': self.opt.state_dict(), 'd': self.state_dict()}
        else:
            data = self.state_dict()

        torch.save(data, path)
        return path

    def get_optimizer(self, lr1, lr2=0, weight_decay=0):
        embed_params = list(map(id, self.embedding.parameters()))
        base_params = filter(lambda p: id(p) not in embed_params, self.parameters())
        optimizer = torch.optim.Adam([
            {'params': self.embedding.parameters(), 'lr': lr2},
            {'params': base_params, 'lr': lr1, 'weight_decay': weight_decay}
        ])
        return optimizer

    # def get_optimizer(self, lr, lr_emb=0, weight_decay=0):
    #     ignored_params = list(map(id, self.embedding.parameters()))
    #     base_params = filter(lambda p: id(p) not in ignored_params,
    #                          self.parameters())
    #     if lr_emb is None:
    #         lr_emb = lr * 0.5
    #     optimizer = torch.optim.Adam([
    #         dict(params=base_params, weight_decay=weight_decay, lr=lr),
    #         {'params': self.embedding.parameters(), 'lr': lr_emb}
    #     ])
    #     return optimizer
