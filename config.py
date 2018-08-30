# -*- coding: utf-8 -*-

"""
@Author  : captain
@time    : 18-7-11 上午12:33
@ide     : PyCharm  
"""


class DefaultConfig(object):
    '''
    列出所有的参数，只根据模型需要获取参数
    '''
    env = 'default'  # visdom环境
    model = 'LSTM'  # 使用的模型，名字必须与models/__init__.py中的名字一致
    model_path = None  # 如果有就加载
    result_path = ''
    save_dir = 'snapshot/'  # where to save the snapshot
    id = 't1'
    device = 0
    boost = False  ## 是否使用adboost
    finetune = False  ## 是否对训练完成的模型进行finetune

    # 数据集参数
    data_path = '/data/yujun/captain/datasets/'
    text_type = 'word'  # or 'article'  # 决定数据集位置
    # train_data_path = '/data/yujun/datasets/daguanbei_data/new_split/new_train_set.csv'
    # val_data_path = '/data/yujun/datasets/daguanbei_data/new_split/val_set.csv'
    # test_data_path = '/data/yujun/datasets/daguanbei_data/test_set.csv'
    embedding_path = '/data/yujun/captain/emb'  # 使用的预训练词向量
    embedding_dim = 300  # number of embedding dimension

    # 在自己的PC上小数据集调试代码用
    # train_data_path = 'D:/git/dataset/val_set.csv'
    # test_data_path = 'D:/git/dataset/val_set.csv'
    # val_data_path = 'D:/git/dataset/val_set.csv'
    # embedding_path = 'D:/git/emb'  # 预训练词向量的位置
    # embedding_dim = 100

    batch_size = 64
    vocab_size = 10000  # 词库规模，配置中写的值没有意义，实际是预处理阶段获取
    label_size = 19  # 分类类别数
    max_text_len = 2000  # 之后会处理成变长的，这里的设置没有意义

    # 训练参数
    lr1 = 1e-3  # learning rate
    lr2 = 0  # embedding层的学习率
    min_lr = 1e-5  # 当学习率低于这个nvi值时，就退出训练
    lr_decay = 0.8  # 当一个epoch的损失开始上升时，lr ＝ lr*lr_decay
    decay_every = 10000  # 每多少个batch  查看val acc，并修改学习率
    weight_decay = 0  # 2e-5 # 权重衰减
    max_epochs = 50
    cuda = True

    # 模型通用
    linear_hidden_size = 500  # 原来为2000(500)，之后还需要修改，感觉数值有点大

    # TextCNN
    kernel_num = 200  # number of each kind of kernel
    kernel_sizes = '3,4,5'  # kernel size to use for convolution
    dropout_rate = 0.5  # the probability for dropout

    # LSTM
    hidden_dim = 256
    lstm_dropout = 0.5  # 只有当lstm_layers > 1时，设置lstm_dropout才有意义
    lstm_layers = 1
    kmax_pooling = 2

    # RCNN

    def parse(self, kwargs):
        '''
        根据字典kwargs 更新 config参数
        '''

        # 更新配置参数
        for k, v in kwargs.items():
            if not hasattr(self, k):
                raise Exception("Warning: config has not attribute <%s>" % k)
            setattr(self, k, v)

    def print_config(self):
        # 打印配置信息
        print('user config:')
        for k, v in self.__class__.__dict__.items():
            if not k.startswith('__') and k != 'parse' and k != 'print_config':
                print('    {} : {}'.format(k, getattr(self, k)))
