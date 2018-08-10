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

    # 数据集参数
    train_data_path = '/data/yujun/datasets/daguanbei_data/new_train_set.csv'
    val_data_path = '/data/yujun/datasets/daguanbei_data/val_set.csv'
    test_data_path = '/data/yujun/datasets/daguanbei_data/test_set.csv'
    # test，在自己的PC上小数据集调试代码用
    # train_data_path = 'dataset/val_set.csv'
    # test_data_path = 'dataset/val_set.csv'
    # val_data_path = 'dataset/val_set.csv'
    # word_embedding_path = 'emb/word_100.txt'
    # art_embedding_path = 'emb/article_100.txt'

    word_embedding_path = '/data/yujun/captain/emb/word_100.txt'  # 使用的预训练词向量
    art_embedding_path = '/data/yujun/captain/emb/article_100.txt'  # 使用的预训练字符向量
    embedding_dim = 100  # number of embedding dimension

    word_vocab_size = 10000  # 词库规模，配置中写的值没有意义，实际是预处理阶段获取
    art_vocab_size = 10000  # 字符规模
    label_size = 19  # 分类类别数
    batch_size = 64
    max_text_len = 1000  # 之后会处理成变长的，这里的设置没有意义
    max_article_len = 5000

    # 训练参数
    # cuda = True
    lr1 = 2e-3  # learning rate
    lr2 = 0  # embedding层的学习率
    min_lr = 1e-5  # 当学习率低于这个值时，就退出训练
    # lr_decay = 0.8 # 当一个epoch的损失开始上升时，lr ＝ lr*lr_decay
    # decay_every = 100 #每多少个batch查看val acc，并修改学习率
    weight_decay = 0  # 2e-5 # 权重衰减
    epochs = 50
    save_dir = 'snapshot/'  # where to save the snapshot
    id = 't1'
    device = 0

    # TextCNN
    kernel_num = 100  # number of each kind of kernel
    kernel_sizes = '3,4,5'  # kernel size to use for convolution
    dropout_rate = 0.5  # the probability for dropout

    # LSTM
    hidden_dim = 256
    lstm_dropout = 0  # 只有当lstm_layers > 1时，设置lstm_dropout才有意义
    lstm_layers = 1
    kmax_pooling = 2

    # RCNN

    # fastText
    linear_hidden_size = 2000  # 全连接层隐藏元数目

    def parse(self, kwargs, print_=True):
        '''
        根据字典kwargs 更新 config参数
        '''

        # 更新配置参数
        for k, v in kwargs.iteritems():
            if not hasattr(self, k):
                raise Exception("opt has not attribute <%s>" % k)
            setattr(self, k, v)

    def print_config(self):
        # 打印配置信息
        print('user config:')
        for k, v in self.__class__.__dict__.items():
            if not k.startswith('__') and k != 'parse' and k != 'print_config':
                print('    {} : {}'.format(k, getattr(self, k)))
