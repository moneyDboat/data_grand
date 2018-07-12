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
    model = 'TextCNN'  # 使用的模型，名字必须与models/__init__.py中的名字一致

    # 数据集参数
    train_data_path = '/data/yujun/datasets/daguanbei_data/new_train_set.csv'
    val_data_path = '/data/yujun/datasets/daguanbei_data/val_set.csv'
    test_data_path = '/data/yujun/datasets/daguanbei_data/test_set.csv'
    vocab_size = 10000  # 词库规模
    label_size = 19  # 分类类别数
    batch_size = 128  
    max_text_len = 1000
    embedding_dim = 100  # number of embedding dimension

    # 训练参数
    use_gpu = True
    lr = 0.001  # learning rate
    epochs = 100
    save_dir = 'snapshot/'  # where to save the snapshot
    device = 1

    # 模型参数
    dropout_rate = 0.5  # the probability for dropout

    # TextCNN
    kernel_num = 100  # number of each kind of kernel
    kernel_sizes = '3,4,5'  # kernel size to use for convolution

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
