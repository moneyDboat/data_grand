import pandas as pd

print('loading datasets......')
train_data = pd.read_csv('/data/yujun/datasets/daguanbei_data/train_set.csv')
test_data = pd.read_csv('/data/yujun/datasets/daguanbei_data/test_set.csv')

print('{} lines in train datasets'.format(len(train_data)))
print('{} lines in test datasets'.format(len(test_data)))

print('making raw_word.txt......')
with open('raw_word.txt', 'w') as f:
    f.writelines([text + '\n' for text in train_data['word_seg']])
    f.writelines([text + '\n' for text in test_data['word_seg']])

print('making raw_article.txt......')
with open('raw_article.txt', 'w') as f:
    f.writelines([text + '\n' for text in train_data['article']])
    f.writelines([text + '\n' for text in test_data['article']])

