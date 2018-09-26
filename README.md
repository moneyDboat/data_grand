# 2018“达观杯”文本智能处理挑战赛
## Top 10 “万里阳光号”解决方案
更详细的比赛经验分享见知乎专栏文章https://zhuanlan.zhihu.com/p/45391378  
比赛详情见 [达观杯文本智能处理挑战赛](http://www.dcjingsai.com/common/cmpt/%E2%80%9C%E8%BE%BE%E8%A7%82%E6%9D%AF%E2%80%9D%E6%96%87%E6%9C%AC%E6%99%BA%E8%83%BD%E5%A4%84%E7%90%86%E6%8C%91%E6%88%98%E8%B5%9B_%E7%AB%9E%E8%B5%9B%E4%BF%A1%E6%81%AF.html)  
最终排名见 [排行榜](
http://www.dcjingsai.com/common/cmpt/%E2%80%9C%E8%BE%BE%E8%A7%82%E6%9D%AF%E2%80%9D%E6%96%87%E6%9C%AC%E6%99%BA%E8%83%BD%E5%A4%84%E7%90%86%E6%8C%91%E6%88%98%E8%B5%9B_%E6%8E%92%E8%A1%8C%E6%A6%9C.html)
### 环境配置
代码基于Pytorch，版本为0.4.1，Python版本为3.6。需安装：
- pytorch
- torchtext
- word2vec
- pandas
- sklearn
- numpy
- fire
### 文件说明
```
emb_build/: word2vec训练词／字向量
models/: 深度学习模型
result/: 生成测试集提交csv
val_result/: 模型融合
script/: 脚本文件
config.py: 模型配置
fine_tune.py: 模型fine tune
gen_result.py: 生成模型在测试集上的预测概率结果
test_ensemble.py: 生成模型在验证集上的预测概率结果
data.py: 数据预处理
main.py: 模型训练
```

### 词／字向量训练
词／字向量训练使用word2vec包，见[word2vec](https://github.com/danielfrg/word2vec)。分别使用训练集和测试集中所有词文本和字文本训练词向量和字向量，向量维度设置为300维。  
代码见emb_build文件夹下read_csv.py和tran_emb.py，注意修改代码中训练集和测试集的文件路径，依次运行即可得到词／字向量文件word_300.txt和article_300.txt。
```
python read_csv.py
python tran_emb.py
```
### 文本预处理
将比赛提供的训练数据按9:1的比例，划分为训练集和验证集。
```
# util/
python split_val.py
```
word文本平均长度为717，按照覆盖95%样本的标准，取截断长度为2000；article文本平均长度为1177，按同样的标准取截断长度为3200。  
从csv文件中提取文本数据，使用torchtext进行文本预处理，并进一步构造batch，这部分代码见data.py的类GrandDataset和方法load_data()。
### 训练模型
主要用到了五个模型

- TextCNN: models/TextCNN.py
- GRU: models/GRU.py
- RCNN: models/RCNN.py
- FastText: models/FastText.py
- Attention: models/bigru_att.py  

分别训练两个对应的word模型和article模型，注意文本数据和词/字向量的存放路径。 
注意模型配置位于 config.py，模型训练代码位于main.py中的main方法，命令示例如下（也可见script/run.sh）:  
```
python main.py main --model='LSTM' --device=5 --id='word4'
python main.py main --model='GRU' --device=6 --id='word4'
python main.py main --model='RCNN1' --device=4 --id='word4'
python main.py main --model='GRU' --device=8 --id='word41'
python main.py main --model='TextCNN' --device=10 --id='rev4'
```
### 训练策略
- 优化器选用torch.optim.Adam，初始学习率设置为1e-3（注意embedding层的学习率另外设置）。
- 先固定embedding层的参数，每个epoch后计算模型在验证集上的f1值，如果开始下降（一般为2-3个epoch之后），将embedding层的学习率设为2e-4。
- 每个epoch后计算验证集上的f1值，如上升则保存当前模型，位于文件夹snapshot/；如果下降则从snapshot中加载当前最好的模型，并降低学习率。
- 如果学习率低于config.py中设置的最低学习率，则终止训练。如果设置的最低学习率为1e-5，一般15个epoch左右后训练终止。


### 模型融合
尝试了多种模型融合方法后，只采用了最简单但有效的模型融合方法－概率等权重融合，代码见val_result/ensemble.py，修改代码中的
```
model_list = ['GRU_word_rev_0.721350215541.pth.npy', 'GRU_word_0.771128871335.pth.npy',  'RCNN1_word_0.769018920031.pth.npy', 'LSTM_word_0.768979776301.pth.npy', 'TextCNN_word_0.760456622816.pth.npy', 'FastText_word_0.75425891649.pth.npy', 'GRU_article_0.747660923499.pth.npy', 'TextCNN_article_0.735192435177.pth.npy']
```
运行ensemble.py得到融合模型在验证集上的f1值，根据f1值选取参与融合的模型。

### 生成提交csv
将上一步模型融合选取的模型在测试集上生成预测概率结果，进行等权重相加，将概率最大的类别作为预测类别生成提交csv，代码见result/ensemble.py。  

更详细的比赛经验分享见知乎专栏文章https://zhuanlan.zhihu.com/p/45391378  
