nohup python main.py main --model='bigru_attention' --device=0 --id='word' > att_word.log &
nohup python main.py main --model='bigru_attention' --device=1 --id='art' --max_text_len=3200 --text_type='article' > att_art.log &
nohup python main.py main --model='bigru_attention' --device=2 --id='word_n' --data_path='/data/yujun/captain/datasets1/' > n_att_word.log &
nohup python main.py main --model='bigru_attention' --device=3 --id='art_n' --max_text_len=3200 --text_type='article' --data_path='/data/yujun/captain/datasets1/' > n_att_art.log &

# nohup python main.py main --model='LSTM' --device=5 --id='word4' --emb=4 --data_path='/data/yujun/captain/datasets1/' > n_lstm_word4.log &
# nohup python main.py main --model='GRU' --device=6 --id='word4' --emb=4 --data_path='/data/yujun/captain/datasets1/' > n_gru_word4.log &
# nohup python main.py main --model='RCNN1' --device=4 --id='word4' --emb=4 --data_path='/data/yujun/captain/datasets1/' > n_rcnn_word4.log &
# nohup python main.py main --model='GRU' --device=8 --id='word41' --emb=4 --data_path='/data/yujun/captain/datasets1/' > n_gru_word41.log &
# nohup python main.py main --model='TextCNN' --device=10 --id='rev4' --emb=4 --data_path='/data/yujun/captain/datasets1/' > n_cnn_4.log &

# new datasets:
# nohup python main.py main --model='GRU' --device=1 --id='word_new' > n_gru_word.log &
# nohup python main.py main --model='GRU' --device=2 --id='art_new' --text_type='article' --max_text_len=3200 > n_gru_art.log &
# nohup python main.py main --model='TextCNN' --device=3 --id='word_new' > n_cnn_word.log &
# nohup python main.py main --model='TextCNN' --device=4 --id='art_new' --text_type='article' --max_text_len=3200 > n_cnn_art.log &
# nohup python main.py main --model='RCNN1' --device=6 --id='art_new' --text_type='article' --max_text_len=3200 > n_rcnn_art.log &
# nohup python main.py main --model='FastText' --device=9 --id='word_new' > n_fast_word.log &


# nohup python main.py main --model='TextCNN' --device=2 --id='word' > cnn_word &
# nohup python main.py main --model='GRU' --device=9 --id='e3' > gru_e3.log &
# nohup python main.py main --model='TextCNN' --device=2 --kernel_sizes='1 3 5 7 9' --id='e41' > cnn_e41.log &
# nohup python main.py main --model='TextCNN' --device=3 --kernel_sizes='2 4 6 8 10' --id='e42' > cnn_e42.log &
# nohup python main.py main --model='RCNN1' --device=4 --id='e4' > rcnn_e4.log &
# nohup python main.py main --model='RCNN1' --device=10 --id='e4_rev' --rev=True > rcnn_e4_rev.log &
# nohup python main.py main --model='RCNN1' --device=5 --data_path='/data/yujun/captain/datasets1/' --id='e4_new' > rcnn_e4_new.log &
# nohup python main.py main --model='GRU' --device=6 --max_text_len=3200 --text_type='article' --id='ar_e4' > gru_ar_e4.log &
# nohup python main.py main --model='TextCNN' --device=6 --id='art1' --text_type='article' --max_text_len=3200 > cnn_art1.log &
# nohup python main.py main --model='TextCNN' --device=9 --id='art2' --text_type='article' --max_text_len=3200 --kernel_sizes='4 5 6 7 8' > cnn_art2.log &
# nohup python main.py main --model='GRU' --device=10 --id='word_rev' --rev=True > gru_rev.log &
# nohup python main.py main --model='GRU' --device=2 --embedding_dim=100 --id='100' > gru_100.log &
# nohup python main.py main --model='GRU' --device=3 --embedding_dim=200 --id='200' > gru_200.log &
# nohup python main.py main --model='InCNN' --device=10 --id='' > in.log &
# nohup python main.py main --model='GRU' --max_epochs=0 --device=0 --id='del' --max_text_len=2000 --text_type='word' > gru_del.log &
# nohup python main.py main --model='LSTM' --model_path=None --device=1 --id='new1' --max_text_len=2000 --text_type='word' > lstm_new1.log &
# nohup python main.py main --model='GRU' --model_path='snapshot/GRU_base.pth' --device=0 --id='base' --max_text_len=2000 --text_type='word' --lr1=5e-5 --lr2=5e-5 > gru_tune.log &
# nohup python main.py main --model='GRU' --model_path=None --device=2 --id='100' --max_text_len=2000 --text_type='word' --linear_hidden_size=100 > gru_100.log &
# nohup python main.py main --model='GRU' --device=4 --id='emb4' > gru_emb4.log &
# nohup python main.py main --model='GRU' --device=1 --id='art2_l' --text_type='article' --max_text_len=4000 --kmax_pooling=2 > gru_art2_l.log &
# nohup python main.py main --model='GRU' --device=2 --id='art3_l' --text_type='article' --max_text_len=4000 --kmax_pooling=3 > gru_art3_l.log &
# nohup python main.py main --model='GRU' --device=5 --id='art3' --text_type='article' --max_text_len=3200 --kmax_pooling=3 > gru_art3.log &
# nohup python main.py main --model='GRU' --device=6 --id='art4' --text_type='article' --max_text_len=3200 --kmax_pooling=4 > gru_art4.log &
# nohup python main.py main --model='FastText' --device=1 --id='word1' --batch_size=128  > fast_word1.log &
# nohup python main.py main --model='FastText' --device=3 --id='art1' --batch_size=128 --max_text_len=4000 --text_type='article'> fast_art1.log &
# nohup python main.py main --model='GRU' --device=9 --id='art_lay2' --text_type='article' --max_text_len=3200 -lstm_layers=2 --lstm_dropout=0.5 > gru_art_lay2.log &
# nohup python main.py main --model='GRU' --device=5 --id='art' --text_type='article' --max_text_len=5000 > gru_art.log &
# nohup python main.py main --model='LSTM' --device=6 --id='art' --text_type='article' --max_text_len=5000 > lstm_art.log &
# nohup python main.py main --model='LSTM' --device=7 --id='base' --text_type='word' --max_text_len=2000 > lstm_base.log &
# nohup python main.py main --model='TextCNN' --device=8 --id='new' > cnn_new.log &
# nohup python main.py main --model='TextCNN' --device=9 --id='art' --text_type='article' --max_text_len=5000 > cnn_art.log &
# nohup python main.py main --model='LSTM' --device=3 --id='100' --linear_hidden_size=100 > lstm_100.log &
# nohup python main.py main --model='GRU' --device=1 --id='k1' --kmax_pooling=1 > gru_k1.log &
# nohup python main.py main --model='GRU' --device=2 --id='k3' --kmax_pooling=3 > gru_k3.log &
# nohup python main.py main --model='GRU' --device=6 --id='k4' --kmax_pooling=4 > gru_k4.log &
# nohup python main.py main --model='GRU' --device=4 --hidden_dim=400 --id='h400' > gru_h400.log &
# nohup python main.py main --model='GRU' --device=5 --kmax_pooling=4 --max_text_len=4000 --text_type='article' --id='art' > gru_art.log &
# nohup python main.py main --model='GRU' --device=3 --batch_size=32 --lstm_dropout=0.5 --lstm_layers=2 --id='lay2' > gru_lay2.log &
# nohup python main.py main --model='RCNN1' --device=0 --kmax_pooling=1 --rcnn_kernel=200 --id=200 > rcnn_200.log &
# nohup python main.py main --model='RCNN1' --device=5 --id='512' > rcnn_512.log &
# nohup python main.py main --model='RCNN1' --device=4 --id='art' --text_type='article' --max_text_len=3200 > rcnn_art.log &
# nohup python main.py main --model='RCNN' --device=7 --id='art' --text_type='article' --max_text_len=4000 > rcnn_art.log &
# nohup python main.py main --model='TextCNN' --device=8 --id='base' > cnn_base.log &
# nohup python main.py main --model='TextCNN' --device=9 --id='art' --text_type='article' --max_text_len=4000 > cnn_art.log &
# nohup python main.py main --model='GRU' --device=4 --aug=True --id='aug' > gru_aug.log &
# nohup python main.py main --model='GRU' --device=2 --aug=True --max_text_len=4000 --text_type='article' --id='art_aug' > gru_art_aug.log &
# nohup python main.py main --model='GRU' --device=2 --id='seed1' > gru_seed1.log &
# nohup python main.py main --model='GRU' --device=3 --id='seed2' > gru_seed2.log &

