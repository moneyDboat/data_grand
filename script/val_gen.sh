nohup python test_ensemble.py gen_test --model_path='result/bigru_attention_article_0.74793217006.pth' --device=0 &
nohup python test_ensemble.py gen_test --model_path='result/RCNN1_word_0.763518737493.pth' --device=1 &
# nohup python test_ensemble.py gen_test --model_path='result/LSTM_word_0.767802666207.pth' --device=1 &
# nohup python test_ensemble.py gen_test --model_path='result/GRU_word_0.773231866725.pth' --device=0 &
# nohup python test_ensemble.py gen_test --model_path='result/TextCNN_word_0.764570136444.pth' --device=2 &
#nohup python test_ensemble.py gen_test --model_path='result/GRU_article_0.734679227475.pth' --device=1 &
#nohup python test_ensemble.py gen_test --model-path='result/RCNN1_article_0.744593116562.pth' --device=10 &
#nohup python test_ensemble.py gen_test --model_path='result/n_RCNN1_word_0.767059778648.pth' --device=10 & 
