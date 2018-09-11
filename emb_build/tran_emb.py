import word2vec
import fire

paths = ['raw_word.txt', 'raw_article.txt']
sizes = [300]


def tran(path):
    model = word2vec.load(path)
    vocab, vectors = model.vocab, model.vectors
    print(path)
    print('shape of word embeddings : ')
    print(vectors.shape)

    new_path = path.split('.')[0] + '_.txt'
    print('Transform start....')
    f = open(new_path, 'w')
    for word, vector in zip(vocab, vectors):
        f.write(str(word) + ' ' + ' '.join(map(str, vector)) + '\n')
    print('Transform Complete!\n')


for path in paths:
    for size in sizes:
        emb_path = path.split('.')[0].split('_')[1] + '_' + str(size) + '.bin'
        word2vec.word2vec(path, emb_path, min_count=5, size=size, verbose=True)
        tran(emb_path)
