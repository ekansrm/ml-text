from app.kaggle.challenge.ToxicCommentClassification import DAT_TRAIN, DAT_TEST, DAT
from app.kaggle.challenge.ToxicCommentClassification import VAR

import pandas as pd
import numpy as np
from gensim.models import Word2Vec
import multiprocessing
from collections import defaultdict
import json

MAX_SEQUENCE_LENGTH = 150
MAX_NB_WORDS = 100000
EMBEDDING_DIM = 300

PATH_TOKENIZER = VAR('tokenizer')
PATH_EMBEDDING = VAR('embedding')


class Tokenizer(object):
    def __init__(self):
        self.tokenizer = defaultdict(lambda: 0)
        self.tokenizer.update(json.load(open(PATH_TOKENIZER, 'r')))
        # tokens = list(set([self.tokenizer[x] for x in self.tokenizer]))
        # token_min = min(tokens)
        # token_max = max(tokens)

    def tokenize(self, word_list):
        return list(map(lambda word: self.tokenizer[word], word_list))


if __name__ == '__main__':

    train = pd.read_csv(DAT_TRAIN)
    test = pd.read_csv(DAT_TEST)

    data = pd.concat([train, test])
    data = data[['comment_text']]

    permutation = np.random.permutation(data.shape[0])
    data = data.iloc[permutation]

    data['comment_text'] = data['comment_text'].fillna('NA').apply(lambda x: x.split(' '))

    model_word2vec = Word2Vec(data['comment_text'], size=64, window=5, min_count=5, workers=multiprocessing.cpu_count())
    vocab = dict([(k, v.index + 1) for k, v in model_word2vec.wv.vocab.items()])
    weights = model_word2vec.wv.syn0

    print(model_word2vec['use'])
    print(weights[vocab['use']])

    json.dump(weights, open(PATH_EMBEDDING, 'w'))
    json.dump(vocab, open(PATH_TOKENIZER, 'w'))
