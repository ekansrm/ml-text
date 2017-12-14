import pandas as pd  # 导入Pandas
import numpy as np  # 导入Numpy
import jieba  # 导入结巴分词

from gensim.models import Word2Vec

PATH_MODEL_WORD_2_VEC = 'word2vec.64.model'
model_word2vec = Word2Vec.load(PATH_MODEL_WORD_2_VEC)

weights = model_word2vec.wv.syn0

#获得词库
vocab = dict([(k, v.index) for k, v in model_word2vec.wv.vocab.items()])


def to_ids(word_list):
    def word_to_id(word):
        id = vocab.get(word)
        if id is None:
            id = 0
        return id

    x = list(map(word_to_id, word_list))
    return np.array(x)


string = '测试文本'

words = jieba.cut(string)
ids = to_ids(words)

print(ids)

from keras.preprocessing import sequence
from keras.models import Sequential
from keras.layers.core import Dense, Dropout, Activation
from keras.layers.embeddings import Embedding
from keras.layers.recurrent import LSTM, GRU


neg = pd.read_excel('../data/taobao-comment/neg.xls', header=None, index=None)
pos = pd.read_excel('../data/taobao-comment/pos.xls', header=None, index=None)  # 读取训练语料完毕
pos['mark'] = 1
neg['mark'] = 0  # 给训练语料贴上标签
pn = pd.concat([pos, neg], ignore_index=True)  # 合并语料
neglen = len(neg)
poslen = len(pos)  # 计算语料数目

cw = lambda x: list(jieba.cut(x))  # 定义分词函数
pn['words'] = pn[0].apply(cw)

pn['sent'] = pn['words'].apply(to_ids)  # 速度太慢

maxlen = 50

print("Pad sequences (samples x time)")
pn['sent'] = list(sequence.pad_sequences(pn['sent'], maxlen=maxlen))

x = np.array(list(pn['sent']))[::2]  # 训练集
y = np.array(list(pn['mark']))[::2]
xt = np.array(list(pn['sent']))[1::2]  # 测试集
yt = np.array(list(pn['mark']))[1::2]
xa = np.array(list(pn['sent']))  # 全集
ya = np.array(list(pn['mark']))

print('Build model...')
model = Sequential()
model.add(Embedding(name='embedding',
                    input_dim=weights.shape[0], output_dim=weights.shape[1], input_length=maxlen,
                    weights=[weights],
                    trainable=False))

model.add(LSTM(256))  # try using a GRU instead, for fun
model.add(Dropout(0.5))
model.add(Dense(1))
model.add(Activation('sigmoid'))

model.compile(loss='binary_crossentropy', optimizer='adam')

model.fit(xa, ya, batch_size=32, nb_epoch=100, validation_split=0.2)  # 训练时间为若干个小时

model.save("sentiment-analysis-lstm-keras.hdf5")

e = model.evaluate(xt, yt)
print('Test accuracy:', e)
