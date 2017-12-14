import pandas as pd  # 导入Pandas
import numpy as np  # 导入Numpy
import jieba  # 导入结巴分词

from keras.preprocessing import sequence
from keras.models import Sequential
from keras.layers.core import Dense, Dropout, Activation
from keras.layers.embeddings import Embedding
from keras.layers.recurrent import LSTM, GRU

neg = pd.read_excel('../data/taobao-comment/neg.xls', header=None, index=None)
pos = pd.read_excel('../data/taobao-comment/pos.xls', header=None, index=None)  # 读取训练语料完毕

neg['text'] = neg[0]
pos['text'] = pos[0]
pos['mark'] = 1
neg['mark'] = 0  # 给训练语料贴上标签

pn = pd.concat([pos, neg], ignore_index=True)  # 合并语料
pn = pn[['text', 'mark']]
permutation = np.random.permutation(pn.shape[0])
pn = pn.iloc[permutation]
print(pn.head(5))
pn.to_csv('../data/taobao-comment/pn.csv', index=False)

cw = lambda x: list(jieba.cut(x))  # 定义分词函数
pn['words'] = pn['text'].apply(cw)

w = []  # 将所有词语整合在一起
for i in pn:
    w.extend(i)

dict = pd.DataFrame(pd.Series(w).value_counts())  # 统计词的出现次数

dict['id'] = list(range(1, len(dict) + 1))

get_sent = lambda x: list(dict['id'][x])
pn['vec'] = pn['words'].apply(get_sent)  # 速度太慢

maxlen = 50

print("Pad sequences (samples x time)")
pn['vec'] = list(sequence.pad_sequences(pn['sent'], maxlen=maxlen))

xa = np.array(list(pn['vec']))  # 全集
ya = np.array(list(pn['mark']))

# print('Build model...')
# model = Sequential()
# model.add(Embedding(len(dict) + 1, 256))
# model.add(LSTM(256))  # try using a GRU instead, for fun
# model.add(Dropout(0.5))
# model.add(Dense(1))
# model.add(Activation('sigmoid'))

# model.compile(loss='binary_crossentropy', optimizer='adam')
#
# model.fit(xa, ya, batch_size=16, nb_epoch=10)  # 训练时间为若干个小时
#
# model.save("sentiment-analysis-lstm-keras.hdf5")
#
# e = model.evaluate(xt, yt)
# print('Test accuracy:', e)
