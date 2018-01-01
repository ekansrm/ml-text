import pandas as pd  # 导入Pandas
import numpy as np  # 导入Numpy
import jieba  # 导入结巴分词
import json
from collections import defaultdict

from keras.preprocessing import sequence
from keras.models import Sequential
from keras.layers.core import Dense, Dropout, Activation
from keras.layers.embeddings import Embedding
from keras.layers.recurrent import LSTM, GRU
from keras.optimizers import SGD
from keras.callbacks import LearningRateScheduler
from keras.callbacks import EarlyStopping
from keras.callbacks import ModelCheckpoint

maxlen = 50

PATH_PN = '../data/taobao-comment/pn.csv'
PATH_TOKENIZER = '../data/taobao-comment/tokenizer'

########################################################################################################################
# 加载 TOKENIZER
tokenizer = defaultdict(lambda: 0)
tokenizer.update(json.load(open(PATH_TOKENIZER, 'r')))

tokens = list(set([tokenizer[x] for x in tokenizer]))
token_min = min(tokens)
token_max = max(tokens)


def tokenize(word_list):
    return list(map(lambda word: tokenizer[word], word_list))

########################################################################################################################


########################################################################################################################

# 读取训练样本
pn = pd.read_csv(PATH_PN)

# 分词
pn['words'] = pn['text'].apply(lambda line: list(jieba.cut(line)))

# tokenizer
pn['vec'] = pn['words'].apply(tokenize)  # 速度太慢

# padding
pn['vec'] = list(sequence.pad_sequences(pn['vec'], maxlen=maxlen))

# 训练数据
x = np.array(list(pn['vec']))
y = np.array(list(pn['mark']))

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

########################################################################################################################
# 模型训练

model = Sequential()
model.add(Embedding(name='embedding',
                    input_dim=token_max+1, output_dim=256, input_length=maxlen,
                    ))

model.add(LSTM(256))  # try using a GRU instead, for fun
model.add(Dropout(0.9))
model.add(Dense(1))
model.add(Activation('sigmoid'))
MODEL_LSTM_PATH = 'sentiment-analysis-lstm'

model_checkpoint_better_path = MODEL_LSTM_PATH + '.checkpoint' \
                                                 '.epoch-{epoch:02d}' \
                                                 '.val_loss-{val_loss:.6f}' \
                                                 '.val_acc-{val_acc:.6f}'
model_checkpoint_best_path = MODEL_LSTM_PATH + '.checkpoint.best'

sgd = SGD(lr=0.0, momentum=0.9, decay=0.0, nesterov=False)

# def step_decay(epoch):
#     initial_lrate = 0.1
#     drop = 0.5
#     epochs_drop = 10.0
#     lrate = initial_lrate * math.pow(drop, math.floor((1 + epoch) / epochs_drop))
#     return lrate
# lrate = LearningRateScheduler(step_decay)

early_stopping = EarlyStopping(monitor='val_loss', min_delta=0.0001, mode='min', patience=5, verbose=1)

checkpoint_better = ModelCheckpoint(
    model_checkpoint_better_path, save_best_only=True, monitor='val_acc',  mode='max', verbose=1)

checkpoint_best = ModelCheckpoint(
    model_checkpoint_best_path, save_best_only=True, monitor='val_acc',  mode='max', verbose=1)

model.compile(loss='binary_crossentropy', optimizer='adam',  metrics=['accuracy'])

model.fit(x, y, batch_size=32, nb_epoch=100, validation_split=0.2, shuffle=True,
          callbacks=[checkpoint_better, checkpoint_best, early_stopping]
          )  # 训练时间为若干个小时

e = model.evaluate(x, y)
print('Test accuracy:', e)
