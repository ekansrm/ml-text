import numpy as np

from keras.models import Sequential
from keras.layers import Dense, Embedding
from keras.layers import GlobalAveragePooling1D
from keras.preprocessing.sequence import pad_sequences
from model.preprocess import X_train_word_ids, X_test_word_ids, y_train, y_test, num_labels, vocab


# 模型结构：词嵌入(n-gram)-最大化池化-全连接
# 生成n-gram组合的词(以3为例)
ngram = 3


# 将n-gram词加入到词表
def create_ngram(sent, ngram_value):
    return set(zip(*[sent[i:] for i in range(ngram_value)]))


X_train_padded_seqs = pad_sequences(X_train_word_ids, maxlen=64)
X_test_padded_seqs = pad_sequences(X_test_word_ids, maxlen=64)

ngram_set = set()
for sentence in X_train_padded_seqs:
    for i in range(2, ngram + 1):
        set_of_ngram = create_ngram(sentence, i)
        ngram_set.update(set_of_ngram)

# 给n-gram词汇编码
start_index = len(vocab) + 2
token_indice = {v: k + start_index for k, v in enumerate(ngram_set)}  # 给n-gram词汇编码
indice_token = {token_indice[k]: k for k in token_indice}
max_features = np.max(list(indice_token.keys())) + 1


# 将n-gram词加入到输入文本的末端
def add_ngram(sequences, token_indice, ngram_range):
    new_sequences = []
    for sent in sequences:
        new_list = sent[:]
        for i in range(len(new_list) - ngram_range + 1):
            for ngram_value in range(2, ngram_range + 1):
                ngram = tuple(new_list[i:i + ngram_value])
                if ngram in token_indice:
                    new_list.append(token_indice[ngram])
        new_sequences.append(new_list)
    return new_sequences


x_train = add_ngram(X_train_word_ids, token_indice, ngram)
x_test = add_ngram(X_test_word_ids, token_indice, ngram)
x_train = pad_sequences(x_train, maxlen=25)
x_test = pad_sequences(x_test, maxlen=25)

model = Sequential()
model.add(Embedding(max_features, 300, input_length=25))
model.add(GlobalAveragePooling1D())
model.add(Dense(num_labels, activation='softmax'))

model.compile(
    optimizer='adam',
    loss='categorical_crossentropy',
    # loss_weights={'y': 1., 'y_aux': 0.7},
    metrics=['accuracy']
)

model.fit(x=x_train, y=y_train, batch_size=128, validation_split=0.26, shuffle=True, epochs=10, verbose=1)
e = model.evaluate(x=x_test, y=y_test, verbose=1)
print(e)
