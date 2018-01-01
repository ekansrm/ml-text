
from keras.preprocessing.sequence import pad_sequences

from model.preprocess import X_train_word_ids, X_test_word_ids, y_train, y_test, num_labels, vocab
# One-hot
# x_train = tokenizer.sequences_to_matrix(X_train_word_ids, mode='binary')
# x_test = tokenizer.sequences_to_matrix(X_test_word_ids, mode='binary')


# 序列模式
x_train = pad_sequences(X_train_word_ids, maxlen=20)
x_test = pad_sequences(X_test_word_ids, maxlen=20)

from keras.layers.merge import concatenate
from keras.models import Model
from keras.layers import Dense, Embedding, Input
from keras.layers import Convolution1D, Flatten, Dropout, MaxPool1D

# GLOVE_DIR = "D:\python\kaggle\game_reviews\glove"
# embeddings_index = {}
# f = open(os.path.join(GLOVE_DIR, 'glove.6B.200d.txt'), encoding = 'utf-8')
# for line in f:
#     values = line.split()
#     word = values[0]
#     coefs = np.asarray(values[1:], dtype='float32')
#     embeddings_index[word] = coefs
# f.close()
#
# embedding_matrix = np.zeros((len(vocab) + 1, 200))
# for word, i in vocab.items():
#     embedding_vector = embeddings_index.get(word)
#     if embedding_vector is not None:
#         embedding_matrix[i] = embedding_vector

# 模型结构：词嵌入-卷积池化*3-拼接-全连接-dropout-全连接
main_input = Input(shape=(20,), dtype='float64')
# 词嵌入（使用预训练的词向量）
# embedder = Embedding(len(vocab) + 1, 300, input_length = 20, weights = [embedding_matrix], trainable = False)
embedder = Embedding(len(vocab) + 1, 300, input_length=20, trainable=False)
embed = embedder(main_input)
# 词窗大小分别为3,4,5
cnn1 = Convolution1D(256, 3, padding='same', strides=1, activation='relu')(embed)
cnn1 = MaxPool1D(pool_size=4)(cnn1)
cnn2 = Convolution1D(256, 4, padding='same', strides=1, activation='relu')(embed)
cnn2 = MaxPool1D(pool_size=4)(cnn2)
cnn3 = Convolution1D(256, 5, padding='same', strides=1, activation='relu')(embed)
cnn3 = MaxPool1D(pool_size=4)(cnn3)
# 合并三个模型的输出向量
cnn = concatenate([cnn1, cnn2, cnn3], axis=-1)
flat = Flatten()(cnn)
drop = Dropout(0.2)(flat)
main_output = Dense(num_labels, activation='softmax')(drop)
model = Model(inputs=main_input, outputs=main_output)

model.compile(
    optimizer='adam',
    loss='categorical_crossentropy',
    # loss_weights={'y': 1., 'y_aux': 0.7},
    metrics=['accuracy']
)

model.fit(x=x_train, y=y_train, batch_size=128, validation_split=0.26, shuffle=True, epochs=10, verbose=1)
e = model.evaluate(x=x_test, y=y_test, verbose=1)
print(e)
