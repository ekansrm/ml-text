from keras.models import Sequential
from keras.layers import Dense, Embedding
from keras.layers import Convolution1D, Flatten, Dropout, MaxPool1D
from keras.layers import BatchNormalization
from keras.preprocessing.sequence import pad_sequences

from model.preprocess import X_train_word_ids, X_test_word_ids, y_train, y_test, num_labels, vocab

X_train_padded_seqs = pad_sequences(X_train_word_ids, maxlen=20)
X_test_padded_seqs = pad_sequences(X_test_word_ids, maxlen=20)

model = Sequential()

model.add(Embedding(len(vocab) + 1, 300, input_length=20))
model.add(Convolution1D(256, 3, padding='same'))
model.add(MaxPool1D(3, 3, padding='same'))
model.add(Convolution1D(128, 3, padding='same'))
model.add(MaxPool1D(3, 3, padding='same'))
model.add(Convolution1D(64, 3, padding='same'))
model.add(Flatten())
model.add(Dropout(0.1))
model.add(BatchNormalization())  # (批)规范化层
model.add(Dense(256, activation='relu'))
model.add(Dropout(0.1))
model.add(Dense(num_labels, activation='softmax'))


model.compile(loss='categorical_crossentropy',
              optimizer='adam',
              metrics=['accuracy'])

model.fit(X_train_padded_seqs, y_train,
          batch_size=32,
          epochs=15,
          validation_data=(X_test_padded_seqs, y_test), verbose=True)

