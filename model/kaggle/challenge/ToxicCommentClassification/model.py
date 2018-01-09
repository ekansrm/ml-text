
class EmbeddedLSTMWithAttention(object):

    class Config(object):
        def __init__(self):
            self.x_dim = None
            self.dropout = 0.2
            self.lstm_units = 32
            self.embedding_word_number = None
            self.embedding_vector_length = None
            self.dense = [16]

    def __init__(self):
        self._config = EmbeddedLSTMWithAttention.Config()
        self._model = None

    def build(self):
        assert self._config is not None, "模型未配置, 不能初始化"

        config = self._config

        # 整型输入进行 embedding lookup
        x = Input(shape=(config.x_int_dim,), dtype='int32', name='x')
        x_vec = Embedding(
            name='embedding',
            embeddings_initializer='lecun_uniform',
            input_dim=config.embedding_word_number,
            output_dim=config.embedding_vector_length,
            input_length=config.x_int_dim
        )(x)

        x_lstm_embedded_out = LSTM(units=config.lstm_units, name='lstm-embedded')(x_vec)

        print(x_lstm_embedded_out.shape)

        if config.dropout > 0:
            x_lstm_embedded_out = \
                Dropout(config.dropout)(x_lstm_embedded_out)

        x_dense = x_lstm_embedded_out
        for i, dim in enumerate(config.dense):
            x_dense = Dense(dim, activation='sigmoid', name='dense_'+str(i))(x_dense)
            if config.dropout > 0:
                x_dense = Dropout(config.dropout)(x_dense)

        # 输出
        y = Dense(1, activation='sigmoid', name='y')(x_dense)

        self._model = Model(inputs=[x], outputs=[y])

    @property
    def model(self) -> Model:
        return self._model

    @model.setter
    def model(self, path):
        self._model = load_model(path)

    @property
    def config(self):
        return self._config

    @config.setter
    def config(self, config: Config):
        self._config = config


if __name__ == '__main__':
    config = EmbeddedLSTM.Config()
    config.x_int_dim = 6
    config.x_float_dim = 3
    config.embedding_word_number = 600
    config.embedding_vector_length = 32

    lstm = EmbeddedLSTM()
    lstm.config = config
    lstm.build()
    lstm.model.summary()






embedding_layer = Embedding(nb_words,
                            EMBEDDING_DIM,
                            weights=[embedding_matrix],
                            input_length=MAX_SEQUENCE_LENGTH,
                            trainable=False)

lstm_layer = LSTM(num_lstm, dropout=rate_drop_lstm, recurrent_dropout=rate_drop_lstm, return_sequences=True)

comment_input = Input(shape=(MAX_SEQUENCE_LENGTH,), dtype='int32')
embedded_sequences = embedding_layer(comment_input)
x = lstm_layer(embedded_sequences)
x = Dropout(rate_drop_dense)(x)
merged = Attention(MAX_SEQUENCE_LENGTH)(x)
merged = Dense(num_dense, activation=act)(merged)
merged = Dropout(rate_drop_dense)(merged)
merged = BatchNormalization()(merged)
preds = Dense(6, activation='sigmoid')(merged)


class

    embedding_layer = Embedding(nb_words,
                                EMBEDDING_DIM,
                                weights=[embedding_matrix],
                                input_length=MAX_SEQUENCE_LENGTH,
                                trainable=False)

lstm_layer = LSTM(num_lstm, dropout=rate_drop_lstm, recurrent_dropout=rate_drop_lstm, return_sequences=True)

comment_input = Input(shape=(MAX_SEQUENCE_LENGTH,), dtype='int32')
embedded_sequences = embedding_layer(comment_input)
x = lstm_layer(embedded_sequences)
x = Dropout(rate_drop_dense)(x)
merged = Attention(MAX_SEQUENCE_LENGTH)(x)
merged = Dense(num_dense, activation=act)(merged)
merged = Dropout(rate_drop_dense)(merged)
merged = BatchNormalization()(merged)
preds = Dense(6, activation='sigmoid')(merged)
