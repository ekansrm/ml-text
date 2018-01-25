from keras.models import Model
from keras.layers import Input
from keras.layers import Dense
from keras.layers import LSTM
from keras.layers import Dropout
from keras.layers.embeddings import Embedding


class EmbeddingLSTM(object):
    """
    用函数式的方法实现模型工厂
    """
    def __init__(self,
                 name='EmbeddingLSTM',
                 input_length=None,

                 embedding_word_number=None,
                 embedding_vector_length=None,
                 embedding_dropout=None,
                 embedding_kwargs=None,

                 lstm_units=None,
                 lstm_dropout=None,
                 lstm_kwargs=None,

                 dense=None

                 ):

        """工厂参数初始化"""

        self.name = name

        self.input_length = input_length

        if embedding_kwargs is not None:
            assert 'input_dim' not in embedding_kwargs
            assert 'output_dim' not in embedding_kwargs
            assert 'input_length' not in embedding_kwargs
            self.embeddings_kwargs = embedding_kwargs
        else:
            self.embeddings_kwargs = {}

        self.embedding_name = 'embedding'
        if 'name' in self.embeddings_kwargs:
            self.embedding_name = self.embeddings_kwargs.pop('name')

        self.embedding_word_number = embedding_word_number
        self.embedding_vector_length = embedding_vector_length
        self.embedding_dropout = embedding_dropout

        if lstm_kwargs is not None:
            assert 'units' not in lstm_kwargs
            self.lstm_kwargs = lstm_kwargs
        else:
            self.lstm_kwargs = {}

        self.lstm_name = 'lstm'
        if 'name' in self.lstm_kwargs:
            self.lstm_name = self.lstm_kwargs.pop('name')
        self.lstm_units = lstm_units
        self.lstm_dropout = lstm_dropout

        if dense is None:
            dense = []
        else:
            for layer in dense:
                assert isinstance(layer, dict)
                assert 'units' in layer and layer['units'] is not None and type(layer['units']) is int
                assert 'activation' in layer and layer['activation'] is not None and type(layer['activation']) is str
                if 'dropout' in layer:
                    assert type(layer['dropout']) is float
        self.dense = dense

    def __call__(self, inputs, **kwargs):
        """构建Layer"""

        embedding_layer_name = self.name + '.' + self.embedding_name
        x_embedded = Embedding(name=embedding_layer_name,
                               input_dim=self.embedding_word_number,
                               output_dim=self.embedding_vector_length,
                               input_length=self.input_length,
                               **self.embeddings_kwargs
                               )(inputs)

        if self.embedding_dropout is not None and self.embedding_dropout > 0:
            x_embedded = Dropout(self.embedding_dropout, name=embedding_layer_name+'.dropout')(x_embedded)

        lstm_layer_name = self.name + '.' + self.lstm_name
        x_lstm = LSTM(name=lstm_layer_name, units=self.lstm_units, **kwargs)(x_embedded)

        if self.lstm_dropout is not None and self.lstm_dropout > 0:
            x_lstm = Dropout(self.lstm_dropout, name=lstm_layer_name + '.dropout')(x_lstm)

        x_dense = x_lstm
        for i, kwargs in enumerate(self.dense):
            kwargs_cp = dict(kwargs)
            dropout = None
            if 'dropout' in kwargs_cp:
                dropout = kwargs_cp.pop('dropout')

            if 'name' in kwargs_cp:
                name = kwargs_cp.pop('name')
            else:
                name = 'dense'

            name = self.name + '.' + name + '.' + str(i)
            x_dense = Dense(name=name, **kwargs_cp)(x_dense)
            if dropout > 0:
                x_dense = Dropout(dropout, name=name+'.dropout')(x_dense)

        return x_dense


if __name__ == '__main__':
    INPUT_LENGTH = 6
    EMBEDDING_WORD_NUMBER = 600
    EMBEDDING_VECTOR_LENGTH = 32

    x = Input(shape=(INPUT_LENGTH,), dtype='int32', name='x')

    y = EmbeddingLSTM(name='ELSTM',

                      input_length=INPUT_LENGTH,

                      embedding_word_number=EMBEDDING_WORD_NUMBER,
                      embedding_vector_length=EMBEDDING_VECTOR_LENGTH,
                      embedding_dropout=0.8,
                      embedding_kwargs={'name': 'e', 'embeddings_initializer': 'lecun_uniform'},

                      lstm_units=16,
                      lstm_dropout=0.8,
                      lstm_kwargs={'name': 'l'},

                      dense=[
                          {'units': 4, 'activation': 'tanh', 'dropout': 0.9, 'name': 'd'},
                          {'units': 1, 'activation': 'sigmoid', 'dropout': 0.9, 'name': 'd'}
                      ]

                      )(x)
    model = Model(inputs=[x], outputs=[y])
    model.summary()



