from xmlrpc.server import SimpleXMLRPCServer

from keras.models import load_model
from keras.preprocessing import sequence
import jieba
import json
import numpy as np  # 导入Numpy
from collections import defaultdict

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


class SentimentAnalysisLstmKerasService(object):

    def __init__(self):
        self.model = None

    def load(self):
        self.model = load_model('sentiment-analysis-lstm.checkpoint.best')
        pass

    def testType(self, pBool: bool, pInt: int, pFloat: float, pStr: str):
        return [pBool, pInt, pFloat, pStr]

    def text2seq(self, text: str):
        words = list(jieba.cut(text))
        wordsTokens = tokenize(words)
        seq = list(sequence.pad_sequences([wordsTokens], maxlen=maxlen))
        return seq

    def predict(self, text: str):
        seq = np.array(self.text2seq(text))
        rv = self.model.predict(seq)
        rv = list(rv)
        rv = rv[0][0]
        return float(rv)


service = SentimentAnalysisLstmKerasService()
service.load()
server = SimpleXMLRPCServer(("localhost", 8888))
server.register_instance(service)
print("Listening on port 8888........")
server.serve_forever()
