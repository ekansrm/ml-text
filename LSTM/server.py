from xmlrpc.server import SimpleXMLRPCServer

# from keras.models import load_model
import jieba


class SentimentAnalysisLstmKerasService(object):

    def __init__(self):
        self.model = None

    def load(self):
        # self.model = load_model('sentiment-analysis-lstm-keras.hdf5')
        pass

    def testType(self, pBool: bool, pInt: int, pFloat: float, pStr: str):
        print(pBool)
        print(pInt)
        print(pFloat)
        print(pStr)
        return [pBool, pInt, pFloat, pStr]

    def text2Vec(self, text: str):
        words = list(jieba.cut(text))
        return words

    def predict(self, text: str):
        print(text)
        return "hhh"


service = SentimentAnalysisLstmKerasService()
server = SimpleXMLRPCServer(("localhost", 8888))
server.register_instance(service)
print("Listening on port 8888........")
server.serve_forever()
