from xmlrpc.server import SimpleXMLRPCServer
import json


class SentimentAnalysisLstmKerasService(object):

    def testType(self, pBool: bool, pInt: int, pFloat: float, pStr: str):
        print(pBool)
        print(pInt)
        print(pFloat)
        print(pStr)
        return [pBool, pInt, pFloat, pStr]

    def predict(self, text: str):
        print(text)
        return "hhh"


service = SentimentAnalysisLstmKerasService()
server = SimpleXMLRPCServer(("localhost", 8888))
server.register_instance(service)
print("Listening on port 8888........")
server.serve_forever()
