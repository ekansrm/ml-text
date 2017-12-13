from xmlrpc.client import ServerProxy

server = ServerProxy('http://localhost:8888/RPC2')
print(server.testType(True, 123, 1.23, 'abc'))
print(server.predict("你是大傻瓜"))
print(server.text2Vec("ztg.alipay.com @卡觅 你在这台工具上部署，一下。"))
