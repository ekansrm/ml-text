from xmlrpc.client import ServerProxy

server = ServerProxy('http://localhost:8888/RPC2')
print(server.testType(True, 123, 1.23, 'abc'))
# print(server.text2seq("ztg.alipay.com @卡觅 你在这台工具上部署，一下。"))
print(server.predict("这个太烂了!!!"))
