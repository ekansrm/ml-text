from xmlrpc.client import ServerProxy

server = ServerProxy('http://localhost:8888/RPC2')
print(server.testType(True, 123, 1.23, 'abc'))
