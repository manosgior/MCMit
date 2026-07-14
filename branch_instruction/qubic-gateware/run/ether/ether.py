#!/usr/bin/python
import sys
import socket
import struct
import time
import sys,getopt
import os
import random
import numpy
import datetime
from time import gmtime, strftime

class c_ether:
	" Ethernet IO class for PSPEPS local bus access through mem_gateway "
	def __init__(self, ip, port,timeout=None):
		self.ip = ip
		self.port = port
		print('connect ',ip,'at port ',port)
		self.socket = socket.socket(socket.AF_INET, socket.SOCK_DGRAM, 0)
		if timeout:
			self.socket.settimeout(timeout)
		self.connect()

	def connect(self):
		self.socket.connect((self.ip, self.port))

	def __del__(self):
		self.socket.close()

def usage():
	print('usage: mbtest.py [commands]')
	print('-t, --target <ip address>')
	print('-h, --help')
	print('-a, --address <address in hex>')

if __name__ == "__main__":
	ip_addr = '192.168.1.224'
	port = 0xd003 
	target = c_ether(ip_addr, port)
	print(dir(target.socket))
	#p=16376*[b'']
	p=184*[b'']
	for i in range(len(p)):
		p[i]=struct.pack('!Q',i)
	print(p)
	target.socket.send(b''.join(p))


'''
	if len(argv)==0:
		usage()
		sys.exit()
	try:
		opts, args = getopt.getopt(argv, 'hoa:t:s:',['help','target=','address=','size='])
	except getopt.GetoptError as err:
		print(str(err))
		sys.exit(2)

	ip_addr = '192.168.21.21'
	port = 3000
	reg_address = 0x100
	reg_size = 0x100

	for opt,arg in opts:
		if opt in ('-h', '--help'):
			usage()
			sys.exit()
		elif opt in ('-t', '--target'):
			ip_addr = arg
		elif opt in ('-a', '--address'):
			reg_address = arg
		elif opt in ('-s', '--size'):
			reg_size = arg

'''
