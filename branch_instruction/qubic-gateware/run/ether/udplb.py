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
from ether import c_ether
import getopt
class c_udplb():
	" Ethernet IO class for PSPEPS local bus access through udplb "
	def __init__(self, interface,run=True,cmds={'write':0x00,'read':0x10}):
		self.interface=interface
		self.run=run
		self.cmds=cmds

		pass

	def __del__(self):
		pass

	def unitrw(self,p):
		if self.run:
			readvalue=None
			while readvalue is None:
				self.interface.socket.send(p)
				time.sleep(0.01)
				try:
					readvalue, addr = self.interface.socket.recvfrom(1450)  # buffer size is 1024 bytes
				except:
					print('timeout...')
				#readvalue=p
		else:
			readvalue = h+p
		return readvalue

	def packet_build(self,cadlist=None):
		p=len(cadlist)*[b'']
		for i,(ctrl,addr,data) in enumerate(cadlist):
			lbwcmd=((int(ctrl)&0xff)<<56)+((int(addr)&0xffffff)<<32)+(int(data)&0xffffffff)
			pack=struct.pack('!Q',lbwcmd)
			p[i]=pack
		#print(p,pack,format(lbwcmd,'x'))
		return b''.join(p)


	def readwrite(self, cadlist=None):
		maxperpacket=180*8
		p = self.packet_build(cadlist=cadlist)
		result=b''
		while (len(p)>maxperpacket):  #  udp max packet size 1472
			readvalue=self.unitrw(p[0:maxperpacket])  # should page them later
			result+=readvalue
			p=p[maxperpacket:]
		if p:
			readvalue=self.unitrw(p)
			result+=readvalue
		return result

	def parse_readvalue(self,readvalue,packformat=None):
		nresponse=len(readvalue)/8
		if not packformat:
			packformat='!%dQ'%(nresponse)
		rcmd=struct.unpack(packformat, readvalue)
		rcmdarray=numpy.array([((r>>56)&0xff,(r>>32)&0xffffff,r&0xffffffff) for r in rcmd]).astype(int)
		return rcmdarray

if __name__=="__main__":
	from ether import c_ether
	lb = c_udplb(c_ether(ip='192.168.1.224',port=0xd003))
	cadlist=[(0x10,0x400,0)
			,(0x10,0x401,0)
			,(0x10,0x402,0)
			,(0x10,0x403,0)
			,(0x10,0x404,0)
			,(0x10,0x405,0)
			,(0x10,0x5ff,0)
			,(0x10,0x600,0)
			,(0x10,0x6ff,0)
			,(0x10,0x700,0)
			,(0x10,0x7f0,0)
			,(0x10,0x7f1,0)
			,(0x10,0x7f2,0)
			,(0x10,0x7f3,0)
			,(0x10,0x7f4,0)
			,(0x10,0x7f5,0)
			,(0x10,0x7f6,0)
			,(0x10,0x7f7,0)
			,(0x10,0x7f8,0)
			,(0x10,0x7f9,0)
			,(0x10,0x7fa,0)
			,(0x10,0x7fb,0)
			,(0x10,0x7fc,0)
			,(0x10,0x7fd,0)
			,(0x10,0x7fe,0)
			,(0x10,0x7ff,0)
			,(0x10,0x56,0)
			,(0x00,0x55,0)
			,(0x10,0x56,0)
			,(0x10,0x56,0)
			,(0x10,0x56,0)
			,(0x10,0x56,0)
			,(0x10,0x56,0)
			,(0x10,0x56,0)
			,(0x10,0x56,0)
			,(0x10,0x56,0)
			,(0x10,0x56,0)
			,(0x10,0x56,0)
			,(0x10,0x956,0)
			]
	result=lb.readwrite(cadlist)
	print(lb.parse_readvalue(result))
#	alist=int(sys.argv[1])*[22]
	#result=lb.readwrite(alist=alist,dlist=alist,write=0,rand=False)
	#print(result)
	#result=lb.readwrite(alist=alist,dlist=alist,write=1,rand=False)
	#print(result)
#	cadlist=numpy.zeros((len(alist),3),dtype=int)
#	cadlist[:,0]=len(alist)*[0x10]
#	cadlist[:,1]=alist
#	cadlist[:,2]=len(alist)*[0x0]
#	index=0
#	numpy.set_printoptions(formatter={'int':hex})
#	if (1):
#		result=lb.readwrite(cadlist=cadlist)
##	print(result)
#		print(index,lb.parse_readvalue(result))
##		index=index+1
