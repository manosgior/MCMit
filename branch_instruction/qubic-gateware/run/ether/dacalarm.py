import numpy
import time
import argparse
from qubichw import c_qubichw
from matplotlib import pyplot
import lmk04828
import ads54j60
import dac39j84
import cpld
import jesdaxi
import si5324

if __name__=="__main__":
	#	parser=argparse.ArgumentParser()
	#parser.add_argument('-sfptx','--sfptx',help='sfp tx',dest='sfp',type=int,default=None)
	#parser.add_argument('-smatx','--smatx',help='sma tx',dest='sma',type=int,default=None)
	#clargs=parser.parse_args()
	qubichw=c_qubichw(init=False)
	for fmc in [qubichw.fmc120_1,qubichw.fmc120_2]:
		qubichw.i2cswitch(fmc.i2cid)
		for addr,data in dac39j84.alarm:
			rdbk=fmc.dacread(addr)
			print('addr',addr,hex(addr),'alarm value',hex(data),data)
