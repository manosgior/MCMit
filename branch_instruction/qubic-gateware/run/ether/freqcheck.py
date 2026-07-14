import ads54j60
import numpy
import time
import argparse
from qubichw import c_qubichw
from matplotlib import pyplot
if __name__=="__main__":
	#parser=argparse.ArgumentParser()
	#clargs=parser.parse_args()
	qubichw=c_qubichw(init=False)
	qubichw.write((('recclk',1),))
	if (1):
		freqdict=(qubichw.freqs())
		for k,v in freqdict.items():
			print('%8.8f %s'%(v,k))
