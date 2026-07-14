import ads54j60
import numpy
import time
import argparse
from qubichw import c_qubichw
from matplotlib import pyplot
if __name__=="__main__":
	qubichw=c_qubichw(init=False)
	print(qubichw.write((('i2cmux_reset_b',1),("hwreset",1))))

#	if (1):
#		freqdict=(qubichw.freqs())
#		for k,v in freqdict.items():
#			print('%8.3f %s'%(v,k))
