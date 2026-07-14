import numpy
import time
import argparse
from qubichw import c_qubichw
from matplotlib import pyplot
def tosigned(val,width=16):
	return (val-2**width) if val>>(width-1) else (val)
vtosigned=numpy.vectorize(tosigned)


if __name__=="__main__":
	#parser=argparse.ArgumentParser()
	#clargs=parser.parse_args()
	qubichw=c_qubichw(init=False)
#	qubichw.write((('dacfreq',131*8),('dacamp',32767),('fmcdacen',3)))

	if 1:
		adc0buf=qubichw.read((('adc0buf'),))
		print(type(adc0buf))
		print('adc0buf read',[hex(i) for i in adc0buf[0:10]])
		adc0buf_1=vtosigned(adc0buf>>16)
		adc0buf_0=vtosigned(adc0buf&0xffff)
		pyplot.plot(adc0buf_0)
		pyplot.plot(adc0buf_1)
		pyplot.show()
	if 0:
		bufreadtest=qubichw.read((('bufreadtest'),))
		print('bufreadtest read',bufreadtest)
	if 0:
		ntest=100
		gap=numpy.zeros(ntest)
		gap0=numpy.zeros(ntest)
		gapf=numpy.zeros(ntest)
		#lastend=0
		for i in range(ntest):
			bufreadtest=qubichw.read((('bufreadtest'),))
#		diff=numpy.diff(bufreadtest)
#		diff2=numpy.diff(bufreadtest[10:-10])
			gap0[i]=bufreadtest[0]
			gapf[i]=bufreadtest[-1]#-lastend
			#lastend=bufreadtest[-1]
		#	print('index',i,'read',max(diff),min(diff))
#		if (max(diff)%(2**32)==1 and min(diff)%(2**32)==1):
#			pass
#		else:
#			print('read2',max(diff2),min(diff2))
#			print('read buf',bufreadtest)
##			pyplot.plot(bufreadtest)
#			pyplot.show()
		gapsec=4e-9*(gap0[1:]-gapf[0:-1])%(2**32)
#	print(max(gap),min(gap),gap.mean(),numpy.median(gap))
		print(max(gapsec),min(gapsec),gapsec.mean(),numpy.median(gapsec))
		pyplot.figure(1)
		pyplot.plot(gapsec,'*')
		pyplot.figure(2)
		pyplot.plot(gap0,'*')
		pyplot.figure(3)
		pyplot.plot(gapf,'*')
		pyplot.show()
