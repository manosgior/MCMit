import numpy
import time
import argparse
from qubichw import c_qubichw
from matplotlib import pyplot
def tosigned(val,width=16):
	return (val-2**width) if val>>(width-1) else (val)
vtosigned=numpy.vectorize(tosigned)


if __name__=="__main__":
	parser=argparse.ArgumentParser()
#	parser.add_argument('-o','--on',help='on/off if no amp/freq set then default is off)',dest='on',action='store_true',default=False)
	parser.add_argument('-c','--channel',help='set channel',dest='chan',type=str,default=None,required=True,choices=['01','23','45','67'],nargs='+')
	parser.add_argument('-f','--freq',help='set freq up to 8 values, in Hz',dest='freq',type=float,nargs='+',default=None)
	parser.add_argument('-a','--amp',help='set amp up to 8 values, in fraction of full scale, you need to maintain the sum less than 1.0. - sign means -  ',dest='amp',type=float,nargs='+',default=None)
	clargs=parser.parse_args()
	qubichw=c_qubichw(init=False)
	print(clargs)
	regs=[('fmcdacen',3)]
	for chan in clargs.chan:
		if clargs.amp is not None:
			if len(clargs.amp)>8:
				print('you got %d amp input, I can only take 8 now'%len(clargs.amp))
			else:
				for index,value in enumerate(clargs.amp):
					regs.append(('dac%s_%damp'%(chan,index),int(round(value*32767))))
		if clargs.freq is not None:
			if len(clargs.freq)>8:
				print('you got %d freq input, I can only take 8 now'%len(clargs.freq))
			else:
				for index,value in enumerate(clargs.freq):
					regs.append(('dac%s_%dfreq'%(chan,index),int(round(value*4.0*2**17*1.0e-9))))
	print(regs)
	if len(regs):
		qubichw.write(regs)
