import argparse
import sys
import time
import numpy
from qubichw import c_qubichw

if __name__=="__main__":
	parser=argparse.ArgumentParser()
	parser.add_argument('-resetpcs','--resetpcs',help='reset pcs',dest='resetpcs',default=False,action='store_true')
	parser.add_argument('-resetphy','--resetphy',help='reset phy',dest='resetphy',default=False,action='store_true')
	parser.add_argument('-read','--read',help='read all register',dest='read',default=False,action='store_true')
	parser.add_argument('-cfg','--cfg',help='cfg in the marvell_88e1111.py',dest='cfg',type=str,default=None)
	clargs=parser.parse_args()
	qubichw=c_qubichw(init=False,comport=True)
	qubichw.uartinit()
	import marvell_88e1111
	if clargs.cfg is not None:
		for devaddr,addr,val in getattr(marvell_88e1111,clargs.cfg):
			print(devaddr,addr,hex(val))
			qubichw.vc707.mdiowrite(addr,val,devaddr=devaddr)
	if clargs.resetphy:
		#		qubichw.vc707.mdiowrite(22,0x0,devaddr=0b00111)
#		qubichw.vc707.mdiowrite(0x4,0x8140,devaddr=0b00111)
#		qubichw.vc707.mdiowrite(0x16,0x1,devaddr=0b00111)
##		qubichw.vc707.mdiowrite(0x1b,0x8480,devaddr=0b00111)
#		qubichw.vc707.mdiowrite(0x4,0x9801,devaddr=0b00111)
##		qubichw.vc707.mdiowrite(0x9,0x0000,devaddr=0b00111)
		qubichw.vc707.mdiowrite(0x0,0x9140,devaddr=0b00111)
#		time.sleep(1)
		qubichw.vc707.mdiowrite(0x0,0x1140,devaddr=0b00111)
##		qubichw.vc707.mdiowrite(0x4,0x8140,devaddr=0b00111)
		time.sleep(1)
	if clargs.resetpcs:
		qubichw.vc707.mdiowrite(0x0,0x9140,devaddr=0b00110)
		qubichw.vc707.mdiowrite(0x0,0x8140,devaddr=0b00110)
		time.sleep(0.5)
	if clargs.read:
		for devaddr in [0b00110,0b00111]:
			print('devaddr',hex(devaddr))
			reg2=(qubichw.vc707.mdioread(2,devaddr=devaddr))
			reg3=(qubichw.vc707.mdioread(3,devaddr=devaddr))
			oui=(reg2<<6)+(reg3>>10)
			model=(reg3>>4)&0x3f
			revnum=(reg3&0xf)
			print('oui:',hex(oui),'model',hex(model),'revnum',hex(revnum))
			for regaddr in range(32):
				readval=qubichw.vc707.mdioread(regaddr,devaddr=devaddr)
				if readval!=marvell_88e1111.default[regaddr]:
					print(format(regaddr,'02x'),format(regaddr,'2d'),format(readval,'04x'))


#	dev=devcom.devcom(devcom.sndev)
#	vc707uart=dev['vc707uart']
#	#ser=c_uartlb('/dev/ttyUSB0',9600,0)
#	ser=c_uartlb(vc707uart,9600,0)
#	ser.write(addr=4,data=1)
#	ser.write(addr=27,data=1000)
#	ser.resetlb()
#	ser.resetbuf()
#	devaddr=0b00111;
#	r1w0=1
#	if (1):
#	if (0):
#		regmap=((0x0,0x140),(0x4,0x9801))
#		mdiowrite(ser,0x0,0x0140)
#		mdiowrite(ser,0x4,0x9801)
#		mdiowrite(ser,0x16,0x1)
#		mdiowrite(ser,0x0,0x8140)
#	if (0):
#		for regaddr in range(32):
#			print(hex(regaddr),hex(mdioread(ser,regaddr)))
#	time.sleep(2)
#	if (0):
#		for regaddr in range(32):
#			print(hex(regaddr),hex(mdioread(ser,regaddr)))
#	time.sleep(5)
#	if (0):
#		for regaddr in range(32):
#			print(hex(regaddr),hex(mdioread(ser,regaddr)))
#	if (0):
#		mdiowrite(ser,0x13,int(sys.argv[1],0))
#		time.sleep(1)
#		print(hex(mdioread(ser,0x13)))
#
