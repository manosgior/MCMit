import sys
import time
import serial
import numpy
from uartlb import c_uartlb
''',"mdiodatatx": {"access": "rw","addr_width": 0,"base_addr": 23,"data_width": 32,"sign": "unsigned","init": 0}
,"mdiostart": {"access": "rw","addr_width": 0,"base_addr": 24,"data_width": 32,"sign": "unsigned","init": 0}
,"mdiodatarx": {"access": "r","addr_width": 0,"base_addr": 25,"data_width": 32,"sign": "unsigned","init": 0}
,"mdiorxvalid": {"access": "r","addr_width": 0,"base_addr": 26,"data_width": 1,"sign": "unsigned","init": 0}
'''
def mdiowrite(ser,addr,data,devaddr=0b00111):
	mdiocmd=(0<<26)+((devaddr&0x1f)<<21)+((int(addr)&0x1f)<<16)+((int(data)&0xffff))
	ser.write(addr=23,data=mdiocmd)
	ser.write(addr=24,data=0)
def mdioread(ser,addr,devaddr=0b00111):
	mdiocmd=(1<<26)+((devaddr&0x1f)<<21)+((int(addr)&0x1f)<<16)+((0&0xffff)<<0)
	ser.write(addr=23,data=mdiocmd)
	ser.write(addr=24,data=0)
	(c,a,v)=ser.read(addr=25)
	(op,devaddr,regaddr,data)=mdiodecode(v)
	return int(data)
def mdiodecode(v):
	vint=int(v)
	st=(vint>>30)&0x3
	op=(vint>>28)&0x3
	devaddr=(vint>>23)&0x1f
	regaddr=(vint>>18)&0x1f
	tr=(vint>>16)&0x3
	data=vint&0xffff
	return (op,devaddr,regaddr,data)
if __name__=="__main__":
	import devcom
	dev=devcom.devcom(devcom.sndev)
	vc707uart=dev['vc707uart']
	#ser=c_uartlb('/dev/ttyUSB0',9600,0)
	ser=c_uartlb(vc707uart,9600,0)
	ser.write(addr=4,data=1)
	ser.write(addr=27,data=1000)
	ser.resetlb()
	ser.resetbuf()
	devaddr=0b00111;
	r1w0=1
	if (1):
		for regaddr in range(32):
			print(hex(regaddr),hex(mdioread(ser,regaddr)))
	if (0):
		regmap=((0x0,0x140),(0x4,0x9801))
		mdiowrite(ser,0x0,0x0140)
		mdiowrite(ser,0x4,0x9801)
		mdiowrite(ser,0x16,0x1)
		mdiowrite(ser,0x0,0x8140)
	if (0):
		for regaddr in range(32):
			print(hex(regaddr),hex(mdioread(ser,regaddr)))
	time.sleep(2)
	if (0):
		for regaddr in range(32):
			print(hex(regaddr),hex(mdioread(ser,regaddr)))
	time.sleep(5)
	if (0):
		for regaddr in range(32):
			print(hex(regaddr),hex(mdioread(ser,regaddr)))
	if (0):
		mdiowrite(ser,0x13,int(sys.argv[1],0))
		time.sleep(1)
		print(hex(mdioread(ser,0x13)))

