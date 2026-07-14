import time
import serial
import numpy
from uartlb import c_uartlb
if __name__=="__main__":
	import random
	ser=c_uartlb('/dev/ttyUSB1',9600,0)
	ser.write(addr=4,data=1)
	ser.resetlb()
	ser.resetbuf()
	ser.write(addr=13,data=2500)
	r0=(0b1110100<<9)+(0b1<<8)+(0x00)
	ser.write(addr=9,data=r0)
	print('write datax',hex(r0))
	(c,a,v)=ser.read(addr=9)
	print('read datatx',hex(int(v)))
	ser.write(addr=10,data=0)
	(c,a,v)=ser.read(addr=12)
	print('read valid',hex(int(v)))
	time.sleep(2)
	(c,a,v)=ser.read(addr=12)
	print('read valid after delay',hex(int(v)))
	(c,a,v)=ser.read(addr=11)
	print('read back',hex(int(v)))

#	r2=int(random.random()*(2**32-1))
#	ser.write(addr=2,data=r2)
#	r3=int(random.random()*(2**32-1))
#	ser.write(addr=3,data=r3)
#	(c,a,v)=ser.read(addr=1)
#	print(((r0+r2+r3+10)&0xffffffff)==v)
#	ser.write(addr=4,data=2)
#	ser.settimeout(1)
#	ser.resetbuf()
#	for i in range(10):
#		print(ser.readline())
#	ser.settimeout(0)
#	ser.write(addr=4,data=0)
#	ser.resetbuf()
#	(c,a,v0)=ser.read(addr=0)
#	(c,a,v1)=ser.read(addr=1)
#	(c,a,v2)=ser.read(addr=2)
#	(c,a,v3)=ser.read(addr=3)
#	print((int(r0+r2+r3+10)&0xffffffff)==v1)
#	print((int(v0+v2+v3+10)&0xffffffff)==v1)
#	for i in range(10):
#		(c,a,v9)=ser.read(addr=9)
#		print('ppscnt',v9)
#		time.sleep(1)
