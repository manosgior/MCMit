import sys
import json
import time
import serial
import numpy
from uartlb import c_uartlb
from fmc120init import *
def i2creadwrite(devaddr,r1w0,data,nack,stop=1):
	r0=((devaddr&0x7f)<<25)+((r1w0&1)<<24)+(data&0xffffff)
	ser.write(addr=9,data=r0,delay=0.1)
	#print('write datax',hex(r0))
#	(c,a,v)=ser.read(addr=9,delay=0.1)
#	print('read i2cdatatx',hex(int(v)))
	v=0
	ser.write(addr=10,data=(nack&0xf)+((stop&0x1)<<4),delay=0.1)
	#print((c,a,v))
	#print('trig start')
	if (r1w0==1):
		(c,a,v)=ser.read(addr=12)
		#print('read i2crxvalid',hex(int(v)))
		(c,a,v)=ser.read(addr=12)
		#print('read i2crxvalid after delay',hex(int(v)))
		(c,a,v)=ser.read(addr=11)
#		print('read back i2cdatarx',hex(int(v)),hex(int(v)>>9),hex((int(v)>>8)&0x1),hex((int(v)&0xff)))
#	print('i2creadwrite',[hex(i) for i in [devaddr,r1w0,data,nack]])
	return int(v)
def i2cwrite(devaddr,data,nack=2,stop=1):
	data=data<<((4-nack)*8)
	i2creadwrite(devaddr=devaddr&0x7f,r1w0=0,data=data,nack=nack,stop=stop)
def i2cread(devaddr,nack=2):
	v=i2creadwrite(devaddr=devaddr&0x7f,r1w0=1,data=0,nack=nack)
#	print('i2cread',hex(int(v)))
	return v
def devwrite(devaddr,addrdata,nack=2,stop=1):
	i2cwrite(devaddr=devaddr&0x7f,data=addrdata,nack=nack,stop=stop)
def devread(devaddr,addrdata,nack=2):
	devwrite(devaddr=devaddr,addrdata=addrdata,nack=2,stop=0)
	v=i2cread(devaddr=devaddr,nack=nack)
	return v
def cpldwrite(addr,data,nack=3):
	addrdata=(addr<<8)+(data<<0)
	devwrite(devaddr=0x1c,addrdata=addrdata,nack=nack)
def cpldread(addr,nack=2):
	addrdata=(addr)
	val=devread(addrdata=addrdata,devaddr=0x1c,nack=nack)
	return val&0xff
def spiwrite(cmd24,action):
	data08=(cmd24>>16)&0xff
	data07=(cmd24>>8)&0xff
	data06=(cmd24>>0)&0xff
	#print([hex(i) for i in [cmd24,data08,data07,data06]])
	cpldwrite(addr=0x08,data=data08)
	cpldwrite(addr=0x07,data=data07)
	cpldwrite(addr=0x06,data=data06)
	cpldwrite(addr=0x00,data=action)
#	cpldwrite(addr=0x00,data=action)
def spiread(cmd24,action):
	spiwrite(cmd24,action)
	#ver=cpldread(addr=0x05)
	#print('cpldread ver',hex(ver))
#	time.sleep(0.5)
	vlsb=cpldread(addr=0x0e)
	vmsb=cpldread(addr=0x0f)
	return (vmsb,vlsb)
def lmkcmd(r1w0,addr,data,w1w0=0):
	cmd24=((r1w0&0x1)<<23)+((w1w0&0x3)<<21)+((addr&0x1fff)<<8)+((data&0xff)<<0)
	return cmd24
def lmkwrite(addr,data):
	cmd24=lmkcmd(r1w0=0,addr=addr,data=data)
	spiwrite(cmd24=cmd24,action=0x01)
def lmkread(addr):
	cmd24=lmkcmd(r1w0=1,addr=addr,data=0)
	vmsb,vlsb=spiread(cmd24=cmd24,action=0x01)
	return vlsb
def adccmd(r1w0,m,p,ch,addr,data,w1w0=0):
	cmd24=((r1w0&0x1)<<23)+((m&0x1)<<22)+((p&0x1)<<21)+((ch&0x1)<<20)+((addr&0xfff)<<8)+((data&0xff)<<0)
	#cmd24=((r1w0&0x1)<<23)+((addr&0x7fff)<<8)+((data&0xff)<<0)
	#print(format(cmd24,'04x'))
	return cmd24
def adcwrite(addr,m,p,ch,data,adca0b1):
	cmd24=adccmd(r1w0=0,m=m,p=p,ch=ch,addr=addr,data=data)
	spiwrite(cmd24=cmd24,action=0x04 if adca0b1==0 else 0x08)
def adcread(addr,adca0b1,m=0,p=0,ch=0):
	cmd24=adccmd(r1w0=1,addr=addr,data=0,m=m,p=p,ch=ch)
	vmsb,vlsb=spiread(cmd24=cmd24,action=0x04 if adca0b1==0 else 0x08)
	#print(hex(vmsb),hex(vlsb))
	return vlsb
def daccmd(r1w0,addr,data):
	cmd24=((r1w0&0x1)<<23)+((addr&0x7f)<<16)+((data&0xffff)<<0)
	return cmd24
def dacwrite(addr,data):
	cmd24=daccmd(r1w0=0,addr=addr,data=data)
	spiwrite(cmd24=cmd24,action=0x02)
def dacread(addr):
	cmd24=daccmd(r1w0=1,addr=addr,data=0)
	vmsb,vlsb=spiread(cmd24=cmd24,action=0x02)
	return ((vmsb&0xff)<<8)+(vlsb&0xff)
def ad7291cmd(r1w0,addr,data):
	cmd24=((r1w0&0x1)<<23)+((addr&0x7f)<<16)+((data&0xffff)<<0)
	return cmd24
def ad7291write(addr,data):
	addrdata=((addr&0xff)<<16)+(data&0xffff)
	devwrite(devaddr=0x2f,addrdata=addrdata,nack=4)
def ad7291calc(val):
	chan=(val>>12)&0xf
	val12=val&0xfff
	if chan in [0,1,2,3]:
		voltage=val12*0.0006105
	elif chan in [4,5,6]:
		voltage=val12*0.001221
	elif chan in [7]:
		voltage=((4095-(val12+2047))*(-0.0016117))
	elif chan in [8,9]:
		voltage=val12/4.0
	else:
		print('which chan? chan=',chan)
		voltage=0
#	print(hex(val),chan,hex(val12),voltage)
	return (chan,voltage)
def ad7291read(addr):
	addrdata=((addr&0xff))#<<16)
	val=devread(devaddr=0x2f,addrdata=addrdata,nack=3)
	return val
def eepromcmd(r1w0,addr,data):
	cmd24=((r1w0&0x1)<<23)+((addr&0x7f)<<16)+((data&0xffff)<<0)
	return cmd24
def eepromwrite(addr,data,devaddr=0x50):
	addrdata=((addr&0xff)<<8)+(data)
	devwrite(devaddr=devaddr,addrdata=addrdata,nack=3)
def eepromread(addr,devaddr=0x50):
	addrdata=((addr&0xff))#<<16)
	val=devread(devaddr=devaddr,addrdata=addrdata,nack=2)
	return val&0xff
def sfpcmd(r1w0,addr,data):
	cmd24=((r1w0&0x1)<<23)+((addr&0x7f)<<16)+((data&0xffff)<<0)
	return cmd24
def sfpwrite(addr,data,devaddr=0x50):
	addrdata=((addr&0xff)<<8)+(data)
	devwrite(devaddr=devaddr,addrdata=addrdata,nack=3)
def sfpread(addr,devaddr=0x50):
	addrdata=((addr&0xff))#<<16)
	val=devread(devaddr=devaddr,addrdata=addrdata,nack=2)
	return val&0xff
def si570write(addr,data,devaddr=0x5d):
	addrdata=((addr&0xff)<<8)+(data)
	devwrite(devaddr=devaddr,addrdata=addrdata,nack=3)
def si570read(addr,devaddr=0x5d):
	addrdata=((addr&0xff))#<<16)
	val=devread(devaddr=devaddr,addrdata=addrdata,nack=2)
	return val&0xff

if __name__=="__main__":
	import devcom
	dev=devcom.devcom(devcom.sndev)
	vc707uart=dev['vc707uart']
#ser=c_uartlb('/dev/ttyUSB0',9600,0)
	ser=c_uartlb(vc707uart,9600,0)
	ser.write(addr=4,data=1)
	ser.write(addr=14,data=0)
#	time.sleep(1)
	ser.write(addr=14,data=1)
	ser.write(addr=18,data=0b1010_1010_1010_1010)
	ser.write(addr=18,data=0b1010_1001_1010_1001)
	ser.write(addr=18,data=0b1010_1010_1010_1010)

	ser.resetlb()
	ser.resetbuf()
	ser.resetbuf()
	print('prsnt',[hex(int(i)) for i in ser.read(addr=15)])
	print('pgm2c',[hex(int(i)) for i in ser.read(addr=17)])
	ser.write(addr=16,data=3)
	ser.write(addr=13,data=2500)
# switch to fmc1
	import dac39j84
	if (1):
		i2cwrite(devaddr=0x74,data=int(sys.argv[1]))#0x04)
		lmkwrite(addr=0x10,data=0x1010)
		for addr,val in dac39j84.default:
			rval=dacread(addr=addr)
			if rval!=val:
				print('addr',format(addr,'04x'),'val',format(rval,'04x'),'default',format(val,'04x'))
#		for addr,val in ads54j60.softreset:
#			m=(addr>>14)&0x1
#			p=(addr>>13)&0x1
#			ch=(addr>>12)&0x1
#			ad=(addr>>0)&0xfff
#			adcwrite(addr=addr,m=m,p=p,ch=ch,data=val,adca0b1=1)
#		for addr,val in ads54j60.default:
#			m=(addr>>14)&0x1
#			p=(addr>>13)&0x1
#			ch=(addr>>12)&0x1
#			ad=(addr>>0)&0xfff
##			print(hex(addr),m,p,ch,val)
##			before=adcread(addr=addr,m=m,p=p,ch=ch,adca0b1=1)
#			if addr in ads54j60.ctrladdr:
#				adcwrite(addr=addr,m=m,p=p,ch=ch,data=val,adca0b1=1)
#			after=adcread(addr=addr,m=m,p=p,ch=ch,adca0b1=1)
#			print('addr',hex(addr),'read',format(after,'x'))

#		m=1;p=0;ch=0;addr=0x03;dataint=0x0;adcwrite(addr=addr,m=m,p=p,ch=ch,data=dataint,adca0b1=1)
#		m=1;p=0;ch=0;addr=0x04;dataint=0x68;adcwrite(addr=addr,m=m,p=p,ch=ch,data=dataint,adca0b1=1)
#		m=1;p=0;ch=0;addr=0xf7;dataint=0x01;adcwrite(addr=addr,m=m,p=p,ch=ch,data=dataint,adca0b1=1)
#		m=1;p=0;ch=0;addr=0x00;dataint=0x01;adcwrite(addr=addr,m=m,p=p,ch=ch,data=dataint,adca0b1=1)
#		m=1;p=0;ch=0;addr=0x00;dataint=0x00;adcwrite(addr=addr,m=m,p=p,ch=ch,data=dataint,adca0b1=1)
#		m=1;p=1;ch=0;addr=0x41;print('adcread ',hex(addr),m,p,ch,hex(int(adcread(addr=addr,m=m,p=p,ch=ch,adca0b1=1))))
#		m=1;p=1;ch=0;addr=0x41;dataint=0xc5;adcwrite(addr=addr,m=m,p=p,ch=ch,data=dataint,adca0b1=1)
#		m=1;p=1;ch=0;addr=0x41;print('adcread ',hex(addr),m,p,ch,hex(int(adcread(addr=addr,m=m,p=p,ch=ch,adca0b1=1))))
#		time.sleep(1)
#		m=1;p=0;ch=0;addr=0x4;print('adcread ',hex(addr),m,p,ch,hex(int(adcread(addr=addr,m=m,p=p,ch=ch,adca0b1=1))))
#		m=1;p=1;ch=0;addr=0x4;print('adcread ',hex(addr),m,p,ch,hex(int(adcread(addr=addr,m=m,p=p,ch=ch,adca0b1=1))))
#		m=1;p=1;ch=1;addr=0x4;print('adcread ',hex(addr),m,p,ch,hex(int(adcread(addr=addr,m=m,p=p,ch=ch,adca0b1=1))))
#		m=1;p=1;ch=0;addr=0x4;print('adcread ',hex(addr),m,p,ch,hex(int(adcread(addr=addr,m=m,p=p,ch=ch,adca0b1=1))))
#		m=1;p=1;ch=0;addr=0x41;print('adcread ',hex(addr),m,p,ch,hex(int(adcread(addr=addr,m=m,p=p,ch=ch,adca0b1=1))))
##		m=0;p=0;ch=0;addr=0x5f;print('adcread ',hex(addr),m,p,ch,hex(int(adcread(addr=addr,m=m,p=p,ch=ch,adca0b1=1))))
#
