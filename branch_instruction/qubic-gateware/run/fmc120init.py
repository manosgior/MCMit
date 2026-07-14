import sys
import json
import time
import serial
import numpy
from uartlb import c_uartlb
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
	print('macmsb24',[hex(int(i)) for i in ser.read(addr=10)])
	print('macmsb24',[hex(int(i)) for i in ser.read(addr=11)])
	exit(3)
	print('prsnt',[hex(int(i)) for i in ser.read(addr=15)])
	print('pgm2c',[hex(int(i)) for i in ser.read(addr=17)])
	ser.write(addr=16,data=3)
	ser.write(addr=13,data=2500)
	# switch to fmc1
	if (0):
		i2cwrite(devaddr=0x74,data=int(sys.argv[1]))#0x04)
		print('mux read',hex(i2cread(devaddr=0x74)))
	if (0):
		cpld05=cpldread(addr=0x05)
		print('cpld05 reading',hex(cpld05))
	if (0): # test i2c to sfp
		i2cwrite(devaddr=0x74,data=16)#0x04)
		print('mux read',hex(i2cread(devaddr=0x74)))
		for addr in range(128):
			val=sfpread(addr=addr)
			print('sfp addr ',hex(addr),addr,'value',hex(val),chr(val))
	if (0): # test on fmc120 board monitor
		i2cwrite(devaddr=0x74,data=int(sys.argv[1]))#0x04)
		ad7291write(addr=0x0,data=0x0002)
		for i in range(7,-1,-1):
			ad7291write(addr=0x0,data=((1<<(i+8))+0x80))
			chv,vout=ad7291calc(ad7291read(0x01))
			cht,temp=ad7291calc(ad7291read(0x02))
			cha,tavr=ad7291calc(ad7291read(0x03))
			print('%d %3.2f %d %4.2f %d %4.2f'%(chv,vout,cht,temp,cha,tavr))
		#			print(hex(vout),hex(temp),hex(tavr))
	if (0): # test on fmc120 eeprom
		for page in range(16):
			val=0
			for addr in range(page*16+0,page*16+16):
				val=(val<<8)+eepromread(addr)
			print(hex(val))
	if (0): # test on single vc707 eeprom write and read
		eepromwrite(addr=0,data=0x5c,devaddr=0x54)
		print('vc707 eeprom wirte and read, should be 0x5c,read:',hex(eepromread(addr=0,devaddr=0x54)))
	if (0): # test on vc707 eeprom write and read
		i2cwrite(devaddr=0x74,data=8)#0x04)
		for page in range(16):
			for addr in range(page*16+0,page*16+16):
				eepromwrite(addr=addr,data=addr,devaddr=0x54)
				print('write page ',page,'addr ',addr)
		time.sleep(0.1)
		for page in range(1):
			val=0
			for addr in range(page*16+0,page*16+16):
				val=(val<<8)+eepromread(addr=addr,devaddr=0x54)
				print('read page ',page,'addr ',addr)
			print(hex(val))
	if (1): # test on fmc120 spi
		i2cwrite(devaddr=0x74,data=int(sys.argv[1]))#0x04)
		print('mux read',hex(i2cread(devaddr=0x74)))
		cpldwrite(addr=0x02,data=0x00)
		cpldwrite(addr=0x02,data=0x20)
		dacwrite(addr=0x02,data=0x2082)
#		lmkwrite(addr=0x15d,data=0x73)
		cpldwrite(addr=0x03,data=0x00)
		cpldwrite(addr=0x03,data=0x07)
		cpldwrite(addr=0x03,data=0x18)
		cpldwrite(addr=0x01,data=0xf0)
		lmkwrite(addr=0,data=0x90)
		lmkwrite(addr=0,data=0x10)
		lmkwrite(addr=0x10,data=0x10)
#		lmkwrite(addr=0x146,data=0x00)
#		lmkwrite(addr=0x147,data=0x10)
		lmkwrite(addr=0x148,data=0x33)
#		lmkwrite(addr=0x149,data=0x00)
#		lmkwrite(addr=0x14a,data=0x00)
#		lmkwrite(addr=0x14b,data=0x05)
#		time.sleep(1)
	if (0): # test on fmc120 spi
		i2cwrite(devaddr=0x74,data=int(sys.argv[1]))#0x04)
#		print('dacread',hex(int(dacread(addr=0x0))))
#		dacwrite(addr=0x0,data=0x18)
		lmkwrite(addr=0x10,data=0x1010)
#		print('lmkread',hex(3),hex(int(lmkread(addr=3))));
#		print('dacread',hex(int(dacread(addr=0x0))))
		for addr in range(16):
	#	for addr in [0,3]:#range(3):
			adca=adcread(addr=addr,adca0b1=0)
			lmk=lmkread(addr=addr)
			dac=dacread(addr=addr)
			adcb=adcread(addr=addr,adca0b1=1)
			print([hex(i) for i in [addr,lmk,dac,adca,adcb]])
		for addr in [0x0011]:#,0x004f,0x0026,0x0059,0x4003,0x4004,0x6842,0x684e,0x4005,0x4004]:
			m=(addr>>14)&0x1
			p=(addr>>13)&0x1
			ch=(addr>>12)&0x1
			adca=adcread(addr=addr,m=m,p=p,ch=ch,adca0b1=0)
			adcb=adcread(addr=addr,m=m,p=p,ch=ch,adca0b1=1)
			print([hex(i) for i in [addr,adca,adcb]])
	if (0):
		cpld03=cpldread(addr=0x03)
		print('cpld03',hex(cpld03))
		cpldwrite(addr=0x03,data=0x00)
		cpld03=cpldread(addr=0x03)
		print('cpld03',hex(cpld03))
		cpldwrite(addr=0x03,data=0x01)
		cpld03=cpldread(addr=0x03)
		print('cpld03',hex(cpld03))
		cpldwrite(addr=0x03,data=0x00)
		cpld03=cpldread(addr=0x03)
		cpldwrite(addr=0x03,data=0x00)
		print('cpld03',hex(cpld03))
		lmkwrite(addr=0,data=0x90)
		lmkwrite(addr=0,data=0x10)
		lmkwrite(addr=0,data=0x10)
		print('lmkread',hex(0),hex(int(lmkread(addr=0))));
		for addr in [0x6d]:
			print('dacread',hex(addr),hex(int(dacread(addr=addr))))
	if (0):
		devwrite(devaddr=0x2f,addrdata=0x0002,nack=3)
		cpldwrite(addr=0x01,data=0xf0)
		cpld03=cpldread(addr=0x03)
		print('cpld03',hex(cpld03))
	#	cpldwrite(addr=0x02,data=0x00)
		cpldwrite(addr=0x03,data=0x00)
		cpld03=cpldread(addr=0x03)
		print('cpld03',hex(cpld03))
		cpldwrite(addr=0x03,data=0x01)
		cpld03=cpldread(addr=0x03)
		print('cpld03',hex(cpld03))
	#	cpldwrite(addr=0x02,data=0x20)
		cpldwrite(addr=0x03,data=0x00)
		cpld03=cpldread(addr=0x03)
		print('cpld03',hex(cpld03))
		lmkwrite(addr=0,data=0x90);
		lmkwrite(addr=0,data=0x10);
	if (1):
		i2cwrite(devaddr=0x74,data=int(sys.argv[1]))#0x04)
		lmkwrite(addr=0x148,data=0x33)
		#	lmkwrite(addr=0,data=0x80);
		#lmkwrite(addr=0,data=0x10);
		#lmkwrite(addr=2,data=0x00);
		with open('lmkinit.json') as f:
			lmkinit=json.load(f)
		#for addr in [0,2,3,4,5,6,0xc,0xd,0x100,0x101,0x103,0x104]:
		for addrstr,datastr in lmkinit:
			addr=int(addrstr,0)
			data=int(datastr,0)
			print('lmkread',hex(addr),hex(int(lmkread(addr=addr))));
			lmkwrite(addr=addr,data=data)
		#	print('lmk init addr %x data %x'%(addr,data))
			print('lmkread',hex(addr),hex(int(lmkread(addr=addr))));
	if (0):
		for addr in [1,2,3,4,5,6,7,8,0x0e,0x0f]:
			print('cpldread',hex(addr),hex(int(cpldread(addr=addr))))
	if (0):
		i2cwrite(devaddr=0x74,data=int(sys.argv[1]))#0x04)
		lmkwrite(addr=0x10,data=0x1010)
		m=0;p=0;ch=0;addr=0x11;print('adcread ',hex(addr),m,p,ch,hex(int(adcread(addr=addr,m=m,p=p,ch=ch,adca0b1=1))))
		m=0;p=0;ch=0;addr=0x11;dataint=0x3c;adcwrite(addr=addr,m=m,p=p,ch=ch,data=dataint,adca0b1=1)
		time.sleep(1)
		m=0;p=0;ch=0;addr=0x11;print('adcread ',hex(addr),m,p,ch,hex(int(adcread(addr=addr,m=m,p=p,ch=ch,adca0b1=1))))
		m=0;p=0;ch=0;addr=0x5f;print('adcread ',hex(addr),m,p,ch,hex(int(adcread(addr=addr,m=m,p=p,ch=ch,adca0b1=1))))

	if (0):
		with open('ads54j60.json') as f:
			adcinit=json.load(f)
		#for addrstr in ["0x4004"]:
		for addrstr,datastr in adcinit:
			addrint=int(addrstr,0)
			addr=addrint&0xfff
			m=(addrint>>14)&0x1
			p=(addrint>>13)&0x1
			ch=(addrint>>12)&0x1
			dataint=int(datastr,0)
			adcwrite(addr=addr,m=m,p=p,ch=ch,data=dataint,adca0b1=1)
			print(addrstr,datastr)
			print('adcread before ',hex(addr),m,p,ch,hex(int(adcread(addr=addr,m=m,p=p,ch=ch,adca0b1=1))))
			print('adcwrite',addr,m,p,ch,dataint)
			print('adcread after  ',hex(addr),m,p,ch,hex(int(adcread(addr=addr,m=m,p=p,ch=ch,adca0b1=1))))
	if (0):
		for addr in range(0x6d):
		#for addr in [0x6d]:
			print('dacread',hex(addr),hex(int(dacread(addr=addr))))
	if (0): # 
		i2cwrite(devaddr=0x74,data=1)
		for addr in [7,8,9,10,11,12,13,14,15,16,17,18,135,137]:
			val=si570read(addr)
			print(format(addr,'3d'),format(addr,'02x'),format(val,'02x'))
		addr=18
		si570write(addr,0x7f)
		val=si570read(addr)
		print('si570',format(addr,'3d'),format(addr,'02x'),format(val,'02x'))
	if (0):
		# cpld verify ADDR_VERSION=0x10
		i2creadwrite(addr=0x1c,r1w0=0,val=0x050000,nack=2,stop=0)
		i2creadwrite(addr=0x1c,r1w0=1,val=0xcf0000,nack=2)
		i2creadwrite(addr=0x1c,r1w0=0,val=0x030000,nack=2,stop=0)
		i2creadwrite(addr=0x1c,r1w0=1,val=0xcf0000,nack=2)
		i2creadwrite(addr=0x1c,r1w0=0,val=0x030100,nack=3)
		i2creadwrite(addr=0x1c,r1w0=0,val=0x030000,nack=2,stop=0)
		i2creadwrite(addr=0x1c,r1w0=1,val=0xcf0000,nack=2)
		i2creadwrite(addr=0x1c,r1w0=0,val=0x030000,nack=3)
		i2creadwrite(addr=0x1c,r1w0=0,val=0x030000,nack=2,stop=0)
		i2creadwrite(addr=0x1c,r1w0=1,val=0xcf0000,nack=2)
	if (0):
		# cpld write to lmk change to 4 wire
		i2creadwrite(addr=0x1c,r1w0=0,val=0x080000,nack=3)
		i2creadwrite(addr=0x1c,r1w0=0,val=0x070000,nack=3)
		i2creadwrite(addr=0x1c,r1w0=0,val=0x061000,nack=3)
		i2creadwrite(addr=0x1c,r1w0=0,val=0x000100,nack=3)
	if (0):
		i2creadwrite(addr=0x1c,r1w0=0,val=0x088000,nack=3)
		i2creadwrite(addr=0x1c,r1w0=0,val=0x070000,nack=3)
		i2creadwrite(addr=0x1c,r1w0=0,val=0x060000,nack=3)
		i2creadwrite(addr=0x1c,r1w0=0,val=0x000100,nack=3)
	if (0):
		i2creadwrite(addr=0x1c,r1w0=0,val=0x088100,nack=3)
		i2creadwrite(addr=0x1c,r1w0=0,val=0x070000,nack=3)
		i2creadwrite(addr=0x1c,r1w0=0,val=0x060000,nack=3)
		i2creadwrite(addr=0x1c,r1w0=0,val=0x000200,nack=3)
	if (0):
		i2creadwrite(addr=0x1c,r1w0=0,val=0x088000,nack=3)
		i2creadwrite(addr=0x1c,r1w0=0,val=0x073c00,nack=3)
		i2creadwrite(addr=0x1c,r1w0=0,val=0x060000,nack=3)
		i2creadwrite(addr=0x1c,r1w0=0,val=0x000400,nack=3)

	if (0):
		i2creadwrite(addr=0x1c,r1w0=0,val=0x0e0000,nack=2,stop=0)
		i2creadwrite(addr=0x1c,r1w0=1,val=0x000000,nack=2)
		i2creadwrite(addr=0x1c,r1w0=0,val=0x0f0000,nack=2,stop=0)
		i2creadwrite(addr=0x1c,r1w0=1,val=0x000000,nack=2)
#		time.sleep(0.1)
	if (0):
		i2creadwrite(addr=0x1c,r1w0=0,val=0x010000,nack=2,stop=0)
		i2creadwrite(addr=0x1c,r1w0=1,val=0xf00000,nack=2)
		i2creadwrite(addr=0x1c,r1w0=0,val=0x020000,nack=2,stop=0)
		i2creadwrite(addr=0x1c,r1w0=1,val=0xff0000,nack=2)
		i2creadwrite(addr=0x1c,r1w0=0,val=0x030000,nack=2,stop=0)
		i2creadwrite(addr=0x1c,r1w0=1,val=0xff0000,nack=2)
		i2creadwrite(addr=0x1c,r1w0=0,val=0x040000,nack=2,stop=0)
		i2creadwrite(addr=0x1c,r1w0=1,val=0xff0000,nack=2)
		i2creadwrite(addr=0x1c,r1w0=0,val=0x050000,nack=2,stop=0)
		i2creadwrite(addr=0x1c,r1w0=1,val=0xff0000,nack=2)
		i2creadwrite(addr=0x1c,r1w0=0,val=0x060000,nack=2,stop=0)
		i2creadwrite(addr=0x1c,r1w0=1,val=0xff0000,nack=2)
		i2creadwrite(addr=0x1c,r1w0=0,val=0x070000,nack=2,stop=0)
		i2creadwrite(addr=0x1c,r1w0=1,val=0xff0000,nack=2)
		i2creadwrite(addr=0x1c,r1w0=0,val=0x080000,nack=2,stop=0)
		i2creadwrite(addr=0x1c,r1w0=1,val=0xff0000,nack=2)
		i2creadwrite(addr=0x1c,r1w0=0,val=0x0e0000,nack=2,stop=0)
		i2creadwrite(addr=0x1c,r1w0=1,val=0xff0000,nack=2)
		i2creadwrite(addr=0x1c,r1w0=0,val=0x0f0000,nack=2,stop=0)
		i2creadwrite(addr=0x1c,r1w0=1,val=0xff0000,nack=2)
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
