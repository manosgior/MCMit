import sys
import json
import time
import serial
import numpy
from uartlb import c_uartlb
from fmc120init import devwrite,devread,i2cwrite,ser
def si570write(addr,data,devaddr=0x5d):
	addrdata=((addr&0xff)<<8)+(data)
	devwrite(devaddr=devaddr,addrdata=addrdata,nack=3)
def si570read(addr,devaddr=0x5d):
	addrdata=((addr&0xff))#<<16)
	val=devread(devaddr=devaddr,addrdata=addrdata,nack=2)
	return val&0xff

if __name__=="__main__":
	import random
	ser=c_uartlb('/dev/ttyUSB0',9600,0)
	ser.write(addr=4,data=1)
	ser.write(addr=14,data=0)
#	time.sleep(1)
	ser.write(addr=14,data=1)
	ser.write(addr=18,data=0b1010_1010_1010_1010)
	ser.write(addr=18,data=0b1010_1001_1010_1001)
	ser.write(addr=18,data=0b1010_1010_1010_1010)

	ser.resetlb()
	ser.resetbuf()
	print('prsnt',ser.read(addr=15))
	print('pgm2c',ser.read(addr=17))
	ser.write(addr=16,data=3)
	ser.write(addr=13,data=2500)
	# switch to fmc1
	if (1):
		i2cwrite(devaddr=0x74,data=int(sys.argv[1]))#0x04)
	if (0):
		cpld05=cpldread(addr=0x05)
	if (0): # test i2c to sfp
		print('mux read',hex(i2cread(devaddr=0x74)))
		print('sfpread')
		for addr in range(128):
			val=sfpread(addr=addr)
			print('sfp addr ',hex(addr),addr,'value',hex(val),chr(val))
	if (0): # test on fmc120 board monitor
		ad7291write(addr=0x0,data=0x0002)
		for i in range(8):
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
	if (1): # test on fmc120 eeprom
		for addr in [7,8,9,10,11,12,13,14,15,16,17,18,135,137]:
			val=si570read(addr)
			print(format(addr,'3d'),format(addr,'02x'),format(val,'02x'))
	if (0): # test on single vc707 eeprom write and read
		eepromwrite(addr=0,data=0x5c,devaddr=0x54)
		print(hex(eepromread(addr=0,devaddr=0x54)))
	if (0): # test on vc707 eeprom write and read
		for page in range(1):
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
	if (0): # test on fmc120 spi
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
#		lmkwrite(addr=0x146,data=0x00)
#		lmkwrite(addr=0x147,data=0x10)
		lmkwrite(addr=0x148,data=0x33)
#		lmkwrite(addr=0x149,data=0x00)
#		lmkwrite(addr=0x14a,data=0x00)
#		lmkwrite(addr=0x14b,data=0x05)
#		time.sleep(1)
	if (0): # test on fmc120 spi
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
	if (0):
		#	lmkwrite(addr=0,data=0x80);
		#lmkwrite(addr=0,data=0x10);
		#lmkwrite(addr=2,data=0x00);
		with open('lmkinit.json') as f:
			lmkinit=json.load(f)
		#for addr in [0,2,3,4,5,6,0xc,0xd,0x100,0x101,0x103,0x104]:
		for addrstr,datastr in lmkinit:
			addr=int(addrstr,0)
			data=int(datastr,0)
		#	lmkwrite(addr=addr,data=data)
		#	print('lmk init addr %x data %x'%(addr,data))
			print('lmkread',hex(addr),hex(int(lmkread(addr=addr))));
	if (0):
		for addr in [1,2,3,4,5,6,7,8,0x0e,0x0f]:
			print('cpldread',hex(addr),hex(int(cpldread(addr=addr))))
	if (0):
		with open('ads54j60.json') as f:
			adcinit=json.load(f)
		for addrstr in ["0x4004"]:
	#	for addrstr,datastr in adcinit:
			addr=int(addrstr,0)
			print('adcread',hex(addr),hex(int(adcread(addr=addr,m=0,p=0,ch=0,adca0b1=0))))
	if (0):
		#for addr in range(0x6d):
		for addr in [0x6d]:
			print('dacread',hex(addr),hex(int(dacread(addr=addr))))
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
