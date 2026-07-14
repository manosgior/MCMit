#import json
#import numpy
import time
#import sys
#import devcom
#from ether import c_ether
#from uartlb import c_uartlb
#from regmap import c_regmap
#from fmc120 import c_fmc120
#from vc707 import c_vc707
#from lbaxi import c_lbaxi
#from udplb import c_udplb
import argparse

from qubichw import c_qubichw
if __name__=="__main__":
	parser=argparse.ArgumentParser()
	parser.add_argument('-ip','--ip',help='ip address',dest='ip',type=str,default=None)
	parser.add_argument('-mac','--mac',help='mac address',dest='mac',type=str,default=None)
	clargs=parser.parse_args()

	qubichw=c_qubichw(init=False,comport=True)
	print('c_qubichw')
	qubichw.macip(mac=clargs.mac,ip=clargs.ip)

#	def macip(mac=None,ip=None):
#		ad=[]
#		if clargs.mac!=None:
#			macval=[int('0x'+i,0) for i in clargs.mac.split(':')]
#			for index,val in enumerate(macval):
#				ad.append((index+0,macval[index]))
#		if clargs.ip!=None:
#			ipval=[int(i) for i in clargs.ip.split('.')]
#			for index,val in enumerate(ipval):
#				ad.append((index+6,ipval[index]))
#
#		qubichw.uartinit()
#		qubichw.vc707.setuarti2c(uarti2c=True)
#		qubichw.vc707.i2cenable()
#		qubichw.vc707.i2cswitch('eeprom')
#		for addr,val in ad:
#			qubichw.vc707.eepromwrite(addr=addr,data=val,devaddr=0x54)
#		time.sleep(0.5)
#		if len(ad)>0:
#			qubichw.uartregmap.write((('hwreset',0),))
#			time.sleep(0.5)
#
#		macmsb24,maclsb24,ipaddr,hwresetstatus=qubichw.uartregmap.read(('macmsb24','maclsb24','ipaddr','hwresetstatus'))
#		print('mac and ip',[hex(i) for i in [macmsb24,maclsb24,ipaddr,hwresetstatus]])
#		print('mac %02x:%02x:%02x:%02x:%02x:%02x'%((macmsb24>>16)&0xff,(macmsb24>>8)&0xff,macmsb24&0xff,(maclsb24>>16)&0xff,(maclsb24>>8)&0xff,maclsb24&0xff))
#		print('ip addr %d.%d.%d.%d'%((ipaddr>>24)&0xff,(ipaddr>>16)&0xff,(ipaddr>>8)&0xff,ipaddr&0xff))
#		print('hw reset status %x'%hwresetstatus)
#		qubichw.vc707.setuarti2c(uarti2c=False)
#		print(ad)
#
#		qubichw.vc707.i2cswitch('eeprom')
#		qubichw.vc707.eepromwrite(addr=9,data=ipmac8b,devaddr=0x54)
#		time.sleep(0.5)
#		print('eeprom readback, expect to bo 0x%x=%d'%(ipmac8b,ipmac8b),qubichw.vc707.eepromread(addr=9,devaddr=0x54))
#
#	import random
#	lastip=random.randint(0,255)
#	print(hex(lastip))
#	qubichw.changeipmac(lastip)
#	for i in range(10):
#		print(hex(qubichw.uartregmap.read((('hwresetstatus'),))))
#		time.sleep(0.1)
#	print([hex(i) for i in qubichw.uartregmap.read(('macmsb24','macmsb24','maclsb24','ipaddr','hwresetstatus'))])


