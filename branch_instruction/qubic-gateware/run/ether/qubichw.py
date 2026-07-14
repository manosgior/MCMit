import os
import json
import numpy
import time
import sys
import devcom
from ether import c_ether
from uartlb import c_uartlb
from regmap import c_regmap
from fmc120 import c_fmc120
from vc707 import c_vc707
from lbaxi import c_lbaxi
from udplb import c_udplb
import lmk04828
import ads54j60
import dac39j84
import jesdaxi
import si5324
class c_qubichw():
	def __init__(self,ip=None,port=None,regmappath='regmap.json',init=False,commcheck=True,freqoffpercent=1,comport=False):
		t0=time.time()
		if ip is not None:
			self.ip=ip
		elif 'QUBICIP' in os.environ:
			self.ip=os.environ['QUBICIP']
		else:
			self.ip='192.168.1.224'
		if port is not None:
			self.port=port
		elif 'QUBICPORT' in os.environ:
			self.port=os.environ['QUBICPORT']
		else:
			self.port=0xd003
		self.ether=c_ether(ip=self.ip,port=self.port,timeout=1)
		self.udplb=c_udplb(interface=self.ether,run=True)
		self.regmap=c_regmap(interface=self.udplb,regmappath='regmap.json',wavegrppath='wavegrp.json')
		print(time.time()-t0)

		self.etherd002=c_ether(ip=self.ip,port=0xd002,timeout=1)
		self.udplbd002=c_udplb(interface=self.etherd002)
		if comport:
			if 'QUBICUSBSN' in os.environ:
				self.qubicusbsn=os.environ['QUBICUSBSN']
			else:
				self.qubicusbsn='vc707uart'
			print('qubic usb sn',self.qubicusbsn)
			dev=devcom.devcom(devcom.sndev)
			print(dev.keys())
			vc707uart=dev[self.qubicusbsn]
			self.uartlb=c_uartlb(vc707uart,9600,0)
			self.uartregmap=c_regmap(self.uartlb,regmappath='uartregmap.json',wavegrppath='uartwavegrp.json')
		else:
			self.uartregmap=None
		print(time.time()-t0)

		self.vc707=c_vc707(self.regmap,self.uartregmap)
		self.read=self.vc707.read
		self.write=self.vc707.write
		self.i2cenabled=False
		print(time.time()-t0)

#		self.vc707.i2cswitch('si570')
#		import random
#		lastip=random.randint(0,255)
#		print(hex(lastip))
#		self.changeipmac(lastip)
#		for i in range(10):
#			print(hex(self.read((('hwresetstatus'),))))
#			time.sleep(0.1)
#		print([hex(i) for i in self.uartregmap.read(('macmsb24','macmsb24','maclsb24','ipaddr','hwresetstatus'))])
		#
#		exit(34)
		#
		self.freqnominal={'freq_lb':125
,'freq_sgmiiclk':125
,'freq_sma_mgt_refclk':0
,'freq_si5324_out_c':250
,'freq_si5324_out_div2':62.5
,'freq_smamgtclk_div2':62.5
,'freq_pcie_clk_qo':0
,'freq_ethclk':125
,'freq_fmc1_llmk_dclkout_2':250
,'freq_fmc1_llmk_sclkout_3':1.953
,'freq_fmc1_lmk_dclk8_m2c_to_fpga':500
,'freq_fmc1_lmk_dclk10_m2c_to_fpga':0
,'freq_fmc2_llmk_dclkout_2':250
,'freq_fmc2_llmk_sclkout_3':1.953
,'freq_fmc2_lmk_dclk8_m2c_to_fpga':500
,'freq_fmc2_lmk_dclk10_m2c_to_fpga':0
,'freq_rxusrclk_sfp':100
,'freq_txusrclk_sfp':100
,'freq_user_clock':100
,'freq_rxusrclk_smasfp':100
,'freq_txusrclk_smasfp':100
}
		self.fmc120_1=c_fmc120(i2cid='fmc1',devread=self.vc707.devread,devwrite=self.vc707.devwrite)
		self.fmc120_2=c_fmc120(i2cid='fmc2',devread=self.vc707.devread,devwrite=self.vc707.devwrite)
		self.axiinsts={'fmc1':['axifmc1adc0','axifmc1adc1','axifmc1dac'],'fmc2':['axifmc2adc0','axifmc2adc1','axifmc2dac']}
		self.axinames=['axifmc1adc0','axifmc1adc1','axifmc1dac','axifmc2adc0','axifmc2adc1','axifmc2dac']
		self.lbaxi={}
		for name in self.axinames:
			self.lbaxi[name]=c_lbaxi(read=self.vc707.read,write=self.vc707.write)
		if init :
			self.fmcprsntpg()
			#self.vc707.i2cenable()
			freqdict=self.freqs()
			ok=True
			for k,v in freqdict.items():
				n=self.freqnominal[k]
				maxoffset=n*freqoffpercent*0.01
				if (abs(v-n)>maxoffset):
					print(k,n,v)
					ok=False
			for k,v in freqdict.items():
				print('%8.3f %s'%(v,k))
#		print(freqdict)
			print('clkinit ok?', ok)
			if ok:
				print('all clk frequency seems ok, skip lmk init')
			else:
				if (1):
					self.fmcreset()
					self.clkinit(lmkinitregs=lmk04828.init)
					self.si570setfreq(105.005e6)
					self.si5324init(si5324.init125)
#			if commcheck:
#				if self.commcheck(lmkreglist=lmk04828.ver,adcreglist=ads54j60.reg5f,dacreglist=dac39j84.vid,axireglist=jesdaxi.axiver):
			#if commcheck:
			#	print(self.vc707.si570readinit())
				if 0:
					self.adcinit(initregs=ads54j60.init,checkregs=ads54j60.init)
				if 1:
					self.dacinit(initregs=dac39j84.init)
#			for name in self.lbaxi:
#				if ('adc' in name):
#						self.axiinit(name,initregs=jesdaxi.adc)
#				elif ('dac' in name):
#						self.axiinit(name,initregs=jesdaxi.dac)
#				else:
#					print('unknown',name)
				#self.axiinit(initregs=axiinit.axiinit)


			freqdict=(self.freqs())
			for k,v in freqdict.items():
				print('%8.3f %s'%(v,k))
			for fmc in [self.fmc120_1,self.fmc120_2]:
				self.i2cswitch(fmc.i2cid)
				print(fmc.i2cid,'ad7291',fmc.ad7291check())
		print(time.time()-t0)
	def fmcreset(self):
		for fmc in [self.fmc120_1,self.fmc120_2]:
			self.i2cswitch(fmc.i2cid)
			print('reset fmc')
#			fmc.reset()
	def si570setfreq(self,freq):
		self.i2cswitch('si570')
		print(self.vc707.si570setfreq(freq))
	def i2cenable(self,clk4ratio=5000,enable=True):
		print('enable i2c')
		self.vc707.i2cenable(clk4ratio=clk4ratio,enable=enable)
		self.i2cenabled=True
	def i2cswitch(self,i2cid):
		print('i2cswitch to ',i2cid)
		if not self.i2cenabled:
			self.i2cenable()
		self.vc707.i2cswitch(i2cid)

	def uartinit(self):
		self.uartregmap.write((('uartmode',1),('clk4ratio',5000)))
		self.uartlb.resetlb()
		self.uartlb.resetbuf()
		time.sleep(0.1)
		self.uartlb.resetbuf()
		print(self.uartregmap.read((('uartmode','uartmode','clk4ratio','macmsb24','maclsb24','ipaddr'))))
	def commcheck(self,lmkreglist,adcreglist,dacreglist,axireglist):
		commpass=True
		for fmc in [self.fmc120_1,self.fmc120_2]:
			self.i2cswitch(fmc.i2cid)
			commpass=commpass & fmc.lmkcheck(lmkreglist)
			for adc in range(2):
				commpass=commpass & fmc.adccheck(adca0b1=adc,reglist=adcreglist)
			commpass=commpass & fmc.daccheck(dacreglist)
		for name,axi in self.lbaxi.items():
			commpass=commpass & axi.axi4lite_check(axiprefix=name,reglist=axireglist)
		print('commcheck','pass' if commpass else 'fail')
		return commpass

	def clkinit(self,lmkinitregs):
		print('clkinit')
		for fmc in [self.fmc120_1,self.fmc120_2]:
			self.i2cswitch(fmc.i2cid)
			fmc.lmk04828load(lmkinitregs)
		print('clkinit done')
	def adcinit(self,initregs=None,checkregs=None,softreset=True):
		#if not self.i2cenabled:
		#	self.vc707.i2cenable()
		for fmc in [self.fmc120_1,self.fmc120_2]:
			self.i2cswitch(fmc.i2cid)
			for adc in range(2):
				print('fmc120 %s adc %d'%(fmc.i2cid,adc))
				if softreset:
					fmc.adcload(adca0b1=adc,reglist=ads54j60.softreset)
					time.sleep(0.5)
				if initregs:
					fmc.adcload(adca0b1=adc,reglist=initregs)
				if checkregs:
					fmc.adccheck(adca0b1=adc,reglist=checkregs,printall=False)
	def dacinit(self,initregs):
		for fmc in [self.fmc120_1,self.fmc120_2]:
			self.i2cswitch(fmc.i2cid)
			fmc.dacload(initregs)
			fmc.daccheck(initregs)
	def axiinit(self,name,initregs):
		axi=self.lbaxi[name]
		if (axi.axi4lite_check(axiprefix=name,reglist=jesdaxi.axiver)):
			for addr,data in initregs:
				#data,valid=axi.axi4lite_read(axiprefix=name,addr=addr)
				axi.axi4lite_write(axiprefix=name,addr=addr,wdata=data)
				time.sleep(0.1)
				print('write axi',addr,data)
			resetval,valid=axi.axi4lite_read(axiprefix=name,addr=4)
			print('read axi %s  addr'%axi,addr,'resetval',resetval,hex(resetval))
			axi.jesd204axi_reset(axiprefix=name)


	def lbi2c(self,enable=True):
		self.write((('lbi2c',1 if enable else 0),))
	def si5324enable(self,enable=True):
		self.write((('si5324_rst',1 if enable else 0),))
	def fmcprsntpg(self):
		prsnt,pg=self.read(('fmcprsnt','fmcpgm2c'))
		self.fmc120_1.prsnt=(prsnt&0x1)==0
		self.fmc120_2.prsnt=((prsnt>>1)&0x1)==0
		self.fmc120_1.pg=(pg&0x1)==1
		self.fmc120_2.pg=((pg>>1)&0x1)==1
		return (prsnt,pg)
	def freqs(self):
		freqregs=list(self.freqnominal.keys())
	#	['freq_lb'
	#,'freq_sgmiiclk'
	#,'freq_sma_mgt_refclk'
	#,'freq_si5324_out_c'
	#,'freq_pcie_clk_qo'
	#,'freq_user_clock'
	#,'freq_fmc1_llmk_dclkout_2'
	#,'freq_fmc1_llmk_sclkout_3'
	#,'freq_fmc1_lmk_dclk8_m2c_to_fpga'
	#,'freq_fmc1_lmk_dclk10_m2c_to_fpga'
	#,'freq_fmc2_llmk_dclkout_2'
	#,'freq_fmc2_llmk_sclkout_3'
	#,'freq_fmc2_lmk_dclk8_m2c_to_fpga'
	#,'freq_fmc2_lmk_dclk10_m2c_to_fpga'
	#,'freq_rxusrclk_sfp'
	#,'freq_txusrclk_sfp'
	#,'freq_ethclk'
	#]
		freqsint=self.vc707.read(freqregs)
		print(freqsint)
	#	freqs=(numpy.array(self.vc707.read(freqregs))*125.0/2**24)
		freqs=numpy.array(freqsint)*125.0/2**24
		freqdict={}
		for index,freq in enumerate(freqs):
			freqdict[freqregs[index]]=freqs[index]
		return freqdict
	def si5324check(self,reglist,printall=False):
		print('si5324init')
		import si5324
		self.si5324enable()
		self.i2cswitch('si5324')
		for addr,data in reglist:#si5324.init250:
			rdbk=self.vc707.si5324read(addr)
			if rdbk!=data:
				print('si5324 init addr',addr,hex(addr), 'should be ', data, hex(data), 'rdbk',rdbk, hex(rdbk))
			else:
				if printall:
					print('si5324 init addr',addr,hex(addr), 'match, val', data, hex(data))

		time.sleep(1)
		print(' si5324 init done')
	def si5324init(self,reglist):
		print('si5324init')
		import si5324
		self.si5324enable()
		self.i2cswitch('si5324')
		self.vc707.si5324load(reglist)
		print(' si5324 init done')
	def macip(self,mac=None,ip=None):
		ad=[]
		if mac!=None:
			macval=[int('0x'+i,0) for i in mac.split(':')]
			for index,val in enumerate(macval):
				ad.append((index+0,macval[index]))
		if ip!=None:
			ipval=[int(i) for i in ip.split('.')]
			for index,val in enumerate(ipval):
				ad.append((index+6,ipval[index]))

		self.uartinit()
		self.vc707.setuarti2c(uarti2c=True)
		self.vc707.i2cenable()
		self.i2cswitch('eeprom')
		for addr,val in ad:
			self.vc707.eepromwrite(addr=addr,data=val,devaddr=0x54)
		time.sleep(0.5)
		if len(ad)>0:
			self.uartregmap.write((('hwreset',0),))
			time.sleep(0.5)

		macmsb24,maclsb24,ipaddr,hwresetstatus=self.uartregmap.read(('macmsb24','maclsb24','ipaddr','hwresetstatus'))
		mac=((macmsb24&0xffffff)<<24)+(maclsb24&0xffffff)
#		print('mac and ip',[hex(i) for i in [macmsb24,maclsb24,ipaddr,hwresetstatus]])
		print('mac read back %02x:%02x:%02x:%02x:%02x:%02x'%((mac>>40)&0xff
,(mac>>32)&0xff
,(mac>>24)&0xff
,(mac>>16)&0xff
,(mac>>8)&0xff
,(mac>>0)&0xff))
		print('ip addr readback %d.%d.%d.%d'%((ipaddr>>24)&0xff,(ipaddr>>16)&0xff,(ipaddr>>8)&0xff,ipaddr&0xff))
		print('hw reset status %x'%hwresetstatus)
		self.vc707.setuarti2c(uarti2c=False)
#		print(ad)
		return (mac,ip)
#	def changeipmac(self,ipmac8b):
#		self.vc707.i2cswitch('eeprom')
#		self.vc707.eepromwrite(addr=9,data=ipmac8b,devaddr=0x54)
#		time.sleep(0.5)
#		print('eeprom readback, expect to bo 0x%x=%d'%(ipmac8b,ipmac8b),self.vc707.eepromread(addr=9,devaddr=0x54))
#		self.write((('hwreset',0),))
#
if __name__=="__main__":
	import argparse
	parser=argparse.ArgumentParser()
	parser.add_argument('-ip','--ip',help='ip address',dest='ip',type=str,default='192.168.1.224')
	clargs=parser.parse_args()

	qubichw=c_qubichw(ip=clargs.ip,init=True)
	freqdict=(qubichw.freqs())
	for k,v in freqdict.items():
		print('%8.3f %s'%(v,k))
