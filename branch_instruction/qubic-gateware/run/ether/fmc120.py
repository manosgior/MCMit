import json
import numpy
import time
import sys
import devcom
from ether import c_ether
from uartlb import c_uartlb
from regmap import c_regmap
class c_fmc120:
	def __init__(self,i2cid,devread,devwrite):
		self.i2cid=i2cid
		self.devread=devread
		self.devwrite=devwrite
		self.prsnt=None
		self.pg=None
	def cpldwrite(self,addr,data,nack=3,devaddr=0x1c):
		addrdata=(addr<<8)+(data<<0)
		self.devwrite(devaddr=devaddr,addrdata=addrdata,nack=nack)
	def cpldread(self,addr,nack=2,devaddr=0x1c):
		addrdata=(addr)
		val=self.devread(addrdata=addrdata,devaddr=devaddr,nack=nack)
		return val&0xff
	def cpldcheck(self,reglist):
		regmatch=True
		for (addr,data) in reglist:
			rdbk=self.cpldread(addr)
			if rdbk!=data:
				regmatch=False
				print('cpld check','addr',addr,hex(addr),'should be',hex(data),'read back',hex(rdbk))
		return regmatch #(self.cpldread(addr=0x05)[0]==0x10)
	def spiwrite(self,cmd24,action):
		data08=(cmd24>>16)&0xff
		data07=(cmd24>>8)&0xff
		data06=(cmd24>>0)&0xff
		#print([hex(i) for i in [cmd24,data08,data07,data06]])
		self.cpldwrite(addr=0x08,data=data08)
		self.cpldwrite(addr=0x07,data=data07)
		self.cpldwrite(addr=0x06,data=data06)
		self.cpldwrite(addr=0x00,data=action)
#	cpldwrite(addr=0x00,data=action)
	def spiread(self,cmd24,action):
		self.spiwrite(cmd24,action)
		#ver=cpldread(addr=0x05)
		#print('cpldread ver',hex(ver))
#	time.sleep(0.5)
		vlsb=self.cpldread(addr=0x0e)
		vmsb=self.cpldread(addr=0x0f)
		return (vmsb,vlsb)
	def lmkcmd(self,r1w0,addr,data,w1w0=0):
		cmd24=((r1w0&0x1)<<23)+((w1w0&0x3)<<21)+((addr&0x1fff)<<8)+((data&0xff)<<0)
		return cmd24
	def lmkwrite(self,addr,data):
		cmd24=self.lmkcmd(r1w0=0,addr=addr,data=data)
		self.spiwrite(cmd24=cmd24,action=0x01)
	def lmkread(self,addr):
		cmd24=self.lmkcmd(r1w0=1,addr=addr,data=0)
		vmsb,vlsb=self.spiread(cmd24=cmd24,action=0x01)
		return vlsb
	def lmkcheck(self,reglist):
		regmatch=True
		for addr,data in reglist:
			rdbk=self.lmkread(addr)
			if rdbk!=data:
				regmatch=False
				print('lmkcheck','addr',addr,hex(addr),'should be',hex(data),'read back',hex(rdbk))
		return regmatch



#	def adccmd(self,r1w0,m,p,ch,addr,data,w1w0=0):
#		cmd24=((r1w0&0x1)<<23)+((m&0x1)<<22)+((p&0x1)<<21)+((ch&0x1)<<20)+((addr&0xfff)<<8)+((data&0xff)<<0)
	def adccmd(self,r1w0,mpchaddr,data,w1w0=0):
		cmd24=((r1w0&0x1)<<23)+((mpchaddr&0x7fff)<<8)+((data&0xff)<<0)
		#cmd24=((r1w0&0x1)<<23)+((addr&0x7fff)<<8)+((data&0xff)<<0)
		#print(format(cmd24,'04x'))
		return cmd24
	def adcwrite(self,mpchaddr,data,adca0b1):
		cmd24=self.adccmd(r1w0=0,mpchaddr=mpchaddr,data=data)
		self.spiwrite(cmd24=cmd24,action=0x04 if adca0b1==0 else 0x08)
	def adcread(self,mpchaddr,adca0b1):
		cmd24=self.adccmd(r1w0=1,mpchaddr=mpchaddr,data=0)
		vmsb,vlsb=self.spiread(cmd24=cmd24,action=0x04 if adca0b1==0 else 0x08)
		#print('adcread',hex(vmsb),hex(vlsb))
		return vlsb
	def adcload(self,adca0b1,reglist):
		print('adclod')
		for mpchaddr,data in reglist:
			self.adcwrite(mpchaddr,data,adca0b1)
		print('adclod done')
	def adccheck(self,adca0b1,reglist,printall=False):
		import ads54j60
		checkpass=True
		for mpchaddr,data in reglist:
			if mpchaddr in ads54j60.ctrladdr:
				self.adcwrite(mpchaddr=mpchaddr,adca0b1=adca0b1,data=data)
				print('change ctrl reg %x to %x'%(mpchaddr,data))
			else:
				rdbk=self.adcread(mpchaddr=mpchaddr,adca0b1=adca0b1)
				if (rdbk!=data or printall):
					print('adc regs check addr 0x%x expected to be 0x%x actual rdbk 0x%x '%(mpchaddr,data,rdbk))
					checkpass=False
		return checkpass


	def daccmd(self,r1w0,addr,data):
		cmd24=((r1w0&0x1)<<23)+((addr&0x7f)<<16)+((data&0xffff)<<0)
		return cmd24
	def dacwrite(self,addr,data):
		cmd24=self.daccmd(r1w0=0,addr=addr,data=data)
		self.spiwrite(cmd24=cmd24,action=0x02)
	def dacread(self,addr):
		cmd24=self.daccmd(r1w0=1,addr=addr,data=0)
		vmsb,vlsb=self.spiread(cmd24=cmd24,action=0x02)
		return ((vmsb&0xff)<<8)+(vlsb&0xff)
	def dacload(self,reglist,delay=None):
		print('dac load init')
		for addr,data in reglist:
			self.dacwrite(addr,data)
		#	print('dacload addr %d 0x%x, value %d 0x%x'%(addr,addr,data,data))
			if delay:
				time.sleep(delay)
		print('dac load init done')

	def daccheck(self,reglist):
		checkpass=True
		for addr,data in reglist:
			rdbk=self.dacread(addr)
			if (rdbk==data):
				pass
			else:
				print('dac check','addr',addr,hex(addr),'should be',hex(data),'read back',hex(rdbk))
				checkpass=False
		return checkpass

	def ad7291cmd(self,r1w0,addr,data):
		cmd24=((r1w0&0x1)<<23)+((addr&0x7f)<<16)+((data&0xffff)<<0)
		return cmd24
	def ad7291write(self,addr,data):
		addrdata=((addr&0xff)<<16)+(data&0xffff)
		self.devwrite(devaddr=0x2f,addrdata=addrdata,nack=4)
	def ad7291calc(self,val):
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
	def ad7291read(self,addr):
		addrdata=((addr&0xff))#<<16)
		val=self.devread(devaddr=0x2f,addrdata=addrdata,nack=3)
		return val
	def ad7291readall(self):
		ad7291result={}
		for i in range(8):
			self.ad7291write(addr=0x0,data=((1<<(i+8))+0x80))
			chv,vout=self.ad7291calc(self.ad7291read(0x01))
			ad7291result[chv]=vout
	#		print('%d %4.2f'%(chv,vout))
		cht,temp=self.ad7291calc(self.ad7291read(0x02))
		print(cht,temp)
		ad7291result[cht]=temp
		cha,tavr=self.ad7291calc(self.ad7291read(0x03))
		print(cha,tavr)
		ad7291result[cha]=tavr
		return ad7291result
	def ad7291check(self,goodvalrange=0.15):
		default=numpy.array([0.9,1.15,1.8,1.9,2.6,3.0,3.3,-2.6,35,35])
		minval=default-abs(default*goodvalrange)
		maxval=default+abs(default*goodvalrange)
		minval[8:10]=[0,0]
		maxval[8:10]=[60,60]
		ad7291result=self.ad7291readall()
		ad7291resultval=[ad7291result[i] for i in sorted(list(ad7291result.keys()))]
		print('minimal',''.join(['%6.2f'%i for i in minval]))
		print('default',''.join(['%6.2f'%i for i in default]))
		print('measure',''.join(['%6.2f'%i for i in ad7291resultval]))
		print('maximum',''.join(['%6.2f'%i for i in maxval]))
		print('error  ',''.join(['%6.2f'%i for i in (numpy.array(ad7291resultval)/default-1)]))
		ad7291pass=numpy.zeros(10)
		for index,val in ad7291result.items():
			ad7291pass[index]=(ad7291result[index]<maxval[index]) and (ad7291result[index]>minval[index])
		return ad7291pass


	def axi4lite_readwrite(self,axi,addr,w0r1,wdata):
		# axi for the name of the axi: axifmc[1/2][adc1/2|dac]
		self.write(((axi+'_addr',addr),(axi+'_wdata',wdata),(axi+'_w0r1',w0r1),(axi+'_start',0)))
		if w0r1:
			index=0;
			rdatavalid=self.read((axi+'_rdatavalid',))
			while (not rdatavalid and index<=20):
				rdatavalid=self.read((axi+'_rdatavalid',))
				time.sleep(0.1)
				index=index+1
				print(index)
			if (rdatavalid):
				rdata=self.read(((axi+'_rdata',)))
			else:
				rdata=0
		else:
			rdata=None
			rdatavalid=None
		return (rdata,rdatavalid)
	def axi4lite_read(self,axi,addr):
		return self.axi4lite_readwrite(axi=axi,addr=addr,w0r1=1,wdata=0)
	def axi4lite_write(self,axi,addr,wdata):
		return self.axi4lite_readwrite(axi=axi,addr=addr,w0r1=0,wdata=wdata)
	def jesd204axi_reset(self,axi):
		self.axi4lite_write(axi=axi,addr=0x04,wdata=0x00001)
		reset,resetvalid=self.axi4lite_read(axi,addr=0x04)
		print('reset',reset,resetvalid)
		ireset=0
		while (int(reset)&0x1):
#			   self.axi4lite_write(ser,axi,0x04,0x10000)
			print('reseting',ireset)
			reset,resetvalid=self.axi4lite_read(axi,0x04)
			print('reset',reset)
			time.sleep(0.1)
			ireset=ireset+1
	def reset(self):
		#reset dac
		import cpld
		import lmk04828
		import dac39j84
		print('c_fmc120 reset')
		for (addr,data) in cpld.dacreset:
			self.cpldwrite(addr=addr,data=data)
		for (addr,data) in dac39j84.if4lane:
			self.dacwrite(addr=addr,data=data)
		# reset adc and lmk
		for (addr,data) in cpld.adcreset:
			self.cpldwrite(addr=addr,data=data)

		for (addr,data) in cpld.lmkreset:
			self.cpldwrite(addr=addr,data=data)
		for (addr,data) in lmk04828.softreset:
			self.lmkwrite(addr=addr,data=data)
		print('c_fmc120 reset done')
	def lmk04828load(self,reglist):
		print('lmk04828 load ')
		for addr,data in reglist:
			self.lmkwrite(addr,data)
		print('lmk04828 load done')


if __name__=="__main__":
	vc707=c_vc707()

	if (1):
		print('test, test1',vc707.read(('test','test1','test1','test1','test','test','test','test','test','test','test','test2','test1')))
		time.sleep(0.1)
#	for i in range(20):
	vc707.write((('clk4ratio',5000),))
	vc707.write((('i2cmux_reset_b',0),))
	vc707.write((('i2cmux_reset_b',1),))

	print('prsnt,pgm2c,should  be [0,3], read:',vc707.read(('fmcprsnt','fmcpgm2c')))
	fmcdest=4
	if len(sys.argv)>1:
		fmcdest=int(sys.argv[1])
	vc707.i2cwrite(devaddr=0x74,data=fmcdest);#int(sys.argv[1]))

	print('mux read',[hex(i) for i in vc707.i2cread(devaddr=0x74)])
	print(vc707.read(('clk4ratio',)))
	if 0:
		cpld05=vc707.cpldread(addr=0x05)
		print('cpld05 reading,should be 0x10=16,  read  ',cpld05)
	if 0:
		sfpinfo=vc707.sfpinfo()
		print(sfpinfo,[chr(i) for i in sfpinfo])

	if 0:
		vc707.i2cwrite(devaddr=0x74,data=0x4)
		for page in range(16):
			val=0
			for addr in range(page*16+0,page*16+16):
				vali=vc707.eepromread(addr)
#			print(hex(vali[0]))
				val=(val<<8)+vali
			print(hex(val[0]))

	if 0: # test on single vc707 eeprom write and read
		vc707.i2cwrite(devaddr=0x74,data=0x8)
		vc707.eepromwrite(addr=0,data=0x5c,devaddr=0x54)
		time.sleep(0.1)
		print('eeprom readback, expect to bo 0x5c=%d'%0x5c,vc707.eepromread(addr=0,devaddr=0x54))
	if 0: # test on vc707 eeprom write and read
		for page in range(1):
			for addr in range(page*16+0,page*16+16):
				vc707.eepromwrite(addr=addr,data=addr,devaddr=0x54)
#				print('write page ',page,'addr ',addr)
		time.sleep(0.1)
		for page in range(1):
			val=0
			for addr in range(page*16+0,page*16+16):
				val=(val<<8)+vc707.eepromread(addr=addr,devaddr=0x54)
#				print('read page ',page,'addr ',addr)
			print(val)
	if 0: # test on fmc120 spi
		#print('mux read',hex(vc707.i2cread(devaddr=0x74)))
		vc707.i2cwrite(devaddr=0x74,data=fmcdest)
		vc707.cpldwrite(addr=0x02,data=0x00)
		vc707.cpldwrite(addr=0x02,data=0x20)
		vc707.dacwrite(addr=0x02,data=0x2082)
#	   lmkwrite(addr=0x15d,data=0x73)
		vc707.cpldwrite(addr=0x03,data=0x00)
		vc707.cpldwrite(addr=0x03,data=0x07)
		vc707.cpldwrite(addr=0x03,data=0x18)
		vc707.cpldwrite(addr=0x01,data=0xf0)
		vc707.lmkwrite(addr=0,data=0x90)
		vc707.lmkwrite(addr=0,data=0x10)
#	   lmkwrite(addr=0x146,data=0x00)
#	   lmkwrite(addr=0x147,data=0x10)
# for 148		vc707.lmkwrite(addr=0x148,data=0x33)
#	   lmkwrite(addr=0x149,data=0x00)
#	   lmkwrite(addr=0x14a,data=0x00)
#	   lmkwrite(addr=0x14b,data=0x05)
	if 0: # test on fmc120 spi
#	   print('dacread',hex(int(dacread(addr=0x0))))
#	   dacwrite(addr=0x0,data=0x18)
		vc707.i2cwrite(devaddr=0x74,data=fmcdest)
		#vc707.lmkwrite(addr=0x10,data=0x1010)
#	   print('lmkread',hex(3),hex(int(lmkread(addr=3))));
#	   print('dacread',hex(int(dacread(addr=0x0))))
		for addr in range(16):
	#   for addr in [0,3]:#range(3):
			adca=vc707.adcread(addr=addr,adca0b1=0)[0]
			lmk=vc707.lmkread(addr=addr)[0]
			dac=vc707.dacread(addr=addr)[0]
			adcb=vc707.adcread(addr=addr,adca0b1=1)[0]
			print([hex(i) for i in [addr,lmk,dac,adca,adcb]])

	if 0:
		vc707.ad7291write(addr=0x0,data=0x0002)
		time.sleep(1)
		for i in range(8):
			vc707.ad7291write(addr=0x0,data=((1<<(i+8))+0x80))
			chv,vout=vc707.ad7291calc(vc707.ad7291read(0x01))
			cht,temp=vc707.ad7291calc(vc707.ad7291read(0x02))
			cha,tavr=vc707.ad7291calc(vc707.ad7291read(0x03))
			print('%d %3.2f %d %4.2f %d %4.2f'%(chv,vout,cht,temp,cha,tavr))
	if 0:
		vc707.i2cwrite(devaddr=0x74,data=0x1)
		for addr in [7,8,9,10,11,12,13,14,15,16,17,18,135,137]:
			val=vc707.si570read(addr)
			print(addr,val)
			#print(format(addr,'3d'),format(addr,'02x'),format(val,'02x'))
		addr=18
		vc707.si570write(addr,0xff)
		val=vc707.si570read(addr)
		print(addr,val)
#		print(format(addr,'3d'),format(addr,'02x'),format(val,'02x'))
	if 0:
		#   lmkwrite(addr=0,data=0x80);
		#lmkwrite(addr=0,data=0x10);
		#lmkwrite(addr=2,data=0x00);
		vc707.i2cwrite(devaddr=0x74,data=fmcdest)
		import lmk04828
#		with open('lmkinit.json') as f:
#			lmkinit=json.load(f)
		#for addr in [0,2,3,4,5,6,0xc,0xd,0x100,0x101,0x103,0x104]:
		for addr,data in lmk04828.default:
			rdbk=vc707.lmkread(addr=addr)[0]
			if data==rdbk:
				pass
			else:
				print('lmk init addr %x data %x %d rdbk %x %d'%(addr,data,data,rdbk,rdbk))
	#	for addr,data in lmk04828.default:
	#		vc707.lmkwrite(addr=addr,data=data)

		for addr,data in lmk04828.init:
			vc707.lmkwrite(addr=addr,data=data)
	if 0:
		for addr in [1,2,3,4,5,6,7,8,0x0e,0x0f]:
			print('cpldread',hex(addr),hex(int(vc707.cpldread(addr=addr)[0])))

	if 0:
		import ads54j60
		vc707.i2cwrite(devaddr=0x74,data=fmcdest);#int(sys.argv[1]))
#		vc707.lmkwrite(addr=0x10,data=0x1010)
		for addr,val in ads54j60.softreset:
			m=(addr>>14)&0x1
			p=(addr>>13)&0x1
			ch=(addr>>12)&0x1
			ad=(addr>>0)&0xfff
			vc707.adcwrite(addr=addr,m=m,p=p,ch=ch,data=val,adca0b1=1)
		for addr,val in ads54j60.default:
			r1w0=(addr>>15)&0x1
			m=(addr>>14)&0x1
			p=(addr>>13)&0x1
			ch=(addr>>12)&0x1
			ad=(addr>>0)&0xfff
#		   print(hex(addr),m,p,ch,val)
#		   before=adcread(addr=addr,m=m,p=p,ch=ch,adca0b1=1)
#			if addr in ads54j60.ctrladdr:
			if (r1w0==0):
				vc707.adcwrite(addr=addr,m=m,p=p,ch=ch,data=val,adca0b1=1)
			else:
				after=vc707.adcread(addr=addr,m=m,p=p,ch=ch,adca0b1=1)[0]
				if (after==val):
					pass
				else:
					print('addr',hex(addr),'val',format(val,'x'),'read',format(after,'x'))

	if 0:
		import dac39j84
		vc707.i2cwrite(devaddr=0x74,data=fmcdest);#int(sys.argv[1]))
		#vc707.lmkwrite(addr=0x10,data=0x1010)
		for addr,val in dac39j84.default:
			rdbk=vc707.dacread(addr=addr)[0]
			if val==rdbk:
				pass
			else:
				print('dacread',hex(addr),'val',hex(val),'rdbk',hex(int(rdbk)),val==rdbk)

#		for addr in range(0x6d):
		#for addr in [0x6d]:
#			print('dacread',hex(addr),hex(int(vc707.dacread(addr=addr)[0])))
	if 0:
		vc707.jesd204axi_reset('axifmc1adc0')
	if 0:
		vc707.axi4lite_write(axi='axifmc1adc0',addr=0x04,wdata=0x00001)
		for addr in range(0,0x3c,4):
			data,valid=vc707.axi4lite_read('axifmc1adc0',addr=addr)
			print('read axi addr',addr,hex(addr),'data',data,hex(data))
#	vc707.write((('axifmc1adc0_w0r1',1),('axifmc1adc0_start',1),))
#	while 1:
#		rdatavalid=vc707.read(('axifmc1adc0_rdatavalid',))
#		print(rdatavalid)
#		time.sleep(0.1)
	if 0:
		import axiinit
		axiinsts={2:['axifmc1adc0','axifmc1adc1','axifmc1dac'],4:['axifmc2adc0','axifmc2adc1','axifmc2dac']}
		for axi in axiinsts[fmcdest]:
			for w0r1,addr,data in axiinit.axiinit:
				if w0r1:
					data,valid=vc707.axi4lite_read(axi,addr=addr)
				else:
					vc707.axi4lite_write(axi,addr=addr,wdata=data)
				time.sleep(0.1)
				print('write axi',addr,data)
			resetval,valid=vc707.axi4lite_read(axi,addr=4)
			print('read axi %s  addr'%axi,addr,'resetval',resetval,hex(resetval))
			vc707.jesd204axi_reset(axi)
		#while resetval:
		#	addr=4;	resetval,valid=vc707.axi4lite_read('axifmc1adc0',addr=addr)
		#	print('read axi addr',addr,'resetval',resetval,hex(resetval))
		#	addr=0;	readdata,valid=vc707.axi4lite_read('axifmc1adc0',addr=addr)
		#	print('read axi addr',addr,'readdata',readdata,hex(readdata))
		#	time.sleep(0.1)
#	for addr in range(4):
#		print('addr',addr,vc707.axi4lite_read('axifmc1adc0',addr=addr))
	if 0:
		freqregs=['freq_lb'
	,'freq_sgmiiclk'
	,'freq_sma_mgt_refclk'
	,'freq_si5324_out_c'
	,'freq_pcie_clk_qo'
	,'freq_user_clock'
	,'freq_fmc1_llmk_dclkout_2'
	,'freq_fmc1_llmk_sclkout_3'
	,'freq_fmc1_lmk_dclk8_m2c_to_fpga'
	,'freq_fmc1_lmk_dclk10_m2c_to_fpga'
	,'freq_fmc2_llmk_dclkout_2'
	,'freq_fmc2_llmk_sclkout_3'
	,'freq_fmc2_lmk_dclk8_m2c_to_fpga'
	,'freq_fmc2_lmk_dclk10_m2c_to_fpga'
	,'freq_rxusrclk_sfp'
	,'freq_txusrclk_sfp'
	]
		freqs=(numpy.array(vc707.read(freqregs))*125.0/2**24)
		freqdict={}
		for index,freq in enumerate(freqs):
			freqdict[freqregs[index]]=freqs[index]
		print(freqdict)
