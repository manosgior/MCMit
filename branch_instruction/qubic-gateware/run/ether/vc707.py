import json
import numpy
import time
import sys
import devcom
from ether import c_ether
from uartlb import c_uartlb
from regmap import c_regmap
from fmc120 import c_fmc120
import si570
class c_vc707:
	def __init__(self,regmap,uartregmap=None):
		self.regmap=regmap
		self.uartregmap=uartregmap
		self.i2csw={'si570':1,'fmc1':2,'fmc2':4,'eeprom':8,'sfp':16,'hdmi':32,'ddr3':64,'si5324':128}
		self.uarti2c=False
		self.read=regmap.read
		self.write=regmap.write
		if uartregmap:
			self.setuarti2c(self.uarti2c)
	def setuarti2c(self,uarti2c=True):
		#		print('setuarti2c',uarti2c)
		self.uarti2c=uarti2c
		self.uartregmap.write((('uarti2c',1 if self.uarti2c else 0),))
	def readfori2c(self,names):
		if self.uarti2c:
			vals=self.uartregmap.read(names)
		else:
			vals=self.regmap.read(names)
		return vals
	def writefori2c(self,namedatalist):
		#print('write for i2c',self.uarti2c)
		if self.uarti2c:
			res=self.uartregmap.write(namedatalist)
		else:
			res=self.regmap.write(namedatalist)
		return res

#	def write(self,namedatalist):
#		return self.regmap.write(namedatalist)
#	def read(self,names):
#		vals=self.regmap.read(names)
#		return vals
	def i2cenable(self,clk4ratio=5000,enable=True):
		print('i2cenable',clk4ratio)
		self.writefori2c((('clk4ratio',clk4ratio),('i2cmux_reset_b',1 if enable else 0)))
	def i2creadwrite(self,devaddr,r1w0,data,nack,stop=1,regs={'i2cdatatx':'i2cdatatx','i2cstart':'i2cstart','i2cdatarx':'i2cdatarx','i2crxvalid':'i2crxvalid','i2cclk4ratio':'clk4ratio','i2cmux_reset_b':'i2cmux_reset_b'}):
		#print('i2creadwrite r1w0 data',r1w0,hex(data))
		r0=((devaddr&0x7f)<<25)+((r1w0&1)<<24)+(data&0xffffff)
		self.writefori2c(((regs['i2cdatatx'],r0),(regs['i2cstart'],(nack&0xf)+((stop&0x1)<<4))))
		#print(",{1'h%01x,4'h%01x,32'h%08x}"%(stop&0x1,nack&0xf,r0))#,hex((nack&0xf)+((stop&1)<<4)+(r0<<5)))
		#print(regs['i2crxvalid'])
		valid=self.readfori2c((regs['i2crxvalid'],))
		index=0
		while (not valid & index<20):
			#time.sleep(0.1)
	#	if (1):
			valid=self.readfori2c((regs['i2crxvalid'],))
#			print('valid',valid)
			time.sleep(0.1)
			index=index+1
		self.readfori2c((regs['i2cdatarx'],))
		regname=regs['i2cdatarx']

		return self.regmap.getregval([regname])
	def i2cwrite(self,devaddr,data,nack=2,stop=1):
		#print('i2cwrite',devaddr,data)
		data=data<<((4-nack)*8)
		self.i2creadwrite(devaddr=devaddr&0x7f,r1w0=0,data=data,nack=nack,stop=stop)
	def i2cread(self,devaddr,nack=2):
		v=self.i2creadwrite(devaddr=devaddr&0x7f,r1w0=1,data=0,nack=nack)
		#print('i2cread',hex(int(v)))
		return v

	def i2cswitch(self,i2cdest):
		#	self.i2cenable()
		#print('i2cswitch i2cdest',i2cdest)
		if i2cdest in self.i2csw.keys():
			self.i2cwrite(devaddr=0x74,data=self.i2csw[i2cdest]);
			time.sleep(1)
#			print('readback',hex(self.i2cread(devaddr=0x74)));
		else:
			print('i2cswitch options: %s'%'|'.join(self.i2csw.keys()))

	def devwrite(self,devaddr,addrdata,nack=2,stop=1):
		self.i2cwrite(devaddr=devaddr&0x7f,data=addrdata,nack=nack,stop=stop)
	def devread(self,devaddr,addrdata,nack=2):
		self.devwrite(devaddr=devaddr,addrdata=addrdata,nack=2,stop=0)
		v=self.i2cread(devaddr=devaddr,nack=nack)
		return v

	def eepromcmd(self,r1w0,addr,data):
		cmd24=((r1w0&0x1)<<23)+((addr&0x7f)<<16)+((data&0xffff)<<0)
		return cmd24
	def eepromwrite(self,addr,data,devaddr=0x50):
		addrdata=((addr&0xff)<<8)+(data)
		self.devwrite(devaddr=devaddr,addrdata=addrdata,nack=3)
	def eepromread(self,addr,devaddr=0x50):
		addrdata=((addr&0xff))#<<16)
		val=self.devread(devaddr=devaddr,addrdata=addrdata,nack=2)
		return val&0xff
	def sfpcmd(self,r1w0,addr,data):
		cmd24=((r1w0&0x1)<<23)+((addr&0x7f)<<16)+((data&0xffff)<<0)
		return cmd24
	def sfpwrite(self,addr,data,devaddr=0x50):
		addrdata=((addr&0xff)<<8)+(data)
		self.devwrite(devaddr=devaddr,addrdata=addrdata,nack=3)
	def sfpread(self,addr,devaddr=0x50):
		addrdata=((addr&0xff))#<<16)
		val=self.devread(devaddr=devaddr,addrdata=addrdata,nack=2)
		return val&0xff
	def si570write(self,addr,data,devaddr=0x5d):
		addrdata=((addr&0xff)<<8)+(data)
		self.devwrite(devaddr=devaddr,addrdata=addrdata,nack=3)
	def si570read(self,addr,devaddr=0x5d):
		addrdata=((addr&0xff))#<<16)
		val=self.devread(devaddr=devaddr,addrdata=addrdata,nack=2)
		return val&0xff
	def si570reset(self):
		for addr,data in si570.reset:
			self.si570write(addr,data)
		time.sleep(1)
	def si570readinit(self):
		print('si570readinit')
		self.si570reset()
		time.sleep(1)
		self.si570readallregs()
		return self.si570readallregs()
	def si570readallregs(self):
		si570regs={}
		for addr,data in si570.default:
			si570regs[addr]=self.si570read(addr)
		return si570regs
	def si570setfreq(self,freq):
		print('start si570setfreq')
		initreg=self.si570readinit()
		print('initreg')
		print({a:format(d,'02x') for a,d in initreg.items()})
		newregs=si570.fset(freq,si570.fxtal(initreg))
		print(newregs)
		print('si570 set freq to %f'%freq)
		if len(newregs)>0:
			newreg=newregs[0]
			print(newreg)
			self.si570write(addr=137,data=initreg[137]|0x10)
			for addr,data in newreg.items():
				self.si570write(addr=addr,data=data)
			self.si570write(addr=137,data=0*initreg[137]&0xef)
		else:
			print('failed find setting')
		time.sleep(1)
		print('si570 set freq done')
	def si570load(self,reglist):
		for addr,data in reglist:
			#			print('si570load',addr,data)
			self.si570write(addr,data)

	def si5324write(self,addr,data,devaddr=0x68):
		addrdata=((addr&0xff)<<8)+(data)
		self.devwrite(devaddr=devaddr,addrdata=addrdata,nack=3)
	def si5324read(self,addr,devaddr=0x68):
		addrdata=((addr&0xff))#<<16)
		val=self.devread(devaddr=devaddr,addrdata=addrdata,nack=2)
		return val&0xff
	def si5324load(self,reglist,delay=1):
		for addr,data in reglist:
			self.si5324write(addr,data)
			if delay:
				time.sleep(delay)
	def sfpinfo(self):
		#		self.i2cwrite(devaddr=0x74,data=16)#0x04)
		self.i2cswitch('sfp')
		#print('mux read',self.i2cread(devaddr=0x74))
		vals=[]
		for addr in range(128):
			val=self.sfpread(addr=addr)
			vals.append(val)
#			print('sfp addr ',hex(addr),addr,'value',hex(val),chr(val))
		return vals


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
	def mdiowrite(self,addr,data,devaddr=0b00111,uart=True):
		mdiocmd=(0<<26)+((devaddr&0x1f)<<21)+((int(addr)&0x1f)<<16)+((int(data)&0xffff))
		#print('mdiocmd',format(mdiocmd,'8x'))
		if uart:
			self.uartregmap.write((('mdiodatatx',mdiocmd),('mdiostart',0)))
		else:
			self.regmap.write((('mdiodatatx',mdiocmd),('mdiostart',0)))
	def mdioread(self,addr,devaddr=0b00111,uart=True):
		mdiocmd=(1<<26)+((devaddr&0x1f)<<21)+((int(addr)&0x1f)<<16)+((0&0xffff)<<0)
	#	print('mdiocmd',format(mdiocmd,'8x'))
		if uart:
			self.uartregmap.write((('mdiodatatx',mdiocmd),('mdiostart',0)))
#			time.sleep(0.01)
			v=self.uartregmap.read(('mdiodatarx',))
		else:
			self.regmap.write((('mdiodatatx',mdiocmd),('mdiostart',0)))
#			time.sleep(0.01)
			v=self.regmap.read(('mdiodatarx',))
		(op,devaddr,regaddr,data)=self.mdiodecode(v)
		return int(data)
	def mdiodecode(self,v):
		vint=int(v)
		st=(vint>>30)&0x3
		op=(vint>>28)&0x3
		devaddr=(vint>>23)&0x1f
		regaddr=(vint>>18)&0x1f
		tr=(vint>>16)&0x3
		data=vint&0xffff
		return (op,devaddr,regaddr,data)



if __name__=="__main__":
	vc707=c_qubichw()

	if (1):
		print('test, test1',vc707.read(('test','test1','test1','test1','test','test','test','test','test','test','test','test2','test1')))
		time.sleep(0.1)
#	for i in range(20):
	vc707.write((('clk4ratio',5000),))
	vc707.write((('i2cmux_reset_b',0),))
	vc707.write((('i2cmux_reset_b',1),))

	print('prsnt,pgm2c,should  be [0,3], read:',vc707.read(('fmcprsnt','fmcpgm2c')))
	dest=vc707.fmc120_1
	fmcdest=dest.i2cid
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
# for 148		vc707.lmkwrite(addr=0x148,data=0x33)

#	   lmkwrite(addr=0x146,data=0x00)
#	   lmkwrite(addr=0x147,data=0x10)
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
	if 1:
		#   lmkwrite(addr=0,data=0x80);
		#lmkwrite(addr=0,data=0x10);
		#lmkwrite(addr=2,data=0x00);
		vc707.i2cwrite(devaddr=0x74,data=fmcdest)
		import lmk04828
#		with open('lmkinit.json') as f:
#			lmkinit=json.load(f)
		#for addr in [0,2,3,4,5,6,0xc,0xd,0x100,0x101,0x103,0x104]:
		for addr,data in lmk04828.default:
			rdbk=dest.lmkread(addr=addr)[0]
			if data==rdbk:
				pass
			else:
				print('lmk init addr %x data %x %d rdbk %x %d'%(addr,data,data,rdbk,rdbk))
	#	for addr,data in lmk04828.default:
	#		vc707.lmkwrite(addr=addr,data=data)

		for addr,data in lmk04828.init:
			dest.lmkwrite(addr=addr,data=data)
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
		import jesdaxi 
		axiinsts={2:['axifmc1adc0','axifmc1adc1','axifmc1dac'],4:['axifmc2adc0','axifmc2adc1','axifmc2dac']}
		for axi in axiinsts[fmcdest]:
			for w0r1,addr,data in jesdaxi.default:
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
	if 1:
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
