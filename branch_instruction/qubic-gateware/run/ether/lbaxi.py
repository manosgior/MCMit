import time
import sys
import devcom
from ether import c_ether
from uartlb import c_uartlb
from regmap import c_regmap
from fmc120 import c_fmc120
from vc707 import c_vc707
class c_lbaxi():
	def __init__(self,read,write):
		self.read=read
		self.write=write
		pass
	def axi4lite_readwrite(self,axiprefix,addr,w0r1,wdata):
		# axi for the name of the axi: axifmc[1/2][adc1/2|dac]
		self.write(((axiprefix+'_addr',addr),(axiprefix+'_wdata',wdata),(axiprefix+'_w0r1',w0r1),(axiprefix+'_start',0)))
		if w0r1:
			index=0;
			rdatavalid=self.read((axiprefix+'_rdatavalid',))
			while (not rdatavalid and index<=20):
				rdatavalid=self.read((axiprefix+'_rdatavalid',))
				time.sleep(0.1)
				index=index+1
				print(index)
			if (rdatavalid):
				rdata=self.read(((axiprefix+'_rdata',)))
			else:
				rdata=0
		else:
			rdata=None
			rdatavalid=None
		return (rdata,rdatavalid)
	def axi4lite_read(self,axiprefix,addr):
		return self.axi4lite_readwrite(axiprefix=axiprefix,addr=addr,w0r1=1,wdata=0)
	def axi4lite_write(self,axiprefix,addr,wdata):
		return self.axi4lite_readwrite(axiprefix=axiprefix,addr=addr,w0r1=0,wdata=wdata)
	def axi4lite_load(self,axiprefix,reglist):
		for addr,data in reglist:
			rdbk=self.axi4lite_write(axiprefix=axiprefix,addr=addr,wdata=data)[0]
	def axi4lite_check(self,axiprefix,reglist):
		print('axi4lite %s check begin'%axiprefix)
		checkpass=True
		for addr,data in reglist:
			rdbk=self.axi4lite_read(axiprefix=axiprefix,addr=addr)[0]
			if rdbk!=data:
				print('axi check','addr',addr,hex(addr),'should be',hex(data),'read back',hex(rdbk))

				checkpass=False
		print('axi4lite check end')
		return checkpass
	def jesd204axi_reset(self,axiprefix):
		self.axi4lite_write(axiprefix=axiprefix,addr=0x04,wdata=0x00001)
		reset,resetvalid=self.axi4lite_read(axiprefix,addr=0x04)
		print('reset',reset,resetvalid)
		ireset=0
		while (int(reset)&0x1):
#			   self.axi4lite_write(ser,axi,0x04,0x10000)
			print('reseting',axiprefix,ireset)
			reset,resetvalid=self.axi4lite_read(axiprefix,0x04)
			print('reset',reset)
			time.sleep(0.1)
			ireset=ireset+1
