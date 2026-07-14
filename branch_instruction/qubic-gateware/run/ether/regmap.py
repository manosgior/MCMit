from register import c_register
#from mem_gateway import c_mem_gateway
from ether import c_ether
from udplb import c_udplb
import time
import json
import numpy
try:
	basestring
except NameError:
	basestring = str
class c_regmap():
	DWIDTH=32
	def __init__(self,interface,regmappath='regmap.json',wavegrppath='wavegrp.json',run=True,timeout=None):
		self.interface=interface 
		self.regs={}
		self.regindex={}
		with open(regmappath) as jsonfile:
			self.registers=json.load(jsonfile)
		with open(wavegrppath) as jsonfile:
			self.wavegrp=json.load(jsonfile)
		for name,prop in self.registers.items():
			self.regs[name]=c_register(**dict({'name':name,'DWIDTH':self.DWIDTH},**prop))
			for addr in self.regs[name].readaddr():
				self.regindex[addr]=self.regs[name]
		for name,wavegrp in self.wavegrp.items():
			self.regs[name].setwaveregs(wavegrp)
		self.timeout=timeout


	def wavecheckreadallreset(self,wavename):
		print('wavecheckreadallreset')
		wave=self.regs[wavename]
		if not wave.resetafter:
			self.write(((wave.reset_reg,wave.reset_word),))
		status=(self.read(((wave.status_reg),))>>wave.status_bit)&0x1
		status=(self.read(((wave.status_reg),))>>wave.status_bit)&0x1
		index=0
		while (not status) and (self.timeout is not None and index<timeout):
			status=(self.read(((wave.status_reg),))>>wave.status_bit)&0x1
			index=index+1
		if self.timeout and index==timeout:
			print('timeout on ',wavename)
		else:
			#			print('wavecheckreadallreset',index,status)
			self.readregs(((wavename),))
		if wave.resetafter:
			self.write(((wave.reset_reg,wave.reset_word),))

#	def readandwrite(self,opnamedatalist,addr=None):
#		#		print opnamedatalist
#		cadlist=numpy.array((len(opnamedatalist),3))
#		for i,(o,n,d) in enumerate(opnamedatalist):
#			cadlist[i][0]=self.cmds[o]
#			cmdlist[i][1]=
#		alist=[]
#		dlist=[]
#		wlist=[]
#		for (o,n,d) in opnamedatalist:
#			if o=='write':
#				(a,d,w)=self.writeadw(((n,d),))
#			elif o=='read':
#				(a,d,w)=self.readadw(((n),))
#			alist.extend(a)
#			dlist.extend(d)
#			wlist.extend(w)
#		return self.rawadw(alist,dlist,wlist)
	def write(self,namedatalist):
		#(alist,dlist,wlist)=self.writeadw(namedatalist)
#		print 'write',alist,dlist,wlist
#		print len(namedatalist)
		regs=[self.regs[name] for name,data in namedatalist]
		reglengths=numpy.array([r.addr_length for r in regs])
		regaddrlen=sum(reglengths)
		cad=numpy.zeros((regaddrlen,3))
		cad[:,0]=regaddrlen*[self.interface.cmds['write']]
		cnt=0
		for index,(regname,val) in enumerate(namedatalist):
			#print(index,regname,val)
			cad[cnt:cnt+reglengths[index],1]=self.regs[regname].readaddr()
			cad[cnt:cnt+reglengths[index],2]=self.regs[regname].writeval(val)
			cnt=cnt+reglengths[index]
	#	print('write cad',cad)
		return self.rawadw(cad)
	#cadarray=numpy.zeros((cmdlen,3))
	#	cadarray[:,0]=cmdlen*[self.cmds['write']]
	#	cadarray[:,1:3]=numpy.array([(r if isinstance(r,basestring) else r.name,v) for r,v in namedatalist])
	#	return self.writeregs(nvlist)
	def writeregs(self,namedatalist):
		adlist=[]
		for name,data in namedatalist:
			offset=(None if len(item)==2 else item[2])
			adlist.append(self.regs[name].write(data,offset))
			###if name=='period_dac0':###
				###print '((((((((((((name==period_dac0))))))))))))',data###
			###if name=='start':###
				###print '((((((((((((name==start))))))))))))',data###
		#print adlist
		(alist,dlist)=zip(*sum(adlist,[]))
		wlist=len(alist)*[1]
		return self.rawadw(alist,dlist,wlist)
	def readregs(self,names):
		regs=[self.regs[name] for name in names]
		reglengths=numpy.array([r.addr_length for r in regs])
		regaddrlen=sum(reglengths)
		cad=numpy.zeros((regaddrlen,3))
		cad[:,0]=regaddrlen*[self.interface.cmds['read']]
		cad[:,2]=numpy.zeros(regaddrlen)
		cnt=0
		#print(reglengths)
		for index,r in enumerate(regs):

			cad[cnt:cnt+reglengths[index],1]=r.readaddr()
			cnt=cnt+reglengths[index]
#		print(names)
#		print(cad)
		return self.rawadw(cad)
	def read(self,names):
		namelists=[]
		reglist=[]
		wavelist=[]
		reglist = [n for n in names if n not in self.wavegrp]
		wavelist = [n for n in names if n in self.wavegrp]
		if reglist:
			self.readregs(reglist)
		if wavelist:
			#print('0 step in read^^^^^^^^',wavelist)###
			for wavename in wavelist:
				self.wavecheckreadallreset(wavename)
		return self.getregval(names)
	def rawadw(self,cadlist):
		result=self.interface.readwrite(cadlist)
		cad=self.interface.parse_readvalue(result)
#		print('regmap rawadw cad',cad)
#		print('regmap rawadw cad hex',[[hex(i) for i in l] for l in cad])
		#print 'rawadw',len(alist),len(result),len(value)
		self.updatereadvalue(cad)
		return cad
	def updatereadvalue(self,cad):
		#result=[]
		for cmd,addr,val in cad:
			if addr in self.regindex and cmd==self.interface.cmds['read']:
				self.regindex[addr].setvalue(val,addr)
		#		result.append((self.regindex[addr].name,self.regindex[addr].getvalue()))
		#return result
#		print result
	def getregval(self,names):
		vals=numpy.array([self.regs[name].getvalue() for name in names])
		if (len(vals))==1:
			vals=vals[0]
		return vals
	def readregval(self,name):
		self.read(names=[name])

if __name__=="__main__":
	interface=c_udplb(interface=c_ether('192.168.1.224',port=0xd003))
	regmap=c_regmap(interface)
	import sys
	numpy.set_printoptions(formatter={'int':hex})
	print('write',regmap.write((('test',int(sys.argv[1],0)),('test2',0xdeadbaaf),('err',0))))
	print('write2',regmap.write((('test',int(sys.argv[1],0)),('test2',0xdeadbaaf),('err',0))))
	print('write3',regmap.write((('test',int(sys.argv[1],0)**2),)))
	regs=['test','test2','test1']
	print('read',regmap.read(regs))
	print('read2',numpy.array(regmap.getregval(regs)))
	bufreadtest=regmap.read((('bufreadtest'),))
	diff=numpy.diff(bufreadtest)
	diff2=numpy.diff(bufreadtest[2:-2])
	print('read buf',bufreadtest)
	print('read',max(diff),min(diff))
	print('read2',max(diff2),min(diff2))
	from matplotlib import pyplot
	pyplot.plot(bufreadtest)
	pyplot.show()
#	print('getval',[hex(i) for i in [regmap.getregval(i) for i in regs]])
