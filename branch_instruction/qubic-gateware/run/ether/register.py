import numpy
from address import c_address
#{"banyan_data": {"access": "r","addr_width": 13,"base_addr": "0x10000","data_width": 32,"sign": "unsigned"}
#	def __init__(self,base,width=0,totalwidth=32):
class c_register:
	def __init__(self,access,base_addr,addr_width,data_width,sign,name,DWIDTH=32,init=0):
		self.DWIDTH=DWIDTH
		self.name=name
		self.access=access
		self.base_addr=int(str(base_addr),0)
		self.addr_width=int(str(addr_width),0)
		self.data_width=int(str(data_width),0)
		self.msb=self.data_width-1
		self.lsb=0
		self.addrperdata=(int(numpy.ceil(self.data_width*1.0/self.DWIDTH)))
		self.addrwidthperdata=int(numpy.ceil(numpy.log2(self.addrperdata)))
		self.addr_length=(1<<self.addr_width)*self.addrperdata
		self.dmask1=self.mask1(self.data_width)
		self.dmask0=[self.dmask1^(self.mask1(self.DWIDTH)<<(i*self.DWIDTH)) for i in range(self.addrperdata)]
		self.amaskin=self.mask1(self.addrwidthperdata)
		self.amaskindex1=self.mask1(self.addrwidthperdata)
		self.sign=True if sign=='signed' else False
		self.address=c_address(base=self.base_addr,length=self.addr_length)
		self.value=(1<<self.addr_width)*[0]
		self.DWIDTHmask1=(1<<self.DWIDTH)-1
	def setwaveregs(self,wavegrp):
		if 'status' in wavegrp:
			if 'name' in wavegrp['status']:
				self.status_reg=wavegrp['status']['name']
			if 'bit' in wavegrp['status']:
				self.status_bit=wavegrp['status']['bit']
		if 'reset' in wavegrp:
			self.reset_word=0
			if 'name' in wavegrp['reset']:
				self.reset_reg=wavegrp['reset']['name']
			if 'bit' in wavegrp['reset']:
				self.reset_bit=wavegrp['reset']['bit']
		if 'resetafter' in wavegrp:
			self.resetafter=wavegrp['resetafter']
		if 'flip' in wavegrp:
			self.flip_word=0
			if 'name' in wavegrp['flip']:
				self.flip_reg=wavegrp['flip']['name']
			if 'bit' in wavegrp['flip']:
				self.flip_bit=wavegrp['flip']['bit']
		if 'flipafter' in wavegrp:
			self.flipafter=wavegrp['flipafter']
	def readaddr(self):
		#		print 'register readaddr',self.name,self.base_addr,offset
		return range(self.base_addr,self.base_addr+self.addr_length)
	def writeval(self,val):
		if self.addr_length==1:
			valarray=val
		else:
			valarray=numpy.zeros(self.addr_length)
			valsr=val
			for i in self.addr_length:
				valarray[i]=valsr&self.dmask1
				valsr=valsr>>self.DWIDTH
		return valarray

	def write(self,valuein,offset=None):
		#print 'register write',self.name,'len%d'%len(valuein) if isinstance(valuein,list) else 'val %d'%valuein
		#print 'c_register write valuein',valuein
		#print 'offset',offset
		#if (isinstance(valuein,int) or isinstance(valuein,long) or isinstance(valuein,float)) and self.addr_width==0 and offset==None:
		if (isinstance(valuein,int) or isinstance(valuein,float)) and self.addr_width==0 and offset==None:
			adlist=[(self.base_addr,valuein)]
		else:
			#print valuein,self.addr_width,len(valuein),(1<<self.addr_width)
			if len(valuein)==(1<<self.addr_width):
				#		alist=range(self.base_addr,self.base_addr+self.addr_length)
				#dlist=sum([[((val&self.dmask1)>>(indata*self.DWIDTH))&self.DWIDTHmask1 for indata in range(self.addrperdata)] for val in valuein],[])
				#adlist=zip(alist,dlist)
				#adlist=sum([self.adlist(val,addr) for (addr,val) in zip(range((1<<self.addr_width)),valuein)],[])
				adlist=[]
				for index in range(len(valuein)):
					adlist.extend(self.adlist(valuein[index],index))
				#print 'sel1',adlist
			elif len(valuein)<(1<<self.addr_width):
				#adlist=sum([self.adlist(val,addr) for (addr,val) in zip(range(len(valuein)),valuein)],[])
				adlist=[]
				for index in range(len(valuein)):
					adlist.extend(self.adlist(valuein[index],index))
			elif offset!=None and len(offset)==len(valuein):
				adlist=sum([self.adlist(val,addr) for (addr,val) in zip(offset,valuein)],[])
			else:
				print("ERROR in c_register write:",self.name,valuein,offset,len(valuein),(1<<self.addr_width))
		return adlist
	def adlist(self,val,offset):
		alist=range(self.base_addr+offset*self.addrperdata,self.base_addr+offset*self.addrperdata+self.addrperdata)
		dlist=[((val&self.dmask1)>>(indata*self.DWIDTH))&self.DWIDTHmask1 for indata in range(self.addrperdata)]
		adlist=zip(alist,dlist)
		return adlist
	def setvalue(self,val,addr=None):
		if addr:
			offset=addr-self.base_addr
		else:
			offset=0
		index=offset>>self.addrwidthperdata
		inaddr=offset&self.amaskin
		val=((self.value[index]&self.dmask0[inaddr])|(val<<(inaddr)*self.DWIDTH))&self.dmask1
		if self.sign:
			val=val-2**self.data_width if (val>>(self.data_width-1)) else val
		self.value[index]=val
		return self.value[index]
	def getvalue(self):
		return self.value[0] if len(self.value)==1 else self.value 
	def mask1(self,width):
		return ((1<<width)-1)
				#return (~((1<<((numofword+1)*self.DWIDTH))-1)&(~((1<<(numofword*self.DWIDTH))-1)))
#	def __hash__(self):
#		return hash(self.name)
#	def __repr__(self):
#		return self.name

if __name__=="__main__":
	regs={}
	regindex={}
	regmap={"banyan_data": {"access": "r","addr_width": 13,"base_addr": "0x10000","data_width": 60,"sign": "unsigned"},"U2_dout":{"access":"r", "addr_width": 0,"base_addr":44,"data_width":64,"sign":"unsigned"}}
	for name in regmap:
		regs[name]=c_register(**dict({'name':name},**regmap[name]))
		regindex[regs[name].address]=regs[name]
	print(hex(regs['banyan_data'].setvalue(0x12340922,0x1c)))
	print(hex(regs['banyan_data'].setvalue(0x98754321,0x1d)))
	print(hex(regs['banyan_data'].setvalue(0xdeadbeef,0x1e)))
	print(hex(regs['banyan_data'].setvalue(0xabcdef12,0x1f)))
	print(hex(regs['banyan_data'].mask1(1)))
	print(hex(regs['banyan_data'].mask1(2)))
	print(hex(regs['banyan_data'].mask1(3)))
	print('aa',regs)
	addrread=c_address(0x10010)
	if addrread in regindex.keys():
		offset=regindex[addrread].address.offset(addrread)
		print(hex(offset))
