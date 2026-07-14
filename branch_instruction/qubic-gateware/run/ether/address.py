class c_address:
	def __init__(self,base,length=0):
		self.base=base
		self.length=length
		self.top=self.base+self.length
		self.offset=0
		self.keybase=self.base
		#print 'c_address init',self.width
		#print 'base',self.base,self.width,self.baseonly

#	def __eq__(self,other):
#		#		print 'eq',other.base,other.base>>self.width,self.baseonly
#		print 'eq',self.base,self.top,other.base,other.top
#		eq1=(other.base>=self.base)&(other.base<self.top)
#		eq2=(self.base>=other.base)&(self.base<other.top)
#		if eq1 and other.base!=self.base:
#			self.offset=other.base-self.base
#			other.keybase=self.keybase
#		elif eq2 and self.base!=other.base:
#			self.offset=self.base-other.base
#			print self.offset
#			self.keybase=other.base
#		else:
#			pass
#
#		return  eq1|eq2
#
#	def __hash__(self):
#		print 'hash'
#		return self.keybase#hash((self.base,self.top) if self.length else self.base)
#
#	def __repr__(self):
#		return str((self.base,self.top) if self.length else self.base)
#	#	return str(self.base)
#	def __ne__(self, other):
#		return not self.__eq__(other)
#
##	def offset(self,other):
##		return other.base-self.base
#
if __name__=="__main__":
	a=c_address(0x30)
	b=c_address(0x50,length=8)
	c={a:a,b:b}
	d=c_address(0x51)
#	print d==b#, d in c.keys()
#	print b==d
	#print d in c.keys()
	print(d in c)
#	print c,d
#	print c.has_key(d)
#	print c,d
#	print c[d]
#	print d.offset
#	print b.offset
	#print c
