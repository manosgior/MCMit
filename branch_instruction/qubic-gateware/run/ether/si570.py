def fxtal(regs,fdefault=156.25e6):
	fxtal=fdefault*HS_DIV(regs)*N1(regs)/(RFREQ(regs)*1.0/2**28)
	if abs(fxtal*1.0/1e6-114.285)>1:
		print(abs(fxtal-114.285))
		print('fdefault',fdefault,'HS_DIV',HS_DIV(regs),'N1',N1(regs),'RFREQ',hex(RFREQ(regs)),RFREQ(regs)*1.0/2**28)
		print({a:format(d,'02x') for a,d in regs.items()})
		print('fxtal should be 114.285',fxtal/1e6)
		exit('si570 fxtal calculation wrong')
	else:
		print('fxtal',fxtal)
	return fxtal
def RFREQ(regs):
	return ((regs[8]&0x3f)<<32)+((regs[9]&0xff)<<24)+((regs[10]&0xff)<<16)+((regs[11]&0xff)<<8)+((regs[12]&0xff)<<0)
def N1(regs):
	return ((regs[7]&0x1f)<<2)+((regs[8]>>6)&0x3)+1
def HS_DIV(regs):
	return hsdivdict[((regs[7]>>5)&0x7)]
def fset(fset,fxtalval):
	N1s=[1]
	N1s.extend(range(2,2**7+1,2))
	validsel=[]
	for hsdiv in sorted([i for i in hsdivdict.keys() if hsdivdict[i]]):
		for n1 in N1s:
			fdco=fset*hsdivdict[hsdiv]*(n1)
			if (fdco<5.67e9 and fdco>4.85e9):
				n0=n1-1
				RFREQ=int(round(fdco/fxtalval*2**28))
				r7=((hsdiv&0x7)<<5)+(n0>>2)
				r8=((n0&0x3)<<6)+((RFREQ>>32)&0x3f)
				r9=(RFREQ>>24)&0xff
				r10=(RFREQ>>16)&0xff
				r11=(RFREQ>>8)&0xff
				r12=(RFREQ>>0)&0xff
				newdict={7:r7,8:r8,9:r9,10:r10,11:r11,12:r12}
				validsel.append(newdict)
	return validsel

def list2dict(l):
	d={}
	for (a,v) in l:
		d[a]=v
	return d

hsdivdict={0:4,1:5,2:6,3:7,4:None,5:9,6:None,7:11}
reset=[(135,0x81),]
default=[(7,0x1)
,(8,0xc2) # 194
,(9,0xbc) # 188
,(10,0x1) # 1
,(11,0x8e) # 142
,(12,0x2a) # 42
,(13,0x7) # 7
,(14,0xc2) # 194
,(15,0xc0) # 192
,(16,0x0)
,(17,0x0)
,(18,0x0)
,(135,0x0)
,(137,0x8)
]
f125=[
(137,0x10)
,(7,0x3)
,(8,0x43)
,(9,16)
,(10,1)
,(11,189)
,(12,242)
#,(137,0x00)
#,(135,0x40)
		]
f1251=[
(137,0x10)
,(7,3)
,(137,0xef)
]
if __name__=="__main__":
	fxtalval=fxtal(list2dict(default))
	print(fxtalval)
	print(fset(250e6,fxtalval))

