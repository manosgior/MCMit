import zlib
import random
import struct
pac=(
[500,'udp',(
(8,0x1000000000000000)
,(8,0x1000000100000000)
,(8,0x1000000100000000)
,(8,0x1000000100000000)
,(8,0x1000000000000000)
,(8,0x1000000000000000)
,(8,0x1000000000000000)
,(8,0x1000000000000000)
,(8,0x1000000000000000)
,(8,0x1000000000000000)
,(8,0x1000000000000000)
,(8,0x1000000200000000)
,(8,0x1000000100000000)
)]
,[1000,'frame',(
(8,0x55555555555555d5)
,(8,0x55aabbccddeec46e)
,(8,0x1f01d90d08004500)
,(8,0x0054670b40004001)
,(8,0x4ea5c0a801c8c0a8)
,(8,0x01e00800648a1ad0)
,(8,0x003c27f001600000)
,(8,0x00008d4603000000)
,(8,0x0000101112131415)
,(8,0x161718191a1b1c1d)
,(8,0x1e1f202122232425)
,(8,0x262728292a2b2c2d)
,(8,0x2e2f303132333435)
,(6,0x3637e07f58c6)
)]
,[1500,'frame',(
(8,0x55555555555555d5)
,(8,0x55aabbccddeec46e)
,(8,0x1f01d90d08004500)
,(8,0x0054534a40004001)
,(8,0x6266c0a801c8c0a8)
,(8,0x01e008008e692205)
,(8,0x0036a6da00600000)
,(8,0x0000da4d07000000)
,(8,0x0000101112131415)
,(8,0x161718191a1b1c1d)
,(8,0x1e1f202122232425)
,(8,0x262728292a2b2c2d)
,(8,0x2e2f303132333435)
,(6,0x3637c8d656bf)

	)]
,[2100,'udp',((8,0xfe_000003_00000000)
	,(8,0x10_00000d_00000000)
	,(8,0x00_000021_00000023)
	,(8,0xff_000000_00000000))]
,[2200,'ping',((32,0x20212223),)]
,[2300,'udp',((8,0x10_000401_00000000)
	,(8,0x10_000403_00000000)
	,(8,0x10_0007ff_00000000)
	)]
,[3300,'udp',((8,0x00_000055_00000000)
	,(8,0x10_001000_00000000)
	,(8,0x10_001001_00000000)
	,(8,0x10_001002_00000000)
	,(8,0x10_0017fe_00000000)
	,(8,0x10_0017ff_00000000)
	,(8,0x10_001800_00000000)
	)]
,[4300,'udp',((8,0x00_000055_00000000)
	,(8,0x00_000100_00000000)
	,(8,0x10_000403_00000000)
	,(8,0xff_000100_00000000)
	)]
,[5300,'udp',((8,0x00_000055_00000000)
	,(8,0x00_000100_00000000)
	,(8,0x10_000403_00000000)
	,(8,0xff_000100_00000000)
	)]
)

class packets():
	def __init__(self):
		self.smac=0x00249b59c771#0xabcdef123456
		self.sip=0xc0a80102
		self.dmac=0x503eaa059701#0x515542494301#0x503eaa059701#,32'hc0a801e0#0x55aabbccddee
		self.dip=0xc0a801e0
		self.sport=0xb92a#0x1234
		self.dport=0xd003
		self.ipv4protocol={'icmp':1,'udp':0x11}
		self.ethertype={'ipv4':0x0800,'arp':0x0806}
		self.itype={'echoreply':0,'echorequest':8}
	def udp(self,payload):
		return self.sport.to_bytes(2,'big')+self.dport.to_bytes(2,'big')+(len(payload)+8).to_bytes(2,'big')+(0).to_bytes(2,'big')+payload
	def ipv4(self,payload,protocol):
		verihl=0x45.to_bytes(1,'big')
		dscpecn=(0).to_bytes(1,'big')
		totallength=(len(payload)+20).to_bytes(2,'big')
		identification=0xb63e.to_bytes(2,'big') # random.randint(0,2**16-1).to_bytes(2,'big')
		flagfrag=0x4000.to_bytes(2,'big')
		ttl=(0x40).to_bytes(1,'big')
		protocol=self.ipv4protocol[protocol].to_bytes(1,'big')
		checksum0=(0).to_bytes(2,'big')
		sip=self.sip.to_bytes(4,'big')
		dip=self.dip.to_bytes(4,'big')
		header0=b''.join([verihl,dscpecn,totallength,identification,flagfrag,ttl,protocol,checksum0,sip,dip])
		checksum=self.checksum(header0).to_bytes(2,'big')
		header=b''.join([verihl,dscpecn,totallength,identification,flagfrag,ttl,protocol,checksum,sip,dip])
		return header+payload
	def checksum(self,frame):
		sum0=sum(struct.unpack('>%dH'%(len(frame)//2),frame))
		checksum=0xffff-((sum0>>16)+(sum0&0xffff))
		return checksum

	def ping(self,payload):
		#		val=b''.join([i.to_bytes(1,'big') for i in range(32)])
		if (len(payload)%2!=0):
			payload=b'\xc5'+payload
		return payload
	def icmp(self,payload,itype):
		itype=self.itype[itype].to_bytes(1,'big')
		icode=(0).to_bytes(1,'big')
		identifier=(0).to_bytes(2,'big')
		seq=(0xc5).to_bytes(2,'big')
		checksum=self.checksum(b''.join([itype,icode,identifier,seq,payload])).to_bytes(2,'big')
		return b''.join([itype,icode,checksum,identifier,seq,payload])


	def ether(self,payload,ethertype):
		preamble=0x55555555555555d5.to_bytes(8,'big')
		tail=(0).to_bytes(12,'big')
		header=self.dmac.to_bytes(6,'big')+self.smac.to_bytes(6,'big')+self.ethertype[ethertype].to_bytes(2,'big')
		crc=(zlib.crc32(header+payload)&0xffffffff).to_bytes(4,'little')
		return preamble+header+payload+crc#+tail



	def parse(self,pac):
		timeframe=[]
		for time,proto,l in pac:
			payload=b''
			for length,data in l:
				payload+=data.to_bytes(length,'big')
			if proto=='udp':
				frame=self.ether(self.ipv4(self.udp(payload),'udp'),'ipv4')
			elif proto=='ping':
				frame=self.ether(self.ipv4(self.icmp(self.ping(payload),'echorequest'),'icmp'),'ipv4')
			elif proto=='arp':
				frame=self.ether(self.arp(payload),'arp')
			elif proto=='frame':
				frame=payload
			else:
				print('unknown proto')
			timeframe.append((time,frame))
		return timeframe
	def veri(self,timeframe):
		simvh='''
always @(posedge ifgmii.tx_clk) begin
%s
%s
end'''
		tcmd1='''if (txclkcnt==%d) begin datasr<=%s<<%d; end'''
		tcmd2='''((txclkcnt>=%d) & (txclkcnt<%d))'''
		MAXNBYTES=200*8
		cmd1=[]
		cmd2=[]
		for (time,frame) in timeframe:
			length=len(frame)
			framehex=("{%s}"%(','.join(["16'h%04x"%(i) for i in struct.unpack('>%dH'%(length//2),frame)])))
			cmd1.append(tcmd1%(time,framehex,(MAXNBYTES-length)*8))
			cmd2.append(tcmd2%(time,time+length))
		print(simvh%(('\nelse '.join(cmd1)+'\nelse datasr<= datasr<<8;'),('simdv='+'\n|'.join(cmd2)+';')))

		return (time,framehex)

if __name__=="__main__":
	a=packets()
	a.veri(a.parse(pac))

