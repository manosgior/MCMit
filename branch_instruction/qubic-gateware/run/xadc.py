import time
import serial
from uartlb import c_uartlb
#from matplotlib import pyplot
#from matplotlib.animation import FuncAnimation


class c_xadc(c_uartlb):
	def __init__(self,port,baudrate,timeout=0):
		c_uartlb.__init__(self,port,baudrate,timeout)
		self.map={'temp':5,'aux4':7,'aux12':8};
	def updatetemp(self):
		(c,a,v0)=ser.read(addr=5)
		return (v0/2**4)*503.975/4096.0-273.15
	def updateaux(self,chan):
		(c,a,v)=ser.read(addr=self.map[chan])
		return v/2**4*1.0/0xfff
	def update(self):
		temp=self.updatetemp()
		aux4=self.updateaux('aux4')
		aux12=self.updateaux('aux12')
		return (temp,aux4,aux12)
		
if __name__=="__main__":
	import random
	ser=c_xadc('/dev/ttyS8',9600,0)
	ser.write(addr=4,data=0)
	ser.resetlb()
	while (1):
		print(ser.update())
		time.sleep(0.5)
