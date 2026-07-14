import usb
import serial.tools.list_ports
import serial
import sys
import time
import datetime

sndev={'B14E4CB55151363448202020FF172513':'attn1'
	,'C1715E095151363448202020FF18241D':'attn2'
	,'85828FEF5151363448202020FF172C4A':'attn3'
	,'849DA7495151363448202020FF183419':'attn4'
	,'AK08KYQC':'switch8'
	,'0001':'vc707uart'
	,'QubiCVC707224':'vc707uart224'
	,'QubiCVC707225':'vc707uart225'
	,'QubiCVC707226':'vc707uart226'
	,'QubiCVC707227':'vc707uart227'
	,'QubiCVC707228':'vc707uart228'
	}

def devcom(sndev,idVendor=0x2341,idProduct=0x0058):
	devs=usb.core.find(idVendor=idVendor,idProduct=idProduct,find_all=True)
#	arduino_sn=[]
#	for dev in devs:
#		arduino_sn.append(usb.util.get_string(dev,dev.iSerialNumber))
#		print('***',usb.util.get_string(dev,dev.iSerialNumber))
	coms=serial.tools.list_ports.comports()
	devcom={}
	for com in coms:
		print('com.device',com.device)
		comsn=com.serial_number
		comdev=com.device
		print('comsn',comsn)
		if comsn in sndev.keys():
			devcom[sndev[comsn]]=comdev
	return devcom
if __name__=="__main__":
	dev=devcom(sndev)
	print(dev)
	print(dev['vc707uart'])

