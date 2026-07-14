from gtextract import extractfromgtwizard,modifygenerate
import argparse
parser=argparse.ArgumentParser()
parser.add_argument('-f1',help='path for file 1',type=str,dest='file1')
parser.add_argument('-f2',help='path for file 2',type=str,dest='file2')
parser.add_argument('-m',help='module name',type=str,dest='module')
clargs=parser.parse_args()
f1=open(clargs.file1)
f2=open(clargs.file2)
s1=f1.read()
s2=f2.read()
f1.close()
f2.close()
[m1,para1,inst1,port1]=extractfromgtwizard(s1,clargs.module)
[m2,para2,inst2,port2]=extractfromgtwizard(s2,clargs.module)
print('-:%s'%clargs.file1)
print('+:%s'%clargs.file2)
print("parameters:")
for k,v in para1.items():
	if k in para2:
		if para2[k]==v:
			pass
		else:
			print('mismatch',k,'-',v,'+',para2[k])
	else:
		print('only in -, missing in +',k,v)
for k,v in para2.items():
	if k in para1:
		pass
	else:
		print('only in +, missing in -',k,v)

(portdict1,portwire1,portzeroone1,portconst1,portempty1,portsamename1,portvariable1)=modifygenerate(port1,portmoddict={})
(portdict2,portwire2,portzeroone2,portconst2,portempty2,portsamename2,portvariable2)=modifygenerate(port2,portmoddict={})
if 1:
	for k,v in portdict1.items():
		if k in portdict2:
			if portdict2[k]==v:
				pass
			else:
				print('port mismatch',k,'-',v,'+',portdict2[k])
		else:
			print('only in -, missing in +',k,v)
	for k,v in portdict2.items():
		if k in portdict1:
			pass
		else:
			print('only in +, missing in -',k,v)


