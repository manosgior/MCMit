import sys
import re

def extractfromgtwizard(strin,modulename='GTXE2_CHANNEL'):
	m=re.match(r"[\s\S]*(?P<moduleinst>%s[\s\S]*?;)[\s\S]*"%modulename,strin)
	if m is None:
		print(modulename,'not found')
	else:
		a=re.match(r"\s*(?P<modulename>\S+)\s*#\s*\((?P<parameters>([\s\S]+?)\s*)\s*\)\s*(?P<instname>[a-zA-Z0-9_-]+)\s*\((?P<ports>([\s\S]+))\s*\)\s*;",m.group('moduleinst'),re.MULTILINE)
		paradict={}
		paralist=re.findall(r"\.(?P<pname>\S+?)\s*\((?P<pval>\S+?)\)",a['parameters'])
		for pname,pval in paralist:
			paradict[pname]=pval
		portdict={}
#print(len(paras),',\n'.join([".%s(%s)"%p for p in paras]))
		portlist=re.findall(r"\.(?P<pname>\S+?)\s*\((?P<pval>\S*?)\)",a['ports'])
		for pname,pval in portlist:
			portdict[pname]=pval
	return (modulename,paradict,a.group('instname'),portdict)
def parastr(paradict,paramoddict={}):
	paraline=[]
	for pname,pval in paradict.items():
		if pname in portmoddict:
			pval=portmoddict[pname]
		paraline.append(".%s(%s)"%(pname,pval))
	return paraline
def modifygenerate(portdict,portmoddict={}):
	portzeroone=[]
	portconst=[]
	portvariable=[]
	portempty=[]
	portsamename=[]
	portwire=[]
	for pname,pval in portdict.items():
		if pname in portmoddict:
			if portmoddict[pname]=="SAME":
				portmoddict[pname]=pname
			pval=portmoddict[pname]
		mgndvec=re.match(r'tied_to_ground_vec_i\[(?P<msb>\d+):(?P<lsb>\d+)\]',pval)
		mvccvec=re.match(r'tied_to_vcc_vec_i\[(?P<msb>\d+):(?P<lsb>\d+)\]',pval)
		mconst=re.match(r"\d+'[bdh][0-9a-fA-F]+",pval)
		mconst1=re.match(r"\d+",pval)
		mempty=re.match(r"\s*$",pval)
		mportin=re.match(r"(?P<name>\S+)_(?:in|IN)",pval)
		mportout=re.match(r"(?P<name>\S+)_[out|OUT]",pval)
		mportsame=re.match(r"(?P<name>\S+)",pval)
		if pval=='tied_to_ground_i':
			pval="1'b0";
			portzeroone.append(".%s(%s)"%(pname,pval))
			#print(pname,1)
		elif pval=='tied_to_vcc_i':
			pval="1'b1";
			portzeroone.append(".%s(%s)"%(pname,pval))
			#print(pname,2)
		elif mgndvec:
			pval="%d'h0"%(int(mgndvec.group('msb'))-int(mgndvec.group('lsb'))+1)
			portzeroone.append(".%s(%s)"%(pname,pval))
			#print(pname,3)
		elif mvccvec:
			pval="{%d{1'b1}}"%(int(mvccvec.group('msb'))-int(mvccvec.group('lsb'))+1)
			portzeroone.append(".%s(%s)"%(pname,pval))
			#print(pname,4)
		elif mconst:
			portconst.append(".%s(%s)"%(pname,pval))
			#print(pname,5)
		elif mconst1:
			portconst.append(".%s(%s)"%(pname,pval))
			#print(pname,6)
		elif mempty:
			portempty.append(".%s(%s)"%(pname,pval))
			#print(pname,7)
		elif mportin:
			if mportin.group('name').lower()==pname.lower():
				pval=pname
				portsamename.append(".%s"%(pname))
				if pname not in portmoddict and pname not in portwire:
					portwire.append("wire %s;"%pname)
			else:
				portvariable.append(".%s(%s)"%(pname,pval))
			#print(pname,8)
		elif mportout:
			if mportout.group('name').lower()==pname.lower():
				pval=pname
				portsamename.append(".%s"%(pname))
			else:
				portvariable.append(".%s(%s)"%(pname,pval))
			#print(pname,9)
			if pname not in portmoddict and pname not in portwire:
				portwire.append("wire %s;"%pname)
		elif mportsame:
			if mportsame.group('name').lower()==pname.lower():
				pval=pname
				portsamename.append(".%s"%(pname))
			else:
				portvariable.append(".%s(%s)"%(pname,pval))
			#print(pname,10)
		else:
			portvariable.append(".%s(%s)"%(pname,pval))
			#print(pname,1)
		portdict[pname]=pval
	return (portdict,portwire,portzeroone,portconst,portempty,portsamename,portvariable)
def modulegen(modulename,instname,paraline,portwire,portzeroone,portconst,portempty,portsamename,portvariable):
	wireuse='\n'.join(portwire)
	parause=','.join(paraline)
	portuse=('\n,'.join([','.join(portzeroone)
	,','.join(portconst)
	,','.join(portempty)
	,','.join(portsamename)
	,'\n,'.join(portvariable)]))
	modulestr='''%s
%s #(%s
)%s(
%s
);'''%(wireuse,modulename,parause,instname,portuse)
	return modulestr


if __name__=="__main__":
	f=open(sys.argv[1])
	s=f.read()
	f.close()
	import importlib
	lib=importlib.import_module(sys.argv[2])
	portmoddict=lib.portmoddict
#	portmoddict={"DRPADDR":"9'h0","DRPCLK":'0',"DRPDI":'0',"DRPDO":'',"DRPEN":'0',"DRPRDY":'',"DRPWE":'0'
#,"DMONITOROUT":''
#,"EYESCANDATAERROR":''
#,"EYESCANRESET":'0'
#,"EYESCANTRIGGER":'0'
#,"RXDFEAGCHOLD":'0'
#,"RXDFELFHOLD":'0'
#,"RXDFELPMRESET":'0'
#,"RXMONITOROUT":''
#,"RXMONITORSEL":'0'
#,"RXPCOMMAALIGNEN":'SAME'
#,"GTNORTHREFCLK0":"SAME"
#,"GTNORTHREFCLK1":"SAME"
#,"GTREFCLK0":"SAME"
#,"GTREFCLK1":"SAME"
#,"GTSOUTHREFCLK0":"SAME"
#,"GTSOUTHREFCLK1":"SAME"
#,"RXSLIDE":'0'
#,"CPLLPD":"1'b0"
#,"CPLLLOCK":"SAME"
#,"CPLLFBCLKLOST":"SAME"
#,"CPLLREFCLKLOST":"SAME"
#,"CPLLRESET":"SAME"
#,"RXBYTEISALIGNED":"SAME"
#,"RXBYTEREALIGN":"SAME"
#,"CPLLLOCKDETCLK":"SAME"
#,"RXDATA":"rxdata64"
#,"RXDISPERR":"rxdisperr8"
#,"RXCHARISK":"rxcharisk8"
#,"RXCHARISCOMMA":"rxchariscomma8"
#,"RXNOTINTABLE":"rxnotintable8"
#,"TXDATA":"txdata64"
#,"TXCHARISK":"txcharisk8"
#,"CPLLREFCLKSEL":"SAME"
#,"QPLLCLK":"SAME"
#,"QPLLREFCLK":"SAME"
#,"RXUSERRDY":"rxuserrdy"
#,"RXUSRCLK":"SAME"
#,"RXUSRCLK2":"SAME"
#,"GTXRXP":"SAME"
#,"GTXRXN":"SAME"
#,"GTRXRESET":"SAME"
#,"RXPMARESET":"SAME"
#,"GTTXRESET":"SAME"
#,"TXUSERRDY":"SAME"
#,"TXUSRCLK":"SAME"
#,"TXUSRCLK2":"SAME"
#,"RXCDRLOCK":"SAME"
#,"RXDLYEN":"SAME"
#,"RXDLYSRESET":"SAME"
#,"RXPHALIGN":"SAME"
#,"RXPHALIGNEN":"SAME"
#,"RXPHDLYRESET":"SAME"
#,"RXLPMHFHOLD":"0"
#,"RXLPMLFHOLD":"0"
#,"TXDLYEN":"SAME"
#,"TXDLYSRESET":"SAME"
#,"TXPHALIGN":"SAME"
#,"TXPHALIGNEN":"SAME"
#,"TXPHDLYRESET":"SAME"
#,"TXPHINIT":"SAME"
#,"TXPHINITDONE":"SAME"
#,"RXELECIDLE":"SAME"
#}
	(modulename,paradict,instname,portdict)=extractfromgtwizard(s)
	paraline=parastr(paradict)
	(portdict,portwire,portzeroone,portconst,portempty,portsamename,portvariable)=modifygenerate(portdict,portmoddict=portmoddict)
	modulestr=modulegen(modulename,instname,paraline,portwire,portzeroone,portconst,portempty,portsamename,portvariable)
	print(modulestr)



