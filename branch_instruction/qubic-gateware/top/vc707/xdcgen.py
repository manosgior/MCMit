import os
import re
import sys
sys.path.append("submodules/tools/")
import fpga_module
filein=sys.argv[1]
basename=(os.path.splitext(os.path.basename(filein))[0])
fpga=fpga_module.xilinx_pinmap(filein)
fpgapins=fpga.verilog_io_pin()
exceptlist=["A12","AA41","AL40","B11","B18","B24","C41","D12","D31","H35","L11","P17"]


yespins=[p for p in fpgapins if p.package_pin not in exceptlist]
nopins=[p for p in fpgapins if p.package_pin in exceptlist]
yesport=[p for p in fpgapins if p.package_pin not in exceptlist]
noport=[p for p in fpgapins if p.package_pin in exceptlist]
f=open(basename+'.vh','w')
f.write(fpga.pininterface(yespins=yespins,nopins=nopins,yesport=yespins,noport=nopins))
f.close()
#print(mgtexcept)
#print(len(exceptlist))
#fpga.svxdc_gen(fpga.verilog_io_pin(),'t2.xdc',)

#mgtexcept=([p.package_pin for p in fpgapins if p.iotype=="GTX" and re.match('MGTX[TR]X[PN][0123]_11[345]',p.pinname)])
#exceptlist.extend(mgtexcept)

#yespins=[p for p in fpgapins if p.package_pin not in exceptlist]
#nopins=[p for p in fpgapins if p.package_pin in exceptlist]

f=open(basename+'.xdc','w')
f.write(fpga.svxdc(yespins,nopins))
f.close()

