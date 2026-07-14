'''brambus.tcl
bramif_portinst.vh
bramif_port.vh
bram_parainst.vh
bram_para.vh
bram_plsv.vh
bram_portinst.vh
bram_port.vh
bram_range.tcl
bram.tcl
bramwithctrl.tcl
bram.xsa

,"qdrvenv1":{"Awidth":512,"Adepth":1024,"Bwidth":512,"latency":2}
'''

#class addrranges:
#    def __init__(self,used,size):
#        self.unused=[(0,size),]
#        self.used=[]
#        for u in used:
#            self.append(u)
#            self.
#    def append(self,other):
#        if not self.overlap(other):
#            self.arange.append(other).sort()
#        else:
#            print('overlap with',other)
#    def opverlap(self,other):
#        return not any([a.overlap(other) for a in self.arange])
#    def firstavail(self,size):
#        
#
#
#class addrrange:
#    def __init__(self,start,stop):
#        self.start=start
#        self.stop=stop
#    def overlap(self,other):
#        return not ((other.start>self.stop and other.stop>self.stop) or (other.start<self.start and other.stop<self.stop))
#
#
#def useaddr(unused,used,new):
#    for index,(start,stop) in enumerate(used):
#        if n
#
#    used.append(new)
#

import numpy
import json
import re
import os

class brams:
    def __init__(self,bramparadict,template):
        self.bramlist=[]
        for name,item in bramparadict.items():
            print('start',name)
            self.bramlist.append(bram(name=name,template=template,**item))
        self.assign()
    def assign(self,keep=True,busdatawidth=32,busaddrwidth=32):
        assigned=[]
        unassigned=self.bramlist
        unassigned.sort(key=lambda x:(-x.membitsize,x.name))
        last=0
        for m in unassigned:
            m.address=last
            last=last+m.membitsize//busaddrwidth
    def addressmap(self):
        #return {m.name:(m.address,m.address>>(m.Bwidth if m.r else m.Awidth) ) for m in self.bramlist}
        return {m.name:(m.address,m.address>>(m.Aaddrwidth if m.r else m.Baddrwidth),(m.Aaddrwidth if m.r else m.Baddrwidth) ) for m in self.bramlist}
            


#        unusedaddr=[i for i in range(1,len(self.regs)+1) if i not in assigned]
#        for index,r in enumerate(unassigned):
#            r.base_addr=unusedaddr[index]
#        self.sorted()

            
    def writefile(self,filename,strval):
        f=open(os.path.join('gensrc', filename),'w')
        f.write(strval)
        f.close()
    def brambus_tcl(self,filename="brambus.tcl"):
        perbus='\n'.join([b.brambus_tcl() for b in self.bramlist])
        strval=template['brambus.tcl']['all']%(dict(perbus=perbus))
        self.writefile(filename,strval)
        return strval

    def bramif_lbportinst_vh(self,filename="bramif_lbportinst.vh"):
        perbus='\n,'.join([b.bramif_lbportinst_vh() for b in self.bramlist])
        strval=template['bramif_lbportinst.vh']['all']%(dict(perbus=perbus))
        self.writefile(filename,strval)
        return strval
    def bramif_lbport_vh(self,filename="bramif_lbport.vh"):
        perbus='\n,'.join([b.bramif_lbport_vh() for b in self.bramlist])
        strval=template['bramif_lbport.vh']['all']%(dict(perbus=perbus))
        self.writefile(filename,strval)
        return strval
    def bramif_portinst_vh(self,filename="bramif_portinst.vh"):
        perbus='\n,'.join([b.bramif_portinst_vh() for b in self.bramlist])
        strval=template['bramif_portinst.vh']['all']%(dict(perbus=perbus))
        self.writefile(filename,strval)
        return strval
    def bramif_port_vh(self,filename="bramif_port.vh"):
        perbus='\n,'.join([b.bramif_port_vh() for b in self.bramlist])
        strval=template['bramif_port.vh']['all']%(dict(perbus=perbus))
        self.writefile(filename,strval)
        return strval

    def bram_parainst_vh(self,filename="bram_parainst.vh"):
        perbuslist=sorted(list(set([b.bram_parainst_vh() for b in self.bramlist])))
        perbus='\n,'.join(perbuslist)
        strval= template['bram_parainst.vh']['all']%(dict(perbus=perbus))
        self.writefile(filename,strval)
        return strval
    def bram_para_vh(self,filename="bram_para.vh"):
        perbuslist=sorted(list(set([b.bram_para_vh() for b in self.bramlist])))
        perbus='\n,'.join(perbuslist)
        strval= template['bram_para.vh']['all']%(dict(perbus=perbus))
        self.writefile(filename,strval)
        return strval
    def bram_plsv_vh(self,filename="bram_plsv.vh"):
        addrperdata='\n'.join(list(set([b.addrperdata() for b in self.bramlist])))
        perbuslist=sorted(list(set([b.bram_plsv_vh() for b in self.bramlist])))
        perbus='\n'.join(perbuslist)
        strval= template['bram_plsv.vh']['all']%(dict(addrperdata=addrperdata,perbus=perbus))
        self.writefile(filename,strval)
        return strval
    def bram_portinst_vh(self,filename="bram_portinst.vh"):
        perbuslist=sorted(list(set([b.bram_portinst_vh() for b in self.bramlist])))
        perbus='\n,'.join(perbuslist)
        strval= template['bram_portinst.vh']['all']%(dict(perbus=perbus))
        self.writefile(filename,strval)
        return strval
    def bram_port_vh(self,filename="bram_port.vh"):
        perbuslist=sorted(list(set([b.bram_port_vh() for b in self.bramlist])))
        perbus='\n,'.join(perbuslist)
        strval= template['bram_port.vh']['all']%(dict(perbus=perbus))
        self.writefile(filename,strval)
        return strval
    def bram_range_tcl(self,filename="bram_range.tcl"):
        perbuslist=sorted(list(set([b.bram_range_tcl() for b in self.bramlist])))
        perbus='\n'.join(perbuslist)
        strval= template['bram_range.tcl']['all']%(dict(perbus=perbus))
        self.writefile(filename,strval)
        return strval
    def bramwithctrl_tcl(self,filename="bramwithctrl.tcl"):
        perbuslist=sorted(list(set([b.bramwithctrl_tcl() for b in self.bramlist])))
        perbus='\n'.join(perbuslist)
        strval= template['bramwithctrl.tcl']['all']%(dict(perbus=perbus))
        self.writefile(filename,strval)
        return strval
    def adw(self):
        perbuslist=sorted(list(set([b.adw() for b in self.bramlist])))
        perbus='\n'.join(perbuslist)
        strval= template['adw']['all']%(dict(perbus=perbus))
        #self.writefile(filename,strval)
        return strval
    def write(self,filename="bram_write.vh"):
        perbuslist=sorted(list(set([b.write() for b in self.bramlist])))
        perbus='\n'.join(perbuslist)
        strval= template['write']['all']%(dict(perbus=perbus))
        self.writefile(filename,strval)
        return strval
    def read(self,filename="bram_read.vh"):
        perbuslist=sorted(list(set([b.read() for b in self.bramlist])))
        perbus='\n'.join(perbuslist)
        strval= template['read']['all']%(dict(perbus=perbus))
        self.writefile(filename,strval)
        return strval
    def ifbramctrl_sv(self,filename="ifbramctrl.sv"):
        wnames='\n'.join([b.wname() for b in self.bramlist if b.wname()])
        rnames='\n'.join([b.rname() for b in self.bramlist if b.rname()])
        walways='\n'.join([b.walways() for b in self.bramlist if b.walways()])
        ralways='\n'.join([b.ralways() for b in self.bramlist if b.ralways()])
        ralwayscase='\n'.join([b.ralwayscase() for b in self.bramlist if b.ralwayscase()])
        strval= template['ifbramctrl.sv']['all']%(dict(wnames=wnames,rnames=rnames,walways=walways,ralways=ralways,ralwayscase=ralwayscase))
        self.writefile(filename,strval)
        return strval
    def bram_json(self,filename="bram.json"):
        perbus='\n,'.join([b.bram_json() for b in self.bramlist])
        strval=template['bram.json']['all']%(dict(perbus=perbus))
        self.writefile(filename,strval)
        return strval

    def braminit_para_vh(self,filename="braminit_para.vh"):
        perbus='\n,'.join([b.braminit_para_vh() for b in self.bramlist])
        strval=template['braminit_para.vh']['all']%(dict(perbus=perbus))
        self.writefile(filename,strval)
        return strval

        return template['braminit_para.vh']['perbus']%(self.paradict())
    def braminit_parainst_vh(self,filename="braminit_parainst.vh"):
        perbus='\n,'.join([b.braminit_parainst_vh() for b in self.bramlist])
        strval=template['braminit_parainst.vh']['all']%(dict(perbus=perbus))
        self.writefile(filename,strval)
        return strval
    def bramsiminit_vh(self,filename="bramsiminit.vh"):
        perbus='\n'.join([b.bramsiminit_vh() for b in self.bramlist])
        strval=template['bramsiminit.vh']['all']%(dict(perbus=perbus))
        self.writefile(filename,strval)
        return strval






class bram:
    def __init__(self,name,access,Awidth,Adepth,Bwidth,latency,address,template,ram_style,prefix):
        self.name=name
        self.ram_style=ram_style
        self.access=access
        self.Awidth=Awidth
        self.Adepth=Adepth
        self.Bwidth=Bwidth
        self.latency=latency
        self.prefix=prefix
        self.membitsize=self.Awidth*self.Adepth
        self.Bdepth=self.membitsize/self.Bwidth
        self._address=-1 if address is None else address
        self.Aaddrwidth=int(numpy.log2(self.membitsize/self.Awidth))
        self.bitaddrwidth=int(numpy.log2(self.membitsize))
        self.byteaddrwidth=int(numpy.ceil(round(numpy.log2(self.membitsize/8),10)))
        self.Baddrwidth=int(numpy.ceil(round(numpy.log2(self.membitsize/self.Bwidth),10)))
        print("*************",self.Baddrwidth)
        self.template=template
    @property
    def address(self):
        return self._address
    @address.setter
    def address(self,address):
        self._address=address
    @property
    def subname(self):
        nosubname=['command']
        subname=self.name
        if self.name not in nosubname:
            m=re.match('(\S+?)([0-9]+)$',self.name)
            #print('debug',self.name,m.groups())
            if m:
                subname='%s[%d]'%(m.groups()[0],int(m.groups()[1],0))
        return subname
    @property
    def sizekM(self):
        if self.membitsize<2**20:
            kM='%dk'%(self.membitsize/1024)
        else:
            kM='%dM'%(self.membitsize/1024/1024)
        return kM
    @property
    def w(self):
        return 'w' in self.access
    @property
    def r(self):
        return 'r' in self.access
    @property
    def addrcheck(self):
        return self.address>>(self.Aaddrwidth if self.r else self.Baddrwidth)
    @property
    def namerw(self):
        return self.name+('_W' if self.w else '_R')
    @property
    def namelbrw(self):
        return self.name+('_R' if self.w else '_W')
    def paradict(self):
        return dict(name=self.name,NAME=self.name.upper(),Awidth=self.Awidth,Adepth=self.Adepth,Bwidth=self.Bwidth,latency=self.latency,prefix=self.prefix,membitsize=self.membitsize,Baddrwidth=self.Baddrwidth,memsizekM=self.sizekM,subname=self.subname,Aaddrwidth=self.Aaddrwidth,Bdepth=self.Bdepth,addrcheck=self.addrcheck,namerw=self.namerw,namelbrw=self.namelbrw,ram_style=self.ram_style,address=self.address,access=self.access)
    def brambus_tcl(self):
        return template['brambus.tcl']['perbus']%(self.paradict())
    def bramif_lbportinst_vh(self):
        return template['bramif_lbportinst.vh']['perbus'+('r' if self.r else 'w')]%(self.paradict())
    def bramif_lbport_vh(self):
        return template['bramif_lbport.vh']['perbus'+('r' if self.r else 'w')]%(self.paradict())
    def bramif_portinst_vh(self):
        return template['bramif_portinst.vh']['perbus'+('r' if self.r else 'w')]%(self.paradict())
    def bramif_port_vh(self):
        return template['bramif_port.vh']['perbus'+('r' if self.r else 'w')]%(self.paradict())
    def braminitfile_parainst_vh(self):
        return template['bram_parainst.vh']['initfile']%(self.paradict())
    def bram_parainst_vh(self):
        return template['bram_parainst.vh']['perbus']%(self.paradict())
    def braminitfile_para_vh(self):
        return template['bram_para.vh']['initfile']%(self.paradict())
    def bram_para_vh(self):
        return template['bram_para.vh']['perbus']%(self.paradict())
    def addrperdata(self):
        return template['bram_plsv.vh']['addrperdata']%(self.paradict())
    def bram_plsv_vh(self):
        return template['bram_plsv.vh']['perbus'+('r' if self.r else 'w')]%(self.paradict())
    def bram_portinst_vh(self):
        return template['bram_portinst.vh']['perbus']%(self.paradict())
    def bram_port_vh(self):
        return template['bram_port.vh']['perbus']%(self.paradict())
    def bram_range_tcl(self):
        return template['bram_range.tcl']['perbus']%(self.paradict())
    def bramwithctrl_tcl(self):
        return template['bramwithctrl.tcl']['perbus']%(self.paradict())
    def adw(self):
        return template['adw']['perbus'+('r' if self.r else 'w')]%(self.paradict())
    def write(self):
        return template['write']['perbus']%(self.paradict()) if self.w else ''
    def read(self):
        return template['read']['perbus']%(self.paradict()) if self.r else ''
    def wname(self):
        return template["ifbramctrl.sv"]['wname']%(self.paradict()) if self.r else ''
    def walways(self):
        return template["ifbramctrl.sv"]['walways']%(self.paradict()) if self.r else ''
    def rname(self):
        return template["ifbramctrl.sv"]['rname']%(self.paradict()) if self.w else ''
    def ralways(self):
        return template["ifbramctrl.sv"]['ralways']%(self.paradict()) if self.w else ''
    def ralwayscase(self):
        return template["ifbramctrl.sv"]['ralwayscase']%(self.paradict()) if self.w else ''
    def bram_json(self):
        return template['bram.json']['perbus']%(self.paradict())
    def braminit_para_vh(self):
        return template['braminit_para.vh']['perbus']%(self.paradict())
    def braminit_parainst_vh(self):
        return template['braminit_parainst.vh']['perbus']%(self.paradict())
    def bramsiminit_vh(self):
        return template['bramsiminit.vh']['perbus']%(self.paradict())

if __name__=="__main__":
    template={"brambus.tcl":{
    "perbus":"brambus %(name)s"
    ,"all":"proc bram_map {busname inst} {\n	set brampins {RST CLK DIN EN DOUT WE ADDR}\n	foreach pin $brampins {\n		ipx::add_port_map $pin [ipx::get_bus_interfaces $busname -of_objects [ipx::current_core]]\n		set_property physical_name ${inst}_[string tolower $pin] [ipx::get_port_maps $pin -of_objects [ipx::get_bus_interfaces $busname -of_objects [ipx::current_core]]]\n	}\n}\nproc brambus {bramname} {\n\tipx::add_bus_interface ${bramname} [ipx::current_core]\n\tset_property abstraction_type_vlnv xilinx.com:interface:bram_rtl:1.0 [ipx::get_bus_interfaces ${bramname} -of_objects [ipx::current_core]]\n\tset_property bus_type_vlnv xilinx.com:interface:bram:1.0 [ipx::get_bus_interfaces ${bramname} -of_objects [ipx::current_core]]\n\tset_property interface_mode master [ipx::get_bus_interfaces ${bramname} -of_objects [ipx::current_core]]\n\tbram_map ${bramname} [string toupper $bramname]\n}\n%(perbus)s"
    ,"perbus":"brambus %(name)s"
    }
    ,"bramwithctrl.tcl":{
        "all":"proc bramwithctrl {bramname Awidth Adepth Bwidth latency} {\n\tincr $Awidth 0\n\tincr $Adepth 0\n\tincr $Bwidth 0\n\tincr $latency 0\n\tcreate_bd_cell -type ip -vlnv xilinx.com:ip:axi_bram_ctrl:4.1 ${bramname}_ctrl\n\tcreate_bd_cell -type ip -vlnv xilinx.com:ip:blk_mem_gen:8.4 ${bramname}_mem\n\tset_property -dict [list CONFIG.READ_LATENCY ${latency} CONFIG.SINGLE_PORT_BRAM {1} CONFIG.DATA_WIDTH ${Awidth}] [get_bd_cells ${bramname}_ctrl]\n\tset_property -dict [list CONFIG.use_bram_block {Stand_Alone} CONFIG.Memory_Type {True_Dual_Port_RAM} CONFIG.Write_Depth_A ${Adepth} CONFIG.Write_Width_A ${Awidth} CONFIG.Read_Width_A ${Awidth} CONFIG.Write_Width_B ${Bwidth} CONFIG.Read_Width_B ${Bwidth} CONFIG.Enable_32bit_Address {true} CONFIG.Use_Byte_Write_Enable {true} CONFIG.Byte_Size {8} CONFIG.Use_RSTA_Pin {false} CONFIG.Use_RSTB_Pin {false} CONFIG.Enable_A {Always_Enabled} CONFIG.Enable_B {Always_Enabled} CONFIG.EN_SAFETY_CKT {false}] [get_bd_cells ${bramname}_mem]\n}\n%(perbus)s"
        ,"perbus":"bramwithctrl %(name)s %(Awidth)d %(Adepth)d %(Bwidth)d %(latency)d"
        }
    ,"bramif_lbportinst.vh":{
        "all":"%(perbus)s"
        ,"perbusr":".%(name)s_W(%(name)s_W)"
        ,"perbusw":".%(name)s_R(%(name)s_R)"
        }
    ,"bramif_lbport.vh":{
        "all":"%(perbus)s"
        ,"perbusr":"ifbram %(name)s_W"
        ,"perbusw":"ifbram %(name)s_R"
        }
    ,"bramif_portinst.vh":{
        "all":"%(perbus)s"
        ,"perbusr":".%(name)s_R(%(name)s_R)"
        ,"perbusw":".%(name)s_W(%(name)s_W)"
        }
    ,"bramif_port.vh":{
        "all":"%(perbus)s"
        ,"perbusr":"ifbram %(name)s_R"
        ,"perbusw":"ifbram %(name)s_W"
        }
    ,"bram_parainst.vh":{
        "all":"%(perbus)s"
        ,"perbus":".%(prefix)s_R_ADDRWIDTH(%(prefix)s_R_ADDRWIDTH),.%(prefix)s_R_DATAWIDTH(%(prefix)s_R_DATAWIDTH),.%(prefix)s_W_ADDRWIDTH(%(prefix)s_W_ADDRWIDTH),.%(prefix)s_W_DATAWIDTH(%(prefix)s_W_DATAWIDTH)"
        }
    ,"bram_para.vh":{
        "all":"%(perbus)s"
        ,"perbus":"parameter integer %(prefix)s_R_ADDRWIDTH=%(Baddrwidth)d,parameter integer %(prefix)s_R_DATAWIDTH=%(Bwidth)d,parameter integer %(prefix)s_R_DEPTH=%(Bdepth)d,parameter integer %(prefix)s_W_ADDRWIDTH=%(Aaddrwidth)d,parameter integer %(prefix)s_W_DATAWIDTH=%(Awidth)d,parameter integer %(prefix)s_W_DEPTH=%(Adepth)d"
        }
    ,"braminit_para.vh":{
        "all":"%(perbus)s"
        ,"perbus":"parameter INIT_%(name)s=\"\""
        }
    ,"braminit_parainst.vh":{
        "all":"%(perbus)s"
        ,"perbus":".INIT_%(name)s(INIT_%(name)s)"
        }
    ,"bramsiminit.vh":{
        "all":"%(perbus)s"
        ,"perbus":"localparam INIT_%(name)s=\"INIT_%(name)s.mem\";"
        }
    ,"bram_plsv.vh":{
        "all":"%(addrperdata)s\n\n%(perbus)s"
        ,"addrperdata":"localparam %(prefix)s_R_ADDRPERDATA=$clog2(%(prefix)s_R_DATAWIDTH)-3;localparam %(prefix)s_W_ADDRPERDATA=$clog2(%(prefix)s_W_DATAWIDTH)-3;"
        ,"perbusrwith_en":"ifbram #(.ADDR_WIDTH(%(prefix)s_W_ADDRWIDTH),.DATA_WIDTH(%(prefix)s_W_DATAWIDTH)) %(name)s_W();\nifbram #(.ADDR_WIDTH(%(prefix)s_R_ADDRWIDTH),.DATA_WIDTH(%(prefix)s_R_DATAWIDTH)) %(name)s_R();\nbram_cfg %(name)s_W_cfg(.bram(%(name)s_W),.clk(lb3_clk),.rst(1'b0),.en(1'b1));\nasym_ram_sdp_read_wider #(.DATAWIDTHA(%(prefix)s_W_DATAWIDTH),.ADDRWIDTHA(%(prefix)s_W_ADDRWIDTH),.SIZEA(%(prefix)s_W_DEPTH),.DATAWIDTHB(%(prefix)s_R_DATAWIDTH),.ADDRWIDTHB(%(prefix)s_R_ADDRWIDTH),.SIZEB(%(prefix)s_R_DEPTH),.RAM_STYLE(\"%(ram_style)s\"))\n%(name)s_mem(.clkA(%(name)s_W.clk),.enaA(%(name)s_W.en),.weA(%(name)s_W.we),.addrA(%(name)s_W.addr),.diA(%(name)s_W.din),.clkB(%(name)s_R.clk),.enaB(%(name)s_R.en),.addrB(%(name)s_R.addr),.doB(%(name)s_R.dout));\n"
        ,"perbusr":"ifbram #(.ADDR_WIDTH(%(prefix)s_W_ADDRWIDTH),.DATA_WIDTH(%(prefix)s_W_DATAWIDTH)) %(name)s_W();\nifbram #(.ADDR_WIDTH(%(prefix)s_R_ADDRWIDTH),.DATA_WIDTH(%(prefix)s_R_DATAWIDTH)) %(name)s_R();\nbram_cfg %(name)s_W_cfg(.bram(%(name)s_W),.clk(lb3_clk),.rst(1'b0),.en(1'b1));\nasym_ram_sdp_read_wider #(.INIT_FILE(INIT_%(name)s),.DATAWIDTHA(%(prefix)s_W_DATAWIDTH),.ADDRWIDTHA(%(prefix)s_W_ADDRWIDTH),.SIZEA(%(prefix)s_W_DEPTH),.DATAWIDTHB(%(prefix)s_R_DATAWIDTH),.ADDRWIDTHB(%(prefix)s_R_ADDRWIDTH),.SIZEB(%(prefix)s_R_DEPTH),.RAM_STYLE(\"%(ram_style)s\"))\n%(name)s_mem(.clkA(%(name)s_W.clk),.weA(%(name)s_W.we),.addrA(%(name)s_W.addr),.diA(%(name)s_W.din),.clkB(%(name)s_R.clk),.addrB(%(name)s_R.addr),.doB(%(name)s_R.dout));\n"
        ,"perbuswwith_en":"ifbram #(.ADDR_WIDTH(%(prefix)s_W_ADDRWIDTH),.DATA_WIDTH(%(prefix)s_W_DATAWIDTH)) %(name)s_W();\nifbram #(.ADDR_WIDTH(%(prefix)s_R_ADDRWIDTH),.DATA_WIDTH(%(prefix)s_R_DATAWIDTH)) %(name)s_R();\nbram_cfg %(name)s_R_cfg(.bram(%(name)s_R),.clk(lb3_clk),.rst(1'b0),.en(1'b1));\nasym_ram_sdp_write_wider #(.DATAWIDTHA(%(prefix)s_W_DATAWIDTH),.ADDRWIDTHA(%(prefix)s_W_ADDRWIDTH),.SIZEA(%(prefix)s_W_DEPTH),.DATAWIDTHB(%(prefix)s_R_DATAWIDTH),.ADDRWIDTHB(%(prefix)s_R_ADDRWIDTH),.SIZEB(%(prefix)s_R_DEPTH))\n%(name)s_mem(.clkA(%(name)s_W.clk),.enaA(%(name)s_W.en),.weA(%(name)s_W.we),.addrA(%(name)s_W.addr),.diA(%(name)s_W.din),.clkB(%(name)s_R.clk),.enaB(%(name)s_R.en),.addrB(%(name)s_R.addr),.doB(%(name)s_R.dout));\n"
        ,"perbusw":"ifbram #(.ADDR_WIDTH(%(prefix)s_W_ADDRWIDTH),.DATA_WIDTH(%(prefix)s_W_DATAWIDTH)) %(name)s_W();\nifbram #(.ADDR_WIDTH(%(prefix)s_R_ADDRWIDTH),.DATA_WIDTH(%(prefix)s_R_DATAWIDTH)) %(name)s_R();\nbram_cfg %(name)s_R_cfg(.bram(%(name)s_R),.clk(lb3_clk),.rst(1'b0),.en(1'b1));\nasym_ram_sdp_write_wider #(.INIT_FILE(INIT_%(name)s),.DATAWIDTHA(%(prefix)s_W_DATAWIDTH),.ADDRWIDTHA(%(prefix)s_W_ADDRWIDTH),.SIZEA(%(prefix)s_W_DEPTH),.DATAWIDTHB(%(prefix)s_R_DATAWIDTH),.ADDRWIDTHB(%(prefix)s_R_ADDRWIDTH),.SIZEB(%(prefix)s_R_DEPTH))\n%(name)s_mem(.clkA(%(name)s_W.clk),.weA(%(name)s_W.we),.addrA(%(name)s_W.addr),.diA(%(name)s_W.din),.clkB(%(name)s_R.clk),.addrB(%(name)s_R.addr),.doB(%(name)s_R.dout));\n"
        }
    ,"bram_portinst.vh":{
        "all":"%(perbus)s"
        ,"perbus":".%(NAME)s_clk(%(NAME)s_clk)\n,.%(NAME)s_rst(%(NAME)s_rst)\n,.%(NAME)s_addr(%(NAME)s_addr)\n,.%(NAME)s_din(%(NAME)s_din)\n,.%(NAME)s_dout(%(NAME)s_dout)\n,.%(NAME)s_en(%(NAME)s_en)\n,.%(NAME)s_we(%(NAME)s_we)\n"
        }
    ,"bram_port.vh":{
        "all":"%(perbus)s"
        ,"perbus":"output %(NAME)s_clk\n,output %(NAME)s_rst\n,output [BRAMADDRWIDTH-1:0] %(NAME)s_addr\n,output [%(prefix)s_DATAWIDTH-1:0] %(NAME)s_din\n,input [%(prefix)s_DATAWIDTH-1:0] %(NAME)s_dout\n,output %(NAME)s_en\n,output [%(prefix)s_DATAWIDTH/8-1:0] %(NAME)s_we\n"
        }
    ,"bram_range.tcl":{
        "all":"%(perbus)s"
        ,"perbus":"assign_bd_address -target_address_space /zynq_ultra_ps_e_0/Data [get_bd_addr_segs %(name)s_ctrl/S_AXI/Mem0] -force\nset_property range %(memsizekM)s [get_bd_addr_segs {zynq_ultra_ps_e_0/Data/SEG_%(name)s_ctrl_Mem0}]"
        }
    ,"adw":{
        "all":"%(perbus)s"
        ,"perbusw":"logic [%(prefix)s_W_DATAWIDTH-1:0] %(name)s_data;\nreg [%(prefix)s_W_ADDRWIDTH-1:0] %(name)s_addr=0;\nreg [%(prefix)s_W_DATAWIDTH/8-1:0] %(name)s_we=0;"
        ,"perbusr":"logic [%(prefix)s_R_DATAWIDTH-1:0] %(name)s_data;\nreg [%(prefix)s_R_ADDRWIDTH-1:0] %(name)s_addr=0;\nreg [%(prefix)s_R_DATAWIDTH/8-1:0] %(name)s_we=0;"
        }
    ,"write":{
        "all":"%(perbus)s"
        ,"perbus":"bram_cfg %(name)s_W_cfg(.bram(%(name)s_W),.clk(dspclk),.rst(1'b0),.en(1'b1));\nbram_write#(.ADDR_WIDTH(%(prefix)s_W_ADDRWIDTH),.DATA_WIDTH(%(prefix)s_W_DATAWIDTH))\n%(name)s_W_write(.bram(%(name)s_W),.addr(dspif.addr_%(subname)s),.data(dspif.data_%(subname)s),.we(dspif.we_%(subname)s));\n"
        }
    ,"read":{
        "all":"%(perbus)s"
        ,"perbus":"bram_cfg %(name)s_R_cfg(.bram(%(name)s_R),.clk(dspclk),.rst(1'b0),.en(1'b1));\nbram_read#(.ADDR_WIDTH(%(prefix)s_R_ADDRWIDTH),.DATA_WIDTH(%(prefix)s_R_DATAWIDTH))\n%(name)s_R_read(.bram(%(name)s_R),.addr(dspif.addr_%(subname)s),.data(dspif.data_%(subname)s));\n"
        }
    ,"ifbramctrl.sv":{
            "all":"interface ifbramctrl#(parameter integer DATA_WIDTH = 32,parameter integer ADDR_WIDTH=24,parameter READDELAY=4\n,`include \"bram_para.vh\"\n,`include \"braminit_para.vh\"\n\t)(iflocalbus.lb lb\n,`include \"bramif_lbport.vh\")\n;\nreg [DATA_WIDTH-1:0] rdata=0;\n%(wnames)s\n    always @(posedge lb.clk) begin\n %(walways)s\n end\n%(rnames)s\nalways @(posedge lb.clk) begin\n            %(ralways)s\nend\n    always @(posedge lb.clk) begin\n        if (lb.rden16[READDELAY]) begin\n            casex (lb.raddr16[READDELAY*ADDR_WIDTH-1:(READDELAY-1)*ADDR_WIDTH])\n            %(ralwayscase)s\n                default:rdata <= 32'hdeadbeef;\n            endcase\n        end\n    end\nassign lb.rdata=rdata;\nassign lb.rvalid=lb.rden16[READDELAY+1];\nassign lb.rvalidlast=lb.rdenlast16[READDELAY+1];\nendinterface"
            ,"wname":"reg %(namelbrw)s_we=0;reg [%(prefix)s_W_ADDRWIDTH-1:0]  %(namelbrw)s_waddr=0;reg [%(prefix)s_W_DATAWIDTH-1:0] %(namelbrw)s_din=0;assign {%(namelbrw)s.we,%(namelbrw)s.addr,%(namelbrw)s.din}={%(namelbrw)s_we,%(namelbrw)s_waddr,%(namelbrw)s_din};"
            ,"walways":"%(namelbrw)s_we<=(lb.waddr[ADDR_WIDTH-1:%(prefix)s_W_ADDRWIDTH]=='h%(addrcheck)x)&lb.wren; %(namelbrw)s_waddr<=lb.waddr[%(prefix)s_W_ADDRWIDTH-1:0]; %(namelbrw)s_din<=lb.wdata; // address: 0x%(address)08.0x"
            ,"rname":"reg [%(prefix)s_R_ADDRWIDTH-1:0]  %(namelbrw)s_raddr=0;reg [%(prefix)s_R_DATAWIDTH-1:0] %(namelbrw)s_dout=0;assign %(namelbrw)s.addr=%(namelbrw)s_raddr;"
            ,"ralways":"%(namelbrw)s_raddr<=lb.raddr[%(prefix)s_R_ADDRWIDTH-1:0];%(namelbrw)s_dout<=%(namelbrw)s.dout;"
            ,"ralwayscase":"{(ADDR_WIDTH-%(prefix)s_R_ADDRWIDTH)'('h%(addrcheck)x),{%(prefix)s_R_ADDRWIDTH{1'bx}}}: rdata <= %(namelbrw)s_dout;  // address: 0x%(address)08.0x"
        }
    ,"bram.json":{
            "all":"{\n%(perbus)s\n}"
            ,"perbus":"\"%(name)s\":{\"Awidth\":%(Awidth)d,\"Adepth\":%(Adepth)d,\"Bwidth\":%(Bwidth)d,\"latency\":%(latency)d,\"prefix\":\"%(prefix)s\",\"access\":\"%(access)s\",\"address\":\"0x%(address)08.0x\",\"ram_style\":\"%(ram_style)s\"}"
            }
    }

#    d1={"qdrvenv1":{"Awidth":512,"Adepth":1024,"Bwidth":512,"latency":2}}
    with open('gensrc/bram.json') as jsonfile:
        bramparadict=json.load(jsonfile)
    bs=brams(bramparadict=bramparadict,template=template)
    print(bs.brambus_tcl())
    print(bs.bramif_portinst_vh())
    print(bs.bramif_lbportinst_vh())
    print(bs.bramif_port_vh())
    print(bs.bramif_lbport_vh())
    print(bs.bram_parainst_vh())
    print(bs.bram_para_vh())
    print(bs.braminit_parainst_vh())
    print(bs.braminit_para_vh())
    print(bs.bram_plsv_vh())
    print(bs.bram_portinst_vh())
    print(bs.bram_port_vh())
    print(bs.bram_range_tcl())
    print(bs.bramwithctrl_tcl())
    print(bs.adw())
    print(bs.write())
    print(bs.read())
    print(bs.ifbramctrl_sv())
    print(bs.bram_json())
    print(bs.bramsiminit_vh())
    bs.assign()

    for k,(v,c,w) in bs.addressmap().items():
        print(k,format(v,'08x'),format(c,'08x'),w)


