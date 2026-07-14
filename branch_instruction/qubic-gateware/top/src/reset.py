def reset(resetdict):
    NCFGRESETN=8
    NDSPRESETN=8
    NPSRESETN=3
    NADC2RESETN=3

    portvhtemplate="output %(name)sresetn%(index)d"
    portinstvhtemplate=".%(name)sresetn%(index)d(%(name)sresetn%(index)d)"
    plsv1template="reg [%(N)d-1:0] %(name)sresetn_r=0;"
#plsv2template="wire %(wirenames)s;"
    plsv3template="%(name)sresetn%(index)d"
    plsv4template="assign {%(wirenames)s}=%(name)sresetn_r;"
    plsv5template="always @(posedge %(name)sclk) begin %(name)sresetn_r<={%(N)d{~%(name)sreset}};end"
    paravhtemplate="parameter N%(NAME)sRESETN=%(N)d"

    portvh=[]
    portinstvh=[]
    plsv=[]
    paravh=[]
    for name,N in resetdict.items():
        plsv1=[]
        plsv2=[]
        plsv3=[]
        plsv4=[]
        plsv5=[]
        paradict=dict(name=name,N=N,NAME=name.upper())
        plsv1.append(plsv1template%paradict)
        for index in range(N):
            paradict.update(dict(index=index))
            portvh.append(portvhtemplate%paradict)
            portinstvh.append(portinstvhtemplate%paradict)
            plsv3.append(plsv3template%paradict)
        paradict.update(dict(wirenames=','.join(plsv3)))
#    plsv2.append(plsv2template%paradict)
        plsv4.append(plsv4template%paradict)
        plsv5.append(plsv5template%paradict)
        paravh.append(paravhtemplate%paradict)
        plsv.extend(plsv1+plsv4+plsv5)
#print('\n,'.join(portinstvh))
#print('\n'.join(plsv))
#print('\n,'.join(paravh))

    with open("gensrc/reset_port.vh",'w') as f:
        f.write('\n,'.join(portvh))
    with open("gensrc/reset_portinst.vh",'w') as f:
        f.write('\n,'.join(portinstvh))
    with open("gensrc/reset_plsv.vh",'w') as f:
        f.write('\n'.join(plsv))
    with open("gensrc/reset_para.vh",'w') as f:
        f.write('\n,'.join(paravh))

if __name__=="__main__":
    import json
    with open('reset.json') as jfile:
        resetdict=json.load(jfile)
#    resetdict=dict(cfg=8,dsp=14,ps=3,adc2=1)
        reset(resetdict)
