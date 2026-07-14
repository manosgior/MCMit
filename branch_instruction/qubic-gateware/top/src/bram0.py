import json
with open("gensrc/bram0.json") as jfile:
    bram0dict=json.load(jfile)
bramdict={}    
bramjson=[]
for name,bdict in bram0dict.items():
    cfg=bdict['cfg']
    for index in range(bdict['cnt']):
        bname='%s%d'%(name,index)
        bramdict[bname]=cfg
        bramjson.append("\"%s\":%s"%(bname,json.dumps(cfg)))
bramstr='{\n%s\n}'%('\n,'.join(bramjson))
with open('gensrc/bram.json','w') as f:
    f.write(bramstr)
        
#for name,bdict in bramdict        
#print(json.dumps(bramdict))

                       
