import os
import re

def svinclude(orig,vhlist):
    with open(orig) as forig:
        sorig=forig.read()
    for vh in vhlist:
        basename=os.path.basename(vh)
        sre=r'`include\s+"%s"'%basename            
        with open(vh) as fvh:
            svh=fvh.read()
        sreplace=re.sub(sre,svh,sorig)
        sorig=sreplace
        #print(basename,vh)
        #print(sreplace)
    return sreplace        
if __name__=="__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('-orig',type=str, help='original sv file', dest="orig",required=True)
    parser.add_argument('-vhfiles',type=str, nargs='+', help='vh files', dest="vhfiles",required=True)
    parser.add_argument('-output',type=str, help='output file', dest="output",required=True)
    clargs = parser.parse_args()
    with open(clargs.output,'w') as fout:
        fout.write(svinclude(clargs.orig,clargs.vhfiles))
    
    

