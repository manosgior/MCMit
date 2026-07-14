import os
import re
import pathlib

def comment_remover(text): # https://stackoverflow.com/questions/241327/remove-c-and-c-comments-using-python
    def replacer(match):
        s = match.group(0)
        if s.startswith('/'):
            return " " # note: a space and not an empty string
        else:
            return s
    pattern = re.compile(
            r'//.*?$|/\*.*?\*/|\'(?:\\.|[^\\\'])*\'|"(?:\\.|[^\\"])*"',
            re.DOTALL | re.MULTILINE
            )
    return re.sub(pattern, replacer, text)

def svinclude(sorig,vhlist,oneline=False):
    cnt=0
    for vh in vhlist:
        basename=os.path.basename(vh)
        sre=r'`include\s+"%s"'%basename
        if re.findall(sre,sorig):
            with open(vh) as fvh:
                svh=fvh.read()
            if oneline:
                svh=comment_remover(svh)
                svh=re.sub("\n"," ",svh)
            sorig=re.sub(sre,svh,sorig)
            [cntnew,sorig]=svinclude(sorig,vhlist,oneline=oneline)
            cnt=cnt+cntnew+1
    return [cnt,sorig]
#def vhreplace(orig,vhlist):
#    vhre='|'.join([p for p in vhdict.keys()])
#    with open(orig) as f:
#        match=re.findall(vhre,f.read())
#    if match:
#        for s in match:
#            re.sub(s,vhdict[
#            print(orig,match,[vhdict[m] for m in match])
if __name__=="__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument(type=str, help='dependent', dest="dep")
    parser.add_argument( "-oneline","--oneline", dest='oneline',action='store_true', default=False,help="remove all \\n in vh")
    clargs = parser.parse_args()
    with open(clargs.dep,'r') as fd:
        sd=fd.read().split('\n')
    files=dict(v=[],sv=[],vh=[])
    for p in [l for l in sd if l]:
        if os.path.isfile(p):
            ext=pathlib.Path(p).suffix[1:]
            if ext not in files:
                files[ext]=[]
            files[ext].append(p)
        else:
            print("%s is not a file\n"%p)
    vsv=files['v']+files['sv']
    for index,f in enumerate(vsv):
        with open(f) as fread:
            orig=fread.read()
        cnt,srep=svinclude(orig,files['vh'],oneline=clargs.oneline)
        if cnt>0:
            fnew='./gensrc/'+os.path.basename(f)
            vsv[index]=fnew
            with open(fnew,'w') as fw:
                fw.write(srep)
    print('\n'.join(vsv))
