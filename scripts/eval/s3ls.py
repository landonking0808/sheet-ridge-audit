import sys, urllib.request, urllib.parse, re
B="https://vesuvius-challenge-open-data.s3.amazonaws.com/"
def ls(prefix, delim=None):
    out=[]; tok=None
    while True:
        q={"list-type":"2","prefix":prefix}
        if delim: q["delimiter"]=delim
        if tok: q["continuation-token"]=tok
        x=urllib.request.urlopen(B+"?"+urllib.parse.urlencode(q)).read().decode()
        out+= [(k,int(s)) for k,s in re.findall(r"<Key>([^<]*)</Key>.*?<Size>(\d+)</Size>",x)]
        out+= [(p,-1) for p in re.findall(r"<Prefix>([^<]*)</Prefix>",x)[1:]] if delim else []
        m=re.search(r"<NextContinuationToken>([^<]*)<",x)
        if not m: return out
        tok=m.group(1)
if __name__=="__main__":
    for k,s in ls(sys.argv[1], sys.argv[2] if len(sys.argv)>2 else None): print(s,k)
