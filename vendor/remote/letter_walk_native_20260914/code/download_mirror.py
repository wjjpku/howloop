"""Download identical HF weights via ModelScope only, with TLS and hash checks."""
import argparse
import concurrent.futures
import hashlib
import json
from pathlib import Path
import shutil
import socket
import time
from urllib.parse import urlsplit, urljoin
import requests

def main():
    p=argparse.ArgumentParser();p.add_argument('--hf',required=True);p.add_argument('--mirror',required=True);p.add_argument('--revision',required=True);p.add_argument('--output',type=Path,required=True);a=p.parse_args()
    assert socket.gethostname()=='lyg0232'
    original=socket.getaddrinfo;dns={}
    def allowed(h):return h=='modelscope.cn' or h.endswith('.modelscope.cn')
    def resolve(host,port,*args,**kwargs):
        if isinstance(host,str) and allowed(host):
            if host not in dns:
                s=requests.Session();s.trust_env=False
                r=s.get('https://dns.alidns.com/resolve',params={'name':host,'type':'A'},timeout=15);r.raise_for_status()
                dns[host]=next(x['data'] for x in r.json()['Answer'] if x['type']==1)
            host=dns[host]
        return original(host,port,*args,**kwargs)
    socket.getaddrinfo=resolve
    s=requests.Session();s.trust_env=False
    r=s.get(f'https://hf-mirror.com/api/models/{a.hf}/revision/{a.revision}',params={'blobs':'true'},timeout=30);r.raise_for_status();hf=r.json()
    assert hf['sha']==a.revision
    r=s.get(f'https://modelscope.cn/api/v1/models/{a.mirror}/repo/files',params={'Revision':'master','Recursive':'true'},timeout=30);r.raise_for_status()
    ms={x['Path']:x for x in r.json()['Data']['Files']}
    files=[]
    for f in hf['siblings']:
        n=f['rfilename']
        if n.endswith('.safetensors'):
            assert ms[n]['Sha256']==f['lfs']['sha256'] and ms[n]['Size']==f['lfs']['size']
            files.append(dict(name=n,size=ms[n]['Size'],sha256=ms[n]['Sha256'],revision=ms[n]['Revision']))
    assert shutil.disk_usage(a.output).free>sum(x['size'] for x in files)+5*2**30
    (a.output/'mirror_weight_manifest.json').write_text(json.dumps(dict(hf=a.hf,revision=a.revision,mirror=a.mirror,files=files),indent=2))
    def download(meta):
        n=meta['name'];dest=a.output/n;part=a.output/(n+'.ms-part')
        assert not dest.exists()
        url=f'https://modelscope.cn/models/{a.mirror}/resolve/{meta["revision"]}/{n}'
        ss=requests.Session();ss.trust_env=False
        for _ in range(8):
            assert urlsplit(url).scheme=='https' and allowed(urlsplit(url).hostname)
            response=ss.get(url,stream=True,allow_redirects=False,timeout=(20,60))
            if response.is_redirect:
                url=urljoin(url,response.headers['Location']);response.close();continue
            response.raise_for_status();break
        else:raise RuntimeError('redirect limit')
        h=hashlib.sha256();count=0;last=0
        with part.open('xb') as f:
            for b in response.iter_content(4*1024**2):
                f.write(b);h.update(b);count+=len(b)
                assert count<=meta['size']
                if time.time()-last>20:
                    print(json.dumps(dict(file=n,bytes=count,total=meta['size'])),flush=True);last=time.time()
        assert count==meta['size'] and h.hexdigest()==meta['sha256']
        part.rename(dest);print(json.dumps(dict(file=n,verified=True)),flush=True)
    with concurrent.futures.ThreadPoolExecutor(max_workers=2) as pool:list(pool.map(download,files))
    (a.output/'weights_verified.json').write_text(json.dumps(files,indent=2))
    print('ALL_WEIGHTS_VERIFIED',flush=True)
if __name__=='__main__':main()
