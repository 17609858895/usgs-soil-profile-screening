"""Download public inputs into the local (git-ignored) data directories."""
from pathlib import Path
import argparse,hashlib,json,urllib.request,shutil,zipfile
ROOT=Path(__file__).resolve().parents[1]
EXTERNAL=[
 ('mrds/mrds-csv.zip','https://mrdata.usgs.gov/mrds/mrds-csv.zip','31be4baaa86b082787bc74989146183db21badf7157db39b3f0a6fe0b38a5477'),
 ('epa_smelting/406370.pdf','https://semspub.epa.gov/work/HQ/406370.pdf','840deac165aa178d9aeb8c0114937315b1f5ad9da7c4765b0fdcb06a43357bab'),
 ('epa_coc/406315.xlsx','https://semspub.epa.gov/work/HQ/406315.xlsx','5abf990be2d2f7acdfa1f844e9fc176cbcc9c8556eb530f71b12fc3fbd443db6'),
]
TRI={1988:'9ed5aaa1bea62928930c91d9f627782cfa1fc1016eca5a8f4021f91e26d06ddd',1995:'85a8a8a6c55f7d8470cbf296d0f5f30ce3cbbaa8a56a77b9b27743a3fb4b65b1',2000:'a18a0f2a47c960b31fc6930ab2a564d454e635e3aaaab6ea1f8a01c8e713cebb',2005:'2e758db87e78e006170cbca31fb67b9fa5cf26b0410e3b6a3da9e15cfb848fb5',2010:'05328de22493791c1bea34690df97d3fffd6ae52d3d201f9fcaead15be42d1a2'}
def digest(p):
 h=hashlib.sha256()
 with p.open('rb') as f:
  for b in iter(lambda:f.read(1024*1024),b''):h.update(b)
 return h.hexdigest()
def download(url,path,sha):
 path.parent.mkdir(parents=True,exist_ok=True)
 if path.exists() and digest(path)==sha:
  print('verified existing',path.relative_to(ROOT));return
 tmp=path.with_suffix(path.suffix+'.part')
 with urllib.request.urlopen(url,timeout=120) as r,tmp.open('wb') as f:shutil.copyfileobj(r,f)
 actual=digest(tmp)
 if actual!=sha:raise RuntimeError(f'Input checksum changed: {url}; expected {sha}, obtained {actual}. Partial file retained for inspection.')
 tmp.replace(path);print('downloaded and verified',path.relative_to(ROOT))
def sources(external=False):
 for d in json.loads((ROOT/'config/ds801_downloads.json').read_text()):yield d['url'],ROOT/'data_raw'/d['filename'],d['sha256']
 if external:
  for name,url,sha in EXTERNAL:yield url,ROOT/'data_external'/name,sha
  for y,sha in TRI.items():yield f'https://data.epa.gov/efservice/downloads/tri/mv_tri_basic_download/{y}_US/csv',ROOT/'data_external/tri'/f'tri_{y}_us.csv',sha
def main():
 p=argparse.ArgumentParser(description=__doc__);p.add_argument('--external',action='store_true',help='Also download MRDS and EPA inputs (more than 350 MB).');p.add_argument('--check-links',action='store_true',help='Check each URL with a GET reading at most 1 KB.');a=p.parse_args()
 for url,dest,sha in sources(a.external):
  if a.check_links:
   try:
    with urllib.request.urlopen(urllib.request.Request(url,headers={'Range':'bytes=0-1023'}),timeout=40) as r:
     r.read(1024);print(r.status,url)
   except Exception as e:print(type(e).__name__,url,str(e))
  else:download(url,dest,sha)
 if a.external and not a.check_links:
  zpath=ROOT/'data_external/mrds/mrds-csv.zip'
  with zipfile.ZipFile(zpath) as z:
   for name in ['mrds.csv','mrds.met']:
    with z.open(name) as src,(zpath.parent/name).open('wb') as dst:shutil.copyfileobj(src,dst)
if __name__=='__main__':main()
